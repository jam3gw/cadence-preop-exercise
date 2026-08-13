#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1.0",
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Run a baseline triage model on prepared pre-op submission packages."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from core import (
    PatientSubmission,
    triage_submission,
)

ROOT = Path(__file__).resolve().parent

# Empty means "use the per-task configuration in rules.model_config",
# which is what the classifiers were tuned against. Passing --model
# overrides both classifiers with a single model id.
DEFAULT_MODEL = ""


@dataclass
class BaselineInputCase:
    case_id: str
    submission: PatientSubmission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "patients_sample_50.jsonl"),
        help="Input patient JSONL file",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "baseline_outputs.jsonl"),
        help="Output JSONL file with model responses",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model id",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent submissions (each makes its own model calls)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum number of records to process (0 means all)",
    )
    return parser.parse_args()


def load_env(path: Path) -> None:
    """Populate os.environ from a .env file without overwriting real env vars."""

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_cases(path: Path) -> list[BaselineInputCase]:
    cases: list[BaselineInputCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if (
                isinstance(payload, dict)
                and "submission" in payload
                and "case_id" in payload
            ):
                case = BaselineInputCase(
                    case_id=str(payload["case_id"]),
                    submission=PatientSubmission.model_validate(payload["submission"]),
                )
            else:
                case = BaselineInputCase(
                    case_id=f"case_{idx:05d}",
                    submission=PatientSubmission.model_validate(payload),
                )
            cases.append(case)
    return cases


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_env(ROOT / ".env")
    cases = load_cases(input_path)
    if args.max_records > 0:
        cases = cases[: args.max_records]

    def run_case(item: tuple[int, BaselineInputCase]) -> dict[str, Any]:
        idx, case = item
        submission = case.submission.model_dump()
        row: dict[str, Any] = {
            "record_index": idx,
            "case_id": case.case_id,
            "submission": submission,
            "model": args.model,
            "output": None,
            "error": None,
        }
        try:
            output = triage_submission(submission=submission, model=args.model or None)
            row["output"] = output.model_dump()
        except Exception as exc:  # pragma: no cover - network/runtime failure path
            row["error"] = str(exc)
        print(f"[{idx + 1}/{len(cases)}] {case.case_id}: "
              f"{row['output']['decision'] if row['output'] else 'ERROR'}")
        return row

    # Submissions are independent, so they run concurrently. Rows are written
    # back in input order regardless of completion order, keeping the output
    # file byte-stable across runs.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(run_case, enumerate(cases)))

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    print(f"Wrote baseline outputs -> {output_path}")


if __name__ == "__main__":
    main()
