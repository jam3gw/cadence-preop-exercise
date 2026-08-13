#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1.0",
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Evaluate the classifiers against labelled fixtures, with real model calls.

Most fixtures are plain assertions and never involve the judge. Only fixtures
marked `"judge": true` and any mismatch are sent to it -- the first because a
single asserted answer would be dishonest, the second to tell a genuine defect
from a fixture whose label deserves revisiting.

Evaluation tooling; nothing here is imported by the triage path.

    uv run evals/run_classifier_evals.py --suite all
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import Document  # noqa: E402
from evals.judge import ACCEPTABLE, judge  # noqa: E402
from rules.classifiers import (  # noqa: E402
    ClassifiedDocumentModel,
    ClassifiedMedicationModel,
    LLMDocumentClassifier,
    LLMMedicationClassifier,
)
from rules.model_config import (  # noqa: E402
    document_classifier_config,
    judge_config,
    medication_classifier_config,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_REPORT = ROOT / "data" / "classifier_eval_report.json"


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


def load_cases(*names: str) -> list[dict]:
    cases: list[dict] = []
    for name in names:
        path = DATA_DIR / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class Outcome:
    case_id: str
    item: dict
    expected: dict
    actual: dict | None
    matched: bool
    fields_wrong: list[str] = field(default_factory=list)
    judged: bool = False
    verdict: str | None = None
    rationale: str | None = None
    error: str | None = None

    @property
    def accepted(self) -> bool:
        if self.error is not None:
            return False
        # The judge cannot overturn a label the classifier and fixture share.
        if self.matched:
            return True
        return self.verdict in ACCEPTABLE


# --- Document suite ------------------------------------------------------

DOC_FIELDS = ("role", "signed", "plan_is_clear")


def run_documents(cases: list[dict], *, batch_size: int, workers: int) -> list[Outcome]:
    batches = chunked(cases, batch_size)

    def run_batch(batch: list[dict]) -> list[Outcome]:
        documents = [
            Document(
                doc_id=case["case_id"],  # carried for reporting, not for matching
                type=case.get("type"),
                date=None,
                author=case.get("author"),
                text=case.get("text"),
            )
            for case in batch
        ]
        # Fresh per batch: caches never hit across batches, and this avoids
        # sharing mutable state between threads.
        classifier = LLMDocumentClassifier()
        try:
            # Refs are 1-based positions in the batch we sent.
            results = {
                batch[r.ref - 1]["case_id"]: r
                for r in classifier(documents)
                if 1 <= r.ref <= len(batch)
            }
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return [
                Outcome(c["case_id"], c, c["expected"], None, False, error=str(exc))
                for c in batch
            ]

        outcomes: list[Outcome] = []
        for case in batch:
            expected = case["expected"]
            result = results.get(case["case_id"])
            if result is None:
                outcomes.append(
                    Outcome(
                        case["case_id"], case, expected, None, False,
                        error="classifier returned no label for this doc_id",
                    )
                )
                continue
            actual = _normalise_document(result)
            wrong = [f for f in DOC_FIELDS if expected.get(f) != actual.get(f)]
            outcomes.append(
                Outcome(case["case_id"], case, expected, actual, not wrong, wrong)
            )
        return outcomes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        nested = list(pool.map(run_batch, batches))
    return [outcome for group in nested for outcome in group]


def _normalise_document(result: ClassifiedDocumentModel) -> dict:
    """Apply the same role-scoping the production reconciler applies."""

    return {
        "role": result.role,
        "signed": result.signed if result.role == "SURGICAL_CONSENT" else None,
        "plan_is_clear": result.plan_is_clear if result.role == "ANTICOAG_PLAN" else None,
    }


# --- Medication suite ----------------------------------------------------


def run_medications(cases: list[dict], *, batch_size: int, workers: int) -> list[Outcome]:
    batches = chunked(cases, batch_size)

    def run_batch(batch: list[dict]) -> list[Outcome]:
        names = [case["name"] for case in batch]
        classifier = LLMMedicationClassifier()
        try:
            results = {r.name: r for r in classifier(names)}
        except Exception as exc:  # noqa: BLE001
            return [
                Outcome(c["case_id"], c, c["expected"], None, False, error=str(exc))
                for c in batch
            ]

        outcomes: list[Outcome] = []
        for case in batch:
            expected = case["expected"]
            result = results.get(case["name"])
            actual = _normalise_medication(result)
            # An empty name is expected to come back unclassified rather than
            # guessed, so absence is itself the correct answer there.
            if result is None and not case["name"].strip():
                outcomes.append(Outcome(case["case_id"], case, expected, actual, True))
                continue
            if result is None:
                outcomes.append(
                    Outcome(
                        case["case_id"], case, expected, None, False,
                        error="classifier returned no label for this name",
                    )
                )
                continue
            wrong = (
                []
                if expected["medication_class"] == actual["medication_class"]
                else ["medication_class"]
            )
            outcomes.append(
                Outcome(case["case_id"], case, expected, actual, not wrong, wrong)
            )
        return outcomes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        nested = list(pool.map(run_batch, batches))
    return [outcome for group in nested for outcome in group]


def _normalise_medication(result: ClassifiedMedicationModel | None) -> dict:
    return {"medication_class": result.medication_class if result else "UNCLASSIFIED"}


# --- Judging -------------------------------------------------------------


def apply_judge(outcomes: list[Outcome], *, template: str, workers: int) -> None:
    """Judge the cases that warrant it, in place."""

    pending = [
        o
        for o in outcomes
        if o.error is None
        and o.actual is not None
        and (o.expected.get("judge") or not o.matched)
    ]

    def run_one(outcome: Outcome) -> None:
        try:
            verdict = judge(
                template=template,
                item=outcome.item,
                expected=outcome.expected,
                actual=outcome.actual or {},
                note=str(outcome.expected.get("note") or ""),
            )
            outcome.judged = True
            outcome.verdict = verdict.verdict
            outcome.rationale = verdict.rationale
        except Exception as exc:  # noqa: BLE001
            outcome.error = f"judge failed: {exc}"

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_one, pending))


# --- Reporting -----------------------------------------------------------


def summarise(name: str, outcomes: list[Outcome], config) -> dict:
    total = len(outcomes)
    exact = sum(1 for o in outcomes if o.matched)
    accepted = sum(1 for o in outcomes if o.accepted)
    errors = [o for o in outcomes if o.error]
    # Classifier matched but the judge thinks the fixture is wrong. These pass,
    # but a silently mislabelled fixture is the worst failure a suite can have.
    disputed = [
        o
        for o in outcomes
        if o.matched and o.judged and o.verdict in {"BOTH_WRONG", "CLASSIFIER_CORRECT"}
    ]
    return {
        "suite": name,
        "model": {"model": config.model, "reasoning_effort": config.reasoning_effort},
        "cases": total,
        "exact_match": exact,
        "accepted_after_judging": accepted,
        "errors": len(errors),
        "fixture_labels_disputed_by_judge": [
            {
                "case_id": o.case_id,
                "label_both_agreed_on": o.expected,
                "verdict": o.verdict,
                "rationale": o.rationale,
            }
            for o in disputed
        ],
        "strict_accuracy_pct": round(100.0 * exact / total, 2) if total else 0.0,
        "adjusted_accuracy_pct": round(100.0 * accepted / total, 2) if total else 0.0,
        "failures": [
            {
                "case_id": o.case_id,
                "item": {k: v for k, v in o.item.items() if k not in {"expected"}},
                "expected": o.expected,
                "actual": o.actual,
                "fields_wrong": o.fields_wrong,
                "verdict": o.verdict,
                "rationale": o.rationale,
                "error": o.error,
            }
            for o in outcomes
            if not o.accepted
        ],
        "judged": [
            {
                "case_id": o.case_id,
                "verdict": o.verdict,
                "rationale": o.rationale,
                "expected": o.expected,
                "actual": o.actual,
            }
            for o in outcomes
            if o.judged
        ],
    }


def print_summary(report: dict) -> None:
    print(f"\n=== {report['suite']} ===")
    model = report["model"]
    print(f"  model: {model['model']} (reasoning: {model['reasoning_effort']})")
    print(f"  cases: {report['cases']}")
    print(f"  exact match:            {report['exact_match']}  ({report['strict_accuracy_pct']}%)")
    print(f"  accepted after judging: {report['accepted_after_judging']}  ({report['adjusted_accuracy_pct']}%)")
    if report["errors"]:
        print(f"  ERRORS: {report['errors']}")
    disputed = report["fixture_labels_disputed_by_judge"]
    if disputed:
        print(f"  fixture labels the judge disputes: {len(disputed)} "
              f"(classifier matched, but review the fixture)")
        for item in disputed:
            print(f"    {item['case_id']}: {item['verdict']} -- {item['rationale']}")
    for failure in report["failures"]:
        label = failure["item"].get("type") or failure["item"].get("name") or failure["case_id"]
        print(f"\n  FAIL {failure['case_id']}: {label!r}")
        if failure["error"]:
            print(f"       error: {failure['error']}")
            continue
        fields = failure["fields_wrong"] or sorted(failure["actual"] or {})
        print(f"       expected: { {k: failure['expected'].get(k) for k in fields} }")
        print(f"       actual:   { {k: (failure['actual'] or {}).get(k) for k in fields} }")
        if failure["verdict"]:
            print(f"       judge: {failure['verdict']} -- {failure['rationale']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["documents", "medications", "all"], default="all")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="cap cases per suite")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (checked environment and .env)")

    reports = []

    if args.suite in {"documents", "all"}:
        cases = load_cases("document_cases.jsonl", "document_cases_synthetic.jsonl")
        if args.limit:
            cases = cases[: args.limit]
        print(f"Running {len(cases)} document cases "
              f"({len(chunked(cases, args.batch_size))} batches, {args.workers} workers)...")
        outcomes = run_documents(cases, batch_size=args.batch_size, workers=args.workers)
        if not args.no_judge:
            apply_judge(outcomes, template="judge_document.j2", workers=args.workers)
        reports.append(summarise("documents", outcomes, document_classifier_config()))

    if args.suite in {"medications", "all"}:
        cases = load_cases("medication_cases.jsonl")
        if args.limit:
            cases = cases[: args.limit]
        print(f"Running {len(cases)} medication cases "
              f"({len(chunked(cases, args.batch_size))} batches, {args.workers} workers)...")
        outcomes = run_medications(cases, batch_size=args.batch_size, workers=args.workers)
        if not args.no_judge:
            apply_judge(outcomes, template="judge_medication.j2", workers=args.workers)
        reports.append(summarise("medications", outcomes, medication_classifier_config()))

    for report in reports:
        print_summary(report)

    judge_cfg = judge_config()
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "judge_model": {
            "model": judge_cfg.model,
            "reasoning_effort": judge_cfg.reasoning_effort,
        },
        "suites": reports,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote report -> {report_path}")

    if any(r["failures"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
