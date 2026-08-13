#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic>=2.8.0",
# ]
# ///

"""Derive a labelled document fixture from the repo's own sample data.

The point of evaluating against `data/patients_sample_50.jsonl` rather than
invented examples is that these are the exact strings the classifier will meet
in production: 143 distinct (type, text) pairs, spanning every abbreviation,
prefix, suffix and misspelling the source systems produce.

Labels here are assigned by matching the small set of body-text templates the
dataset is built from, which is reliable precisely because it is the *text*
being matched rather than the free-text `type`. Anything this cannot label
confidently is emitted with `"label_confidence": "low"` and flagged for review
rather than silently guessed.

The output is committed as data. Regenerate with:

    uv run evals/build_fixtures.py

and review the diff -- this file is a labelling heuristic, not an oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "patients_sample_50.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "document_cases.jsonl"

# --- Role signals --------------------------------------------------------
# Consent and anticoagulation plans are identifiable from body text. H&P is
# not, and this is the subtlety worth understanding before trusting a label:
#
# The dataset pairs every case's current H&P with an older one whose body reads
# "Prior pre-op H&P retained for longitudinal chart context." That same body
# text is attached to *two different kinds of document*: titles that are plainly
# an H&P ("PREOP - Pre-op H and P", "Scanned History and Physical Examination")
# and titles that are plainly something else ("Medical Clearance [PDF]",
# "Preop Pre-anesthesia Evaluation [PDF]").
#
# So the text cannot decide it. A retained prior H&P is still an H&P -- it just
# loses on recency, which is Rule 1's job, not the classifier's. H&P-ness is
# therefore judged on `type`, and the sample oracle agrees: in case_00002 it
# selects exactly such a retained document as the H&P.

HP_TYPE = re.compile(
    r"h\s*&\s*p|h\s*\+\s*p|h\s*/\s*p|\bh and p\b|hx\s*&\s*phys|"
    r"hist\s*&\s*phys|history\s*[&/]\s*physical|history and physical|"
    r"phsyical|physical exam",
    re.I,
)

# Types that are clinical documents but not a history and physical.
NON_HP_TYPE = re.compile(
    r"nursing intake|anesthesia pre-assessment|clinic follow-up", re.I
)

# Types where reasonable clinicians could disagree: a pre-operative assessment
# or clearance note may or may not constitute an H&P depending on the site's
# conventions. These are routed to the judge rather than asserted.
AMBIGUOUS_TYPE = re.compile(
    r"pre-anesthesia evaluation|medical clearance|surgical clearance|"
    r"preoperative assessment|pre-?op evaluation",
    re.I,
)

CONSENT_SIGNED_TEXT = re.compile(
    r"consent obtained and signed|signed surgical consent|"
    r"signed consent scanned and verified|consent obtained; signature on file|"
    r"electronic consent obtained and signed",
    re.I,
)
CONSENT_UNSIGNED_TEXT = re.compile(
    r"unsigned|awaiting patient signature|signature not yet on file|"
    r"requested signature before scheduling",
    re.I,
)

ANTICOAG_TEXT = re.compile(
    r"anticoagulant noted in medication list|anticoagulation mentioned|"
    r"apixaban listed|takes apixaban|discussed blood thinner",
    re.I,
)
ANTICOAG_UNCLEAR_TEXT = re.compile(
    r"not yet documented|to be finalized|pending specialist input|"
    r"no clear hold/resume|follow up with cardiology",
    re.I,
)

OTHER_TEXT = re.compile(
    r"Nursing intake confirms|Anesthesia screening completed|"
    r"Routine clinic follow-up note",
    re.I,
)


def _labelled(
    role: str,
    *,
    signed: bool | None = None,
    plan_is_clear: bool | None = None,
    judge: bool = False,
    note: str = "",
) -> dict[str, object]:
    return {
        "role": role,
        "signed": signed,
        "plan_is_clear": plan_is_clear,
        # Exact-match cases assert; judged cases are arbitrated by the judge
        # model, because there is no single defensible answer to assert.
        "judge": judge,
        "note": note,
    }


def label(doc_type: str, text: str) -> dict[str, object]:
    """Assign an expected label to one (type, text) pair."""

    # Anticoagulation and consent are settled by body text, which states the
    # thing directly, so these are checked before any type-based reasoning.
    if ANTICOAG_TEXT.search(text):
        return _labelled(
            "ANTICOAG_PLAN",
            plan_is_clear=not bool(ANTICOAG_UNCLEAR_TEXT.search(text)),
        )

    if CONSENT_SIGNED_TEXT.search(text) or CONSENT_UNSIGNED_TEXT.search(text):
        return _labelled(
            "SURGICAL_CONSENT",
            signed=not bool(CONSENT_UNSIGNED_TEXT.search(text)),
        )

    # H&P is decided on the title. A retained prior H&P is still an H&P; being
    # out of date is Rule 1's problem, not the classifier's.
    if HP_TYPE.search(doc_type):
        return _labelled("H_AND_P")

    if AMBIGUOUS_TYPE.search(doc_type):
        return _labelled(
            "OTHER",
            judge=True,
            note=(
                "assessment/clearance note with no H&P in the title -- "
                "defensible either way, judged rather than asserted"
            ),
        )

    if NON_HP_TYPE.search(doc_type) or OTHER_TEXT.search(text):
        return _labelled("OTHER")

    return _labelled(
        "OTHER",
        judge=True,
        note="no confident signal from type or text -- judged",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    seen: dict[tuple[str, str], dict[str, object]] = {}
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for doc in row.get("submission", {}).get("documents") or []:
            doc_type = (doc.get("type") or "").strip()
            text = (doc.get("text") or "").strip()
            key = (doc_type, text)
            if key in seen:
                continue
            seen[key] = {
                "case_id": f"doc_{len(seen):03d}",
                "origin": "sample_data",
                "type": doc_type,
                "author": doc.get("author"),
                "text": text,
                "expected": label(doc_type, text),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in seen.values():
            handle.write(json.dumps(case, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    judged = [c for c in seen.values() if c["expected"]["judge"]]
    roles: dict[str, int] = {}
    for case in seen.values():
        role = str(case["expected"]["role"])
        roles[role] = roles.get(role, 0) + 1

    print(f"Wrote {len(seen)} cases -> {output}")
    print(f"  roles: {roles}")
    print(f"  asserted: {len(seen) - len(judged)}   judged: {len(judged)}")
    for case in judged:
        print(f"    judged: {case['type']!r}")


if __name__ == "__main__":
    main()
