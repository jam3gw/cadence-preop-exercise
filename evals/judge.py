"""LLM judge for classifier evaluation.

Arbitrates the cases where a single asserted label would not be honest -- a
pre-operative assessment note that may or may not count as an H&P, an invented
drug name carrying a real drug-class suffix. Never used in the triage path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from evals.prompts import render
from rules.classifiers import _structured_call
from rules.model_config import ModelConfig, judge_config

Verdict = Literal[
    "EXPECTED_CORRECT",
    "CLASSIFIER_DEFENSIBLE",
    "CLASSIFIER_CORRECT",
    "BOTH_WRONG",
]


class JudgeVerdict(BaseModel):
    verdict: Verdict
    rationale: str


#: Verdicts under which a mismatch is not counted as a classifier failure.
ACCEPTABLE = {"CLASSIFIER_DEFENSIBLE", "CLASSIFIER_CORRECT"}


def judge(
    *,
    template: str,
    item: dict[str, object],
    expected: dict[str, object],
    actual: dict[str, object],
    note: str = "",
    config: ModelConfig | None = None,
) -> JudgeVerdict:
    return _structured_call(
        config=config or judge_config(),
        instructions=render("judge_system.j2"),
        prompt=render(template, item=item, expected=expected, actual=actual, note=note),
        text_format=JudgeVerdict,
    )
