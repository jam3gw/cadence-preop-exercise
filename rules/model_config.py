"""Per-task model selection.

Each model-backed step is configured separately rather than sharing one global
model id, because the tasks are not equally hard. Every setting is overridable
by environment variable so a model or effort level can change without a code
edit -- see `from_env`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class ModelConfig:
    """A model and the reasoning effort to run it at."""

    model: str
    reasoning_effort: ReasoningEffort | None = None

    def request_kwargs(self) -> dict[str, object]:
        """Request parameters for this configuration.

        Reasoning models reject an explicit `temperature`, so the two are
        mutually exclusive; without an effort we pin `temperature=0`.
        """

        if self.reasoning_effort is not None:
            return {"reasoning": {"effort": self.reasoning_effort}}
        return {"temperature": 0}

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ModelConfig:
        """Build a config, letting `<PREFIX>_MODEL` / `<PREFIX>_EFFORT` win.

        An effort of "none" or "" disables reasoning and restores temperature=0.
        """

        resolved_model = os.environ.get(f"{prefix}_MODEL", model)
        raw_effort = os.environ.get(f"{prefix}_EFFORT")
        if raw_effort is None:
            resolved_effort = reasoning_effort
        elif raw_effort.strip().lower() in {"", "none"}:
            resolved_effort = None
        else:
            resolved_effort = raw_effort.strip().lower()  # type: ignore[assignment]
        return cls(model=resolved_model, reasoning_effort=resolved_effort)


def document_classifier_config() -> ModelConfig:
    """Documents: mostly an inconsistent-naming problem, but not only that.

    High rather than medium to buy consistency: across three runs of the
    165-case fixture suite medium scored 163/160/162 and high scored 165/164/164.
    """

    return ModelConfig.from_env(
        "PREOP_DOCUMENT_CLASSIFIER",
        model="gpt-5.6-luna",
        reasoning_effort="high",
    )


def medication_classifier_config() -> ModelConfig:
    """Medications: worth a stronger model and more thinking.

    Only runs for names absent from `rules.medications`, so volume is low and
    every case reaching it is one the curated list could not settle.
    """

    return ModelConfig.from_env(
        "PREOP_MEDICATION_CLASSIFIER",
        model="gpt-5.6-terra",
        reasoning_effort="high",
    )


def judge_config() -> ModelConfig:
    """Evaluation judge. Never used in the triage path itself."""

    return ModelConfig.from_env(
        "PREOP_JUDGE",
        model="gpt-5.6-sol",
        reasoning_effort="medium",
    )
