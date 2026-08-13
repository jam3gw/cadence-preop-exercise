"""Per-task model selection.

Each model-backed step gets its own configuration rather than sharing one
global model id, because the tasks are not equally hard and should not cost the
same. Document role classification is largely a naming exercise over an
inconsistent vocabulary; deciding whether a drug is an anticoagulant can need
genuine pharmacological reasoning, and is the judgement where being wrong is a
patient safety event rather than an inconvenience.

Every setting is overridable by environment variable so a model can be swapped,
or an effort level tuned, without editing code -- see `from_env`.
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

        Reasoning models are driven by `reasoning.effort` and reject an
        explicit `temperature`, so the two are mutually exclusive. Where no
        effort is configured we pin `temperature=0`, which is the only
        determinism lever available on a non-reasoning model.
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

        An effort of "none" or "" disables reasoning and restores temperature=0,
        so a non-reasoning model can be substituted without a code change.
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
    """Documents: an inconsistent-naming problem more than a reasoning one."""

    return ModelConfig.from_env(
        "PREOP_DOCUMENT_CLASSIFIER",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
    )


def medication_classifier_config() -> ModelConfig:
    """Medications: worth a stronger model and more thinking.

    This only ever runs for names absent from `rules.medications`, so the
    volume is low and the cases that reach it are by definition the ones the
    curated list could not settle.
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
