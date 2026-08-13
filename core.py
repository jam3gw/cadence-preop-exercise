"""Shared policy, schema, and prompt helpers for the pre-op triage scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import AliasChoices, BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rules.classifiers import DocumentClassifier, MedicationClassifier

# -------------------------
# Policy + system prompt
# -------------------------

BASELINE_SYSTEM_PROMPT = """
You are a clinical operations assistant for pre-op scheduling triage.
Use only the policy below. Do not use outside medical knowledge.

Cadence Surgical Center Pre-Operative Scheduling Policy (effective Jan 1, 2026)

Output exactly one status:
- READY
- NEEDS_FOLLOW_UP
- NOT_CLEARED

Rule 1: Required documentation
- History and Physical (H&P) must exist and be completed within 30 days of procedure date.
- Signed Surgical Consent must exist.
If documentation is missing/outdated -> NEEDS_FOLLOW_UP.

Rule 2: Required testing by procedure risk
- LOW or MODERATE risk: CBC within 30 days of procedure date.
- HIGH risk: CBC within 14 days and CMP within 14 days.
Use only the most recent result for each required test.
If a required test is missing or outside window -> NEEDS_FOLLOW_UP.

Rule 3: Anticoagulation management
If the patient is currently taking an anticoagulant, a perioperative anticoagulation plan must be documented and clear.
If no clear plan is documented -> NEEDS_FOLLOW_UP.

Rule 4: Acute safety exclusions
If any of the following are present at review time -> NOT_CLEARED:
- Systolic BP >= 180 mmHg
- Diastolic BP >= 110 mmHg
- Temperature > 100.4 F
Use the most recent relevant vital.

Final determination
- READY only if all required criteria are satisfied and no exclusions are present.
- If a required field needed to evaluate a rule is missing/unknown -> NEEDS_FOLLOW_UP.

Output requirements
- Return exactly one JSON object.
""".strip()

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    # Deliberately `str` rather than `ProcedureRisk`: an upstream system can
    # send a risk level we don't recognise, and that is a triage finding to be
    # reported, not a parse error that drops the whole submission. Rule 2 owns
    # validating this against `ProcedureRisk`.
    procedure_risk: str | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

class TriageIssueEvidence(BaseModel):

    source: str
    details: str

class TriageIssue(BaseModel):

    category: IssueCategory
    description: str
    evidence: TriageIssueEvidence

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str


class PreparedPatientCase(BaseModel):
    """Serialized eval case with submission payload and expected oracle output."""

    case_id: str
    submission: PatientSubmission
    expected_output: TriageOutput


def triage_output_json_schema() -> dict[str, object]:
    """Return the JSON schema used for structured model outputs."""

    schema = TriageOutput.model_json_schema()
    return schema

# -------------------------
# Rule evaluation primitives
# -------------------------
#
# A rule is a pure function over a `RuleContext`. These live here rather than
# under `rules/` so the dependency runs one way: rules import core, and core
# imports rule modules only inside the functions that call them.

# `documents[].type` is free text, so this enum is the stable vocabulary every
# rule works against.
DocumentRole = Literal[
    "H_AND_P",
    "SURGICAL_CONSENT",
    "ANTICOAG_PLAN",
    "OTHER",
]

# Resolved from `rules.medications` where possible, otherwise from a model.
# UNKNOWN is reported rather than assumed harmless: a spurious follow-up is an
# inconvenience, a missed anticoagulation plan is a safety event.
MedicationClass = Literal["ANTICOAGULANT", "OTHER", "UNKNOWN"]

# Evidence paths are dotted/indexed paths into the submission payload.
DOCUMENTS_SOURCE = "documents"
LABS_SOURCE = "labs"
MEDICATIONS_SOURCE = "medications"
VITALS_SOURCE = "vitals"


@dataclass(frozen=True)
class ClassifiedDocument:
    """A submission document paired with its resolved policy role."""

    index: int
    document: Document
    role: DocumentRole
    #: Whether the document text clearly indicates a signature. Only meaningful
    #: for consent documents; `None` means "not determinable".
    signed: bool | None = None
    #: Whether the document sets out a perioperative anticoagulation plan
    #: clearly enough to act on -- i.e. states how the medication is handled
    #: before and after the procedure. Only meaningful for ANTICOAG_PLAN
    #: documents; `None` means "not determinable".
    plan_is_clear: bool | None = None

    @property
    def doc_date(self) -> date | None:
        return parse_date(self.document.date)

    @property
    def source(self) -> str:
        """Evidence path pointing at this document within the submission."""

        return f"{DOCUMENTS_SOURCE}[{self.index}]"

    @property
    def text(self) -> str:
        return self.document.text or ""


@dataclass(frozen=True)
class LabRef:
    """A lab result paired with its position in the submission.

    `labs[].code` is a controlled vocabulary, so rules match on it directly.
    `display` is the free-text field and is not used for matching.
    """

    index: int
    lab: LabResult

    @property
    def effective_date(self) -> date | None:
        """Calendar date, used for the policy's day-count windows."""

        return parse_date(self.lab.effective_at)

    @property
    def effective_at(self) -> datetime | None:
        """Full instant, used only for ordering."""

        return parse_timestamp(self.lab.effective_at)

    @property
    def source(self) -> str:
        return f"{LABS_SOURCE}[{self.index}]"


@dataclass(frozen=True)
class VitalRef:
    """A vital sign paired with its position in the submission.

    Measurement fields are read with `getattr` rather than by isinstance: the
    `Vital` union resolves a reading missing its values to `GenericVital`,
    which a rule should see as an absent measurement, not the wrong branch.
    """

    index: int
    vital: Vital

    @property
    def vital_type(self) -> str | None:
        raw = self.vital.type
        return raw.strip().lower() if raw else None

    @property
    def measured_on(self) -> date | None:
        """Calendar date, for display."""

        return parse_date(self.vital.date)

    @property
    def measured_at(self) -> datetime | None:
        """Full instant, used for ordering."""

        return parse_timestamp(self.vital.date)

    @property
    def source(self) -> str:
        return f"{VITALS_SOURCE}[{self.index}]"

    def value(self, field: str) -> float | int | None:
        return getattr(self.vital, field, None)


@dataclass(frozen=True)
class ClassifiedMedication:
    """A submission medication paired with its resolved class."""

    index: int
    medication: Medication
    medication_class: MedicationClass

    @property
    def is_active(self) -> bool:
        """Whether the patient is recorded as currently taking this.

        Requires an explicit `True`; a null reads as inactive. Rule 3 still
        reports a null as MISSING_REQUIRED_DATA, so it is never just dropped.
        """

        return self.medication.active is True

    @property
    def source(self) -> str:
        return f"{MEDICATIONS_SOURCE}[{self.index}]"


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule is allowed to see."""

    submission: PatientSubmission
    documents: tuple[ClassifiedDocument, ...] = ()
    medications: tuple[ClassifiedMedication, ...] = ()

    @property
    def procedure_date(self) -> date | None:
        procedure = self.submission.procedure
        return parse_date(procedure.procedure_date) if procedure else None

    def documents_with_role(self, role: DocumentRole) -> list[ClassifiedDocument]:
        return [doc for doc in self.documents if doc.role == role]

    def medications_of_class(
        self, medication_class: MedicationClass
    ) -> list[ClassifiedMedication]:
        return [m for m in self.medications if m.medication_class == medication_class]

    def active_medications_of_class(
        self, medication_class: MedicationClass
    ) -> list[ClassifiedMedication]:
        return [
            med
            for med in self.medications
            if med.is_active and med.medication_class == medication_class
        ]

    @property
    def labs(self) -> list[LabRef]:
        return [LabRef(index=index, lab=lab) for index, lab in enumerate(self.submission.labs)]

    def vitals_of_type(self, vital_type: str) -> list[VitalRef]:
        return [
            VitalRef(index=index, vital=vital)
            for index, vital in enumerate(self.submission.vitals)
            if (vital.type or "").strip().lower() == vital_type
        ]


class Rule(Protocol):
    """Evaluate one policy rule and return the issues it found.

    A rule reports findings; it never decides the final status. Severity is the
    assembler's job and follows from issue category -- an acute safety exclusion
    means NOT_CLEARED, any other issue means NEEDS_FOLLOW_UP, none means READY.
    """

    def __call__(self, ctx: RuleContext) -> list[TriageIssue]: ...


def parse_date(value: str | None) -> date | None:
    """Parse a submission date field into a calendar date.

    Handles bare dates and RFC 3339 timestamps. Anything unparseable is treated
    as absent, which drives the rule toward follow-up rather than a false pass.
    """

    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a submission date field into an instant, for ordering.

    Separate from `parse_date` because collapsing a timestamp to a date makes
    two readings on the same day tie, silently picking whichever came first in
    the list. Bare dates anchor to midnight UTC so naive and aware values stay
    comparable.
    """

    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed_date = parse_date(raw)
        if parsed_date is None:
            return None
        parsed = datetime.combine(parsed_date, datetime.min.time())

    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def most_recent(documents: list[ClassifiedDocument]) -> ClassifiedDocument | None:
    """Return the newest dated document, or `None` if none carry a usable date.

    Ties break toward the earlier position in the submission so the choice is
    stable across runs.
    """

    dated = [doc for doc in documents if doc.doc_date is not None]
    if not dated:
        return None
    return max(dated, key=lambda doc: (doc.doc_date, -doc.index))


def most_recent_lab(labs: list[LabRef]) -> LabRef | None:
    """Return the newest lab result, or `None` if none carry a usable date.

    Ordered on the full timestamp, so two results on the same day resolve by
    time of day rather than tying. Genuine ties break toward the earlier
    position in the submission so the choice is stable across runs.
    """

    dated = [ref for ref in labs if ref.effective_at is not None]
    if not dated:
        return None
    return max(dated, key=lambda ref: (ref.effective_at, -ref.index))


def most_recent_vital(vitals: list[VitalRef]) -> VitalRef | None:
    """Return the newest dated vital, or `None` if none carry a usable date.

    Ties break toward the earlier position in the submission so the choice is
    stable across runs.
    """

    dated = [ref for ref in vitals if ref.measured_at is not None]
    if not dated:
        return None
    return max(dated, key=lambda ref: (ref.measured_at, -ref.index))


def describe_present(entries: list[str], *, noun: str, limit: int = 5) -> str:
    """Render a short inventory of what the submission does contain.

    An issue reporting something absent has no value of its own to quote, so
    naming what *is* on file distinguishes an empty chart from a missed lookup.
    """

    if not entries:
        return f"no {noun} present in the submission"
    shown = entries[:limit]
    suffix = f", and {len(entries) - limit} more" if len(entries) > limit else ""
    return f"{noun} on file: {'; '.join(shown)}{suffix}"


def build_issue(
    category: IssueCategory,
    description: str,
    *,
    source: str,
    details: str,
) -> TriageIssue:
    return TriageIssue(
        category=category,
        description=description,
        evidence=TriageIssueEvidence(source=source, details=details),
    )


# -------------------------
# Decision assembly
# -------------------------

# Only the acute safety rule can push the decision past follow-up.
BLOCKING_CATEGORY: IssueCategory = "ACUTE_SAFETY_EXCLUSION"

READY_EXPLANATION = (
    "All required documentation, testing, anticoagulation planning, "
    "and safety checks are satisfied."
)

PROCEDURE_DATE_SOURCE = "procedure.procedure_date"


def _missing_core_data(ctx: RuleContext) -> list[TriageIssue]:
    """Report submission-level fields that several rules depend on.

    Only the procedure date. Rules 1 and 2 both measure windows against it and
    stop when it is absent, so it is reported here once rather than by each.
    """

    if ctx.procedure_date is not None:
        return []
    return [
        build_issue(
            "MISSING_REQUIRED_DATA",
            "Missing procedure date",
            source=PROCEDURE_DATE_SOURCE,
            details="procedure.procedure_date is null",
        )
    ]


def decide(issues: list[TriageIssue]) -> Decision:
    """Derive the clearance status from the issues found.

    Severity follows from category, not from any rule asserting it.
    """

    if any(issue.category == BLOCKING_CATEGORY for issue in issues):
        return "NOT_CLEARED"
    if issues:
        return "NEEDS_FOLLOW_UP"
    return "READY"


def build_explanation(issues: list[TriageIssue]) -> str:
    if not issues:
        return READY_EXPLANATION
    return " | ".join(f"{issue.category}: {issue.description}" for issue in issues)


def build_context(
    submission: PatientSubmission,
    *,
    document_classifier: "DocumentClassifier | None" = None,
    medication_classifier: "MedicationClassifier | None" = None,
) -> RuleContext:
    """Resolve everything model-backed, once, before any rule runs.

    Both classifiers are injectable so rules can be exercised without an API key.
    """

    # Imported here rather than at module scope: `rules` depends on this
    # module's schema types, so a top-level import would be circular.
    from rules.classifiers import (
        LLMDocumentClassifier,
        LLMMedicationClassifier,
        classify_documents,
        resolve_medications,
    )

    documents = classify_documents(
        submission.documents,
        classifier=document_classifier or LLMDocumentClassifier(),
    )
    medications = resolve_medications(
        submission.medications,
        classifier=medication_classifier or LLMMedicationClassifier(),
    )
    return RuleContext(
        submission=submission,
        documents=tuple(documents),
        medications=tuple(medications),
    )


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str | None = None,
    document_classifier: "DocumentClassifier | None" = None,
    medication_classifier: "MedicationClassifier | None" = None,
) -> TriageOutput:
    """Triage one submission package against the pre-operative policy.

    Interpretation happens once up front; every rule below is a pure function
    over the resulting labels, with all issue prose templated in Python.

    `model` overrides both classifiers with a single id, for the harness's
    `--model` flag. Unset uses the per-task config in `rules.model_config`.
    """

    from rules.acute_safety import evaluate as evaluate_acute_safety
    from rules.anticoagulation import evaluate as evaluate_anticoagulation
    from rules.classifiers import LLMDocumentClassifier, LLMMedicationClassifier
    from rules.documentation import evaluate as evaluate_required_documentation
    from rules.model_config import ModelConfig
    from rules.required_testing import evaluate as evaluate_required_testing

    if isinstance(submission, PatientSubmission):
        parsed = submission
    else:
        parsed = PatientSubmission.model_validate(submission)

    if model:
        override = ModelConfig(model=model)
        document_classifier = document_classifier or LLMDocumentClassifier(config=override)
        medication_classifier = medication_classifier or LLMMedicationClassifier(config=override)

    ctx = build_context(
        parsed,
        document_classifier=document_classifier,
        medication_classifier=medication_classifier,
    )

    issues: list[TriageIssue] = [
        *_missing_core_data(ctx),
        *evaluate_required_documentation(ctx),
        *evaluate_required_testing(ctx),
        *evaluate_anticoagulation(ctx),
        *evaluate_acute_safety(ctx),
    ]

    return TriageOutput(
        decision=decide(issues),
        issues=issues,
        explanation=build_explanation(issues),
    )
