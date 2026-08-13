"""Shared primitives for policy rule evaluation.

A rule is a pure function over a `RuleContext`: it never touches the network and
never mutates its input. Every judgement that needs a language model happens once,
up front, in `rules.classifiers`, and lands in the context as a small enum label.

That split is deliberate. It keeps each rule cheap to unit test without an API
key, and it means the prose in an issue (dates, day counts, thresholds) is
templated in Python rather than generated, so repeated runs over the same
submission produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal, Protocol

from core import (
    Document,
    IssueCategory,
    LabResult,
    Medication,
    PatientSubmission,
    TriageIssue,
    TriageIssueEvidence,
    Vital,
)

# The roles the policy actually cares about. `documents[].type` is free text, so
# this enum is the stable vocabulary every rule works against.
DocumentRole = Literal[
    "H_AND_P",
    "SURGICAL_CONSENT",
    "ANTICOAG_PLAN",
    "OTHER",
]

# Whether a medication is an anticoagulant for the purposes of Rule 3. Resolved
# from `rules.medications` where possible, and only otherwise from a model.
# UNKNOWN is not a failure mode to be swallowed: a drug nobody could classify is
# reported rather than assumed harmless, because the two errors are not
# symmetric -- a spurious follow-up is an inconvenience, a missed
# anticoagulation plan is a patient safety event.
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

    Unlike `documents[].type`, `labs[].code` is a machine-generated value from a
    controlled vocabulary, so rules match on it directly rather than asking a
    model what a result is. `display` is the free-text field and is not used for
    matching.
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

    `vitals[].type` is a machine-generated discriminator, so it is matched
    directly. Measurement fields are read with `getattr` rather than by
    isinstance: the `Vital` union resolves a reading missing its values to
    `GenericVital`, and a rule should see that as an absent measurement rather
    than crash on the wrong branch of the union.
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

        Strictly `active is True`. A null `active` is not treated as "taking":
        the policy scopes Rule 3 to patients *currently* taking an
        anticoagulant, and an unconfirmed entry does not establish that.
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

    Handles both shapes present in submissions: bare dates (`2026-03-01`) on
    documents and RFC 3339 timestamps (`2026-03-01T09:05:00Z`) on labs and
    vitals. Anything unparseable is treated as absent, which drives the rule
    toward NEEDS_FOLLOW_UP rather than a false pass.
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

    `parse_date` deliberately returns a calendar date, because the policy's
    windows are counted in days and the issue prose quotes those day counts.
    Ordering is a different job: labs and vitals carry real timestamps
    (`2026-02-24T10:12:00Z`), and collapsing those to a date makes two readings
    on the same day tie, which would silently pick whichever came first in the
    list rather than the most recent one.

    Bare dates are anchored to midnight UTC so that naive and aware values
    remain mutually comparable.
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
