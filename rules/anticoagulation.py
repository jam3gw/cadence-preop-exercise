"""Rule 3 -- anticoagulation management.

Policy (Cadence Surgical Center, effective 2026-01-01):

    For patients currently taking anticoagulant medication, a documented
    perioperative anticoagulation management plan is required prior to
    scheduling. The plan must clearly describe how the medication will be
    managed before and after the procedure.

    No documented plan, or a plan that is incomplete or ambiguous ->
    NEEDS_FOLLOW_UP.

Two interpretive questions feed this rule, and both are resolved before it
runs: whether a medication is an anticoagulant (`rules.medications` first, then
the model) and whether a plan document is clear enough to act on (the document
classifier's `plan_is_clear`). Everything below is presence logic.
"""

from __future__ import annotations

from core import (
    DOCUMENTS_SOURCE,
    describe_present,
    ClassifiedDocument,
    ClassifiedMedication,
    RuleContext,
    TriageIssue,
    build_issue,
    most_recent,
)


CATEGORY = "ANTICOAGULATION_MANAGEMENT"
MISSING_DATA_CATEGORY = "MISSING_REQUIRED_DATA"


def evaluate(ctx: RuleContext) -> list[TriageIssue]:
    """Evaluate Rule 3 and return any anticoagulation issues found."""

    return [
        *_evaluate_unidentified(ctx),
        *_evaluate_unknown_status(ctx),
        *_evaluate_plan(ctx),
    ]


def _evaluate_unknown_status(ctx: RuleContext) -> list[TriageIssue]:
    """Report anticoagulants whose `active` status is unknown.

    The plan requirement is scoped to patients *currently* taking an
    anticoagulant, so a null `active` does not trigger it -- we do not demand a
    perioperative plan for a drug we cannot confirm the patient takes. But not
    knowing is itself a gap: `active` is a field the rule turns on, and the
    policy's default for a required field being missing or unknown is
    NEEDS_FOLLOW_UP. Someone has to ask the patient.
    """

    return [
        build_issue(
            MISSING_DATA_CATEGORY,
            "Unknown anticoagulant active status",
            source=med.source,
            details=(
                f"Medication {med.medication.name or 'unnamed'} has active=null; "
                "cannot determine if currently taking"
            ),
        )
        for med in ctx.medications_of_class("ANTICOAGULANT")
        if med.medication.active is None
    ]


def _evaluate_unidentified(ctx: RuleContext) -> list[TriageIssue]:
    """Report active medications nothing could classify.

    Whether the patient is on an anticoagulant is the field this rule turns on,
    so a drug neither the reference list nor the model could identify leaves the
    rule genuinely unevaluable -- NEEDS_FOLLOW_UP by the policy's own default.
    """

    unidentified = ctx.active_medications_of_class("UNKNOWN")
    return [
        build_issue(
            MISSING_DATA_CATEGORY,
            "Unable to determine whether medication is an anticoagulant",
            source=med.source,
            details=(
                f"{med.medication.name or 'unnamed medication'} is not in the "
                "anticoagulant reference list and could not be classified"
            ),
        )
        for med in unidentified
    ]


def _evaluate_plan(ctx: RuleContext) -> list[TriageIssue]:
    # The rule is scoped to patients *currently* taking an anticoagulant, so a
    # medication recorded as inactive or unconfirmed does not trigger it.
    anticoagulants = ctx.active_medications_of_class("ANTICOAGULANT")
    if not anticoagulants:
        return []

    plans = ctx.documents_with_role("ANTICOAG_PLAN")
    if not plans:
        return [
            build_issue(
                CATEGORY,
                "Missing perioperative anticoagulation plan",
                source=DOCUMENTS_SOURCE,
                details=_details(
                    anticoagulants,
                    f"no perioperative plan document found; {_documents_on_file(ctx)}",
                ),
            )
        ]

    # A plan only clears the rule if it actually says how the medication is
    # handled. "Incomplete or ambiguous" is a follow-up under the policy, and an
    # undeterminable plan (`None`) is ambiguous by definition.
    if any(plan.plan_is_clear is True for plan in plans):
        return []

    unclear = most_recent(plans) or plans[0]
    return [
        build_issue(
            CATEGORY,
            "Missing perioperative anticoagulation plan",
            source=unclear.source,
            details=_unclear_plan_details(anticoagulants, unclear),
        )
    ]


def _details(anticoagulants: list[ClassifiedMedication], problem: str) -> str:
    return f"{_medication_phrase(anticoagulants)} but {problem}"


def _unclear_plan_details(
    anticoagulants: list[ClassifiedMedication],
    plan: ClassifiedDocument,
) -> str:
    base = _details(anticoagulants, "no clear perioperative plan documented")
    text = plan.text.strip()
    return f"{base}: {text}" if text else base


def _medication_phrase(anticoagulants: list[ClassifiedMedication]) -> str:
    """Name the medications that triggered the rule, with their evidence paths.

    Naming both the drug and its index keeps the issue traceable back to the
    submission without the reader having to guess which entry was meant.
    """

    parts = [
        f"{med.medication.name or 'unnamed medication'} ({med.source})"
        for med in anticoagulants
    ]
    return f"Active anticoagulant medication present: {', '.join(parts)}"


def _documents_on_file(ctx: RuleContext) -> str:
    return describe_present(
        [
            f"{doc.document.type or 'untyped'}"
            + (f" ({doc.document.date})" if doc.document.date else "")
            for doc in ctx.documents
        ],
        noun="documents",
    )
