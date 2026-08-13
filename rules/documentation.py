"""Rule 1 -- Required documentation.

Policy (Cadence Surgical Center, effective 2026-01-01):

    1. History and Physical (H&P) must exist and be completed within 30 days of
       the scheduled procedure date.
    2. A signed Surgical Consent must exist.

    Missing or outdated documentation -> NEEDS_FOLLOW_UP.

Document roles arrive pre-resolved on the context (see `rules.classifiers`);
everything below is deterministic date and presence logic.
"""

from __future__ import annotations

from core import (
    DOCUMENTS_SOURCE,
    ClassifiedDocument,
    RuleContext,
    TriageIssue,
    build_issue,
    describe_present,
    most_recent,
)


CATEGORY = "REQUIRED_DOCUMENTATION"
HP_WINDOW_DAYS = 30


def evaluate(ctx: RuleContext) -> list[TriageIssue]:
    """Evaluate Rule 1 and return any documentation issues found."""

    return [*_evaluate_history_and_physical(ctx), *_evaluate_surgical_consent(ctx)]


def _evaluate_history_and_physical(ctx: RuleContext) -> list[TriageIssue]:
    candidates = ctx.documents_with_role("H_AND_P")

    # No usable date cannot satisfy a 30-day window, so it counts as missing.
    hp = most_recent(candidates)
    if hp is None:
        return [
            build_issue(
                CATEGORY,
                "History and Physical document missing",
                source=DOCUMENTS_SOURCE,
                details=(
                    "No History and Physical document with a valid date found; "
                    f"{_documents_on_file(ctx)}"
                ),
            )
        ]

    procedure_date = ctx.procedure_date
    if procedure_date is None:
        # Not evaluable. core.py reports the missing date; repeating it here
        # would double-count one root cause.
        return []

    assert hp.doc_date is not None  # guaranteed by most_recent
    days_prior = (procedure_date - hp.doc_date).days

    if days_prior < 0:
        # Dated after the procedure: unusable rather than silently accepted.
        return [
            build_issue(
                CATEGORY,
                "H&P outside 30-day window",
                source=hp.source,
                details=(
                    f"H&P date {hp.doc_date.isoformat()} is after procedure_date "
                    f"{procedure_date.isoformat()} ({-days_prior} days after); "
                    f"must be completed within {HP_WINDOW_DAYS} days before the procedure"
                ),
            )
        ]

    if days_prior > HP_WINDOW_DAYS:
        return [
            build_issue(
                CATEGORY,
                "H&P outside 30-day window",
                source=hp.source,
                details=(
                    f"H&P date {hp.doc_date.isoformat()} vs procedure_date "
                    f"{procedure_date.isoformat()} ({days_prior} days prior; "
                    f"must be within {HP_WINDOW_DAYS})"
                ),
            )
        ]

    return []


def _evaluate_surgical_consent(ctx: RuleContext) -> list[TriageIssue]:
    consents = ctx.documents_with_role("SURGICAL_CONSENT")
    if not consents:
        return [
            build_issue(
                CATEGORY,
                "Signed surgical consent missing",
                source=DOCUMENTS_SOURCE,
                details=f"No Surgical Consent document found; {_documents_on_file(ctx)}",
            )
        ]

    # Only an affirmative signature clears a *signed* consent requirement.
    if any(consent.signed is True for consent in consents):
        return []

    unsigned = most_recent(consents) or consents[0]
    return [
        build_issue(
            CATEGORY,
            "Surgical consent not clearly signed",
            source=unsigned.source,
            details=_unsigned_consent_details(unsigned),
        )
    ]


def _unsigned_consent_details(consent: ClassifiedDocument) -> str:
    base = "Consent document text does not clearly indicate signed consent"
    text = consent.text.strip()
    return f"{base}: {text}" if text else base


def _documents_on_file(ctx: RuleContext) -> str:
    return describe_present(
        [
            f"{doc.document.type or 'untyped'}"
            + (f" ({doc.document.date})" if doc.document.date else "")
            for doc in ctx.documents
        ],
        noun="documents",
    )
