"""Rule 2 -- required pre-operative testing by procedure risk level.

    LOW or MODERATE risk: CBC within 30 days of the procedure date.
    HIGH risk:            CBC within 14 days and CMP within 14 days.
    Only the most recent result for each required test is considered.

Unlike Rule 1, this rule needs no language model. `labs[].code` is a
machine-generated value from a controlled vocabulary -- the sample data carries
only CBC, LAB-CBC, CMP, LAB-CMP, and HBA1C -- so matching it is exact-match
code, not classification. The free-text variation lives in `labs[].display`
("Complete Blood Count w/ Differential", "CBC (Complete Blood Count)", ...),
which this rule deliberately ignores.
"""

from __future__ import annotations

from datetime import date

from core import TriageIssue
from rules.base import (
    LABS_SOURCE,
    LabRef,
    RuleContext,
    build_issue,
    most_recent_lab,
)

CATEGORY = "REQUIRED_TESTING"
MISSING_DATA_CATEGORY = "MISSING_REQUIRED_DATA"

PROCEDURE_RISK_SOURCE = "procedure.procedure_risk"

CBC = "CBC"
CMP = "CMP"

#: Human-readable names, used only for issue prose.
TEST_NAMES = {CBC: "CBC", CMP: "CMP"}

#: Namespace prefixes some source systems prepend to an otherwise identical
#: code (`LAB-CBC` and `CBC` are the same test). Stripped before matching.
CODE_PREFIXES = ("LAB-",)

#: Required tests and their windows, in days before the procedure date.
TESTING_REQUIREMENTS: dict[str, dict[str, int]] = {
    "LOW": {CBC: 30},
    "MODERATE": {CBC: 30},
    "HIGH": {CBC: 14, CMP: 14},
}

#: The risk levels the policy defines. Anything else is uninterpretable input.
VALID_RISKS = frozenset(TESTING_REQUIREMENTS)


def normalize_lab_code(code: str | None) -> str | None:
    """Reduce a lab code to its bare form (`LAB-CBC` -> `CBC`)."""

    if not code:
        return None
    value = code.strip().upper()
    for prefix in CODE_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value or None


def normalize_risk(risk: str | None) -> str | None:
    if not risk or not risk.strip():
        return None
    return risk.strip().upper()


def evaluate(ctx: RuleContext) -> list[TriageIssue]:
    """Evaluate Rule 2 and return any testing issues found."""

    risk = normalize_risk(
        ctx.submission.procedure.procedure_risk if ctx.submission.procedure else None
    )

    # Both branches below are the same class of failure: without a risk level we
    # cannot know which tests are required, so the procedure's classification has
    # to be settled before this patient can be approved. The policy's stated
    # default for a required field being missing or unknown is NEEDS_FOLLOW_UP,
    # which any issue already implies -- neither case escalates further.
    if risk is None:
        return [
            build_issue(
                MISSING_DATA_CATEGORY,
                "Missing procedure risk level",
                source=PROCEDURE_RISK_SOURCE,
                details="procedure.procedure_risk is null",
            )
        ]

    if risk not in VALID_RISKS:
        return [
            build_issue(
                MISSING_DATA_CATEGORY,
                "Unrecognized procedure risk level",
                source=PROCEDURE_RISK_SOURCE,
                details=(
                    f"procedure.procedure_risk is {risk!r}; "
                    f"expected one of {', '.join(sorted(VALID_RISKS))}"
                ),
            )
        ]

    # Every window is measured against the procedure date. Without one the rule
    # is not evaluable; the missing-data rule owns reporting that.
    procedure_date = ctx.procedure_date
    if procedure_date is None:
        return []

    issues: list[TriageIssue] = []
    for code, window_days in TESTING_REQUIREMENTS[risk].items():
        issue = _check_test(ctx, code=code, window_days=window_days, procedure_date=procedure_date)
        if issue is not None:
            issues.append(issue)
    return issues


def _check_test(
    ctx: RuleContext,
    *,
    code: str,
    window_days: int,
    procedure_date: date,
) -> TriageIssue | None:
    """Check the most recent result for one required test against its window."""

    name = TEST_NAMES[code]
    matches = [ref for ref in ctx.labs if normalize_lab_code(ref.lab.code) == code]

    if not matches:
        return build_issue(
            CATEGORY,
            f"{name} missing",
            source=LABS_SOURCE,
            details=f"No {name} result found",
        )

    # "Only the most recent result for each required test shall be considered" --
    # an older in-window result does not rescue a newer out-of-window one.
    newest = most_recent_lab(matches)
    if newest is None:
        return build_issue(
            CATEGORY,
            f"{name} missing",
            source=LABS_SOURCE,
            details=f"No {name} result with a valid date found",
        )

    return _check_window(
        newest,
        name=name,
        window_days=window_days,
        procedure_date=procedure_date,
    )


def _check_window(
    ref: LabRef,
    *,
    name: str,
    window_days: int,
    procedure_date: date,
) -> TriageIssue | None:
    lab_date = ref.effective_date
    assert lab_date is not None  # guaranteed by most_recent_lab
    days_prior = (procedure_date - lab_date).days

    if days_prior < 0:
        # Resulted after the procedure date. Treated as a data problem rather
        # than a pass, consistent with Rule 1's handling of future-dated H&Ps.
        return build_issue(
            CATEGORY,
            f"{name} outside {window_days}-day window",
            source=ref.source,
            details=(
                f"{name} date {lab_date.isoformat()} is after procedure_date "
                f"{procedure_date.isoformat()} ({abs(days_prior)} days after); "
                f"must be resulted within {window_days} days before the procedure"
            ),
        )

    if days_prior > window_days:
        return build_issue(
            CATEGORY,
            f"{name} outside {window_days}-day window",
            source=ref.source,
            details=(
                f"{name} date {lab_date.isoformat()} vs procedure_date "
                f"{procedure_date.isoformat()} ({days_prior} days prior; "
                f"must be within {window_days})"
            ),
        )

    return None
