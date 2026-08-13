"""Rule 4 -- acute safety exclusion criteria.

Policy (Cadence Surgical Center, effective 2026-01-01):

    The presence of any of the following at the time of pre-operative review
    results in NOT_CLEARED:

      - Systolic blood pressure >= 180 mmHg
      - Diastolic blood pressure >= 110 mmHg
      - Temperature > 100.4 F

    Only the most recent result for each shall be considered.

No model is involved. `vitals[].type` is a machine-generated discriminator, so
each required vital is located by type, reduced to its most recent reading, and
compared against a fixed threshold.

Note the thresholds are not uniform: blood pressure excludes at or above its
limits, temperature only strictly above. That asymmetry is in the policy.
"""

from __future__ import annotations

from core import (
    VITALS_SOURCE,
    RuleContext,
    TriageIssue,
    VitalRef,
    build_issue,
    most_recent_vital,
)


CATEGORY = "ACUTE_SAFETY_EXCLUSION"
MISSING_DATA_CATEGORY = "MISSING_REQUIRED_DATA"

BLOOD_PRESSURE = "blood_pressure"
TEMPERATURE = "temperature"

SYSTOLIC_THRESHOLD = 180
DIASTOLIC_THRESHOLD = 110
TEMPERATURE_THRESHOLD = 100.4


def evaluate(ctx: RuleContext) -> list[TriageIssue]:
    """Evaluate Rule 4 and return any exclusion or missing-vital issues."""

    return [*_evaluate_blood_pressure(ctx), *_evaluate_temperature(ctx)]


def _evaluate_blood_pressure(ctx: RuleContext) -> list[TriageIssue]:
    latest = most_recent_vital(ctx.vitals_of_type(BLOOD_PRESSURE))
    if latest is None:
        return [_missing(BLOOD_PRESSURE, "Missing latest blood pressure")]

    systolic = latest.value("systolic")
    diastolic = latest.value("diastolic")
    if systolic is None and diastolic is None:
        return [_missing(BLOOD_PRESSURE, "Missing latest blood pressure")]

    # A single reading yields a single finding even when both numbers breach:
    # it is one exclusion, evidenced by one vital.
    breached = (systolic is not None and systolic >= SYSTOLIC_THRESHOLD) or (
        diastolic is not None and diastolic >= DIASTOLIC_THRESHOLD
    )
    if not breached:
        return []

    # The reading's date is part of the evidence, not decoration: these
    # patients carry several blood pressures, and "systolic=184" alone does not
    # say which one the exclusion was drawn from.
    return [
        build_issue(
            CATEGORY,
            "Blood pressure exceeds exclusion threshold",
            source=latest.source,
            details=(
                f"latest BP{_measured_on(latest)}: systolic={_format(systolic)}, "
                f"diastolic={_format(diastolic)}; threshold "
                f"systolic>={SYSTOLIC_THRESHOLD} or diastolic>={DIASTOLIC_THRESHOLD}"
            ),
        )
    ]


def _evaluate_temperature(ctx: RuleContext) -> list[TriageIssue]:
    latest = most_recent_vital(ctx.vitals_of_type(TEMPERATURE))
    if latest is None:
        return [_missing(TEMPERATURE, "Missing latest temperature")]

    value_f = latest.value("value_f")
    if value_f is None:
        return [_missing(TEMPERATURE, "Missing latest temperature")]

    if value_f <= TEMPERATURE_THRESHOLD:
        return []

    return [
        build_issue(
            CATEGORY,
            "Temperature exceeds exclusion threshold",
            source=latest.source,
            details=(
                f"latest temperature value_f={_format(value_f)}; "
                f"threshold is > {TEMPERATURE_THRESHOLD}"
            ),
        )
    ]


def _missing(vital_type: str, description: str) -> TriageIssue:
    """Report a vital the rule needs but cannot read.

    Rule 4 can only rule an exclusion *out* by looking at the measurement, so an
    absent or unusable vital leaves it unevaluated rather than satisfied -- the
    policy's default for a required field being missing or unknown.
    """

    return build_issue(
        MISSING_DATA_CATEGORY,
        description,
        source=VITALS_SOURCE,
        details=f"No {vital_type} vital with valid date and value found",
    )


def _measured_on(ref: VitalRef) -> str:
    """Render the reading's date, or nothing if it has none."""

    measured = ref.measured_on
    return f" on {measured.isoformat()}" if measured else ""


def _format(value: float | int | None) -> str:
    """Render a measurement exactly as recorded.

    No normalising of `101.0` to `101`: evidence quotes a measurement, and a
    reader comparing the issue against the submission should see the same value
    in both places.
    """

    return "unknown" if value is None else str(value)
