"""Tests for the assembled triage pipeline.

These exercise `triage_submission` end to end with both classifiers injected,
so no network call is made and the rules are pinned against a known labelling.
The `no_network` fixture makes that guarantee explicit: constructing an OpenAI
client during any of these tests fails the test rather than quietly reaching
out.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core import PatientSubmission, TriageOutput, triage_submission
from rules.classifiers import ClassifiedDocumentModel, ClassifiedMedicationModel


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything in these tests tries to reach the API."""

    def explode() -> None:
        raise AssertionError("triage_submission attempted a live API call")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=explode))


def documents_as(**roles: str):
    """Classifier stub labelling each document by its `type`.

    Keyed on type so a test can express the labelling it wants without
    restating identifiers.
    """

    def classify(documents):
        out = []
        for index, doc in enumerate(documents):
            role = roles.get(doc.type or "", "OTHER")
            out.append(
                ClassifiedDocumentModel(
                    ref=index + 1,
                    role=role,
                    signed=True if role == "SURGICAL_CONSENT" else None,
                    plan_is_clear=True if role == "ANTICOAG_PLAN" else None,
                )
            )
        return out

    return classify


def medications_as(**classes: str):
    def classify(names):
        return [
            ClassifiedMedicationModel(
                name=name, medication_class=classes.get(name, "OTHER")
            )
            for name in names
        ]

    return classify


CLEAN_DOCS = documents_as(
    history_and_physical="H_AND_P", surgical_consent="SURGICAL_CONSENT"
)


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            },
            {"type": "temperature", "value_f": 98.6, "date": "2026-01-25"},
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "doc_id": "doc-1",
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "doc_id": "doc-2",
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def triage(payload, *, documents=CLEAN_DOCS, medications=medications_as()):
    return triage_submission(
        payload,
        document_classifier=documents,
        medication_classifier=medications,
    )


def test_clean_submission_is_ready(submission_payload) -> None:
    output = triage(submission_payload)

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []
    assert "satisfied" in output.explanation


def test_accepts_an_already_validated_submission(submission_payload) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    assert triage(submission).decision == "READY"


def test_acute_safety_exclusion_is_not_cleared(submission_payload) -> None:
    submission_payload["vitals"][0]["systolic"] = 184
    submission_payload["vitals"][0]["diastolic"] = 111

    output = triage(submission_payload)

    assert output.decision == "NOT_CLEARED"
    categories = {issue.category for issue in output.issues}
    assert categories == {"ACUTE_SAFETY_EXCLUSION"}
    # The evidence names the reading it was drawn from, not just the numbers.
    (issue,) = output.issues
    assert issue.evidence.source == "vitals[0]"
    assert "2026-01-25" in issue.evidence.details
    assert "184" in issue.evidence.details


def test_outdated_hp_needs_follow_up(submission_payload) -> None:
    submission_payload["documents"][0]["date"] = "2025-12-01"

    output = triage(submission_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.category for i in output.issues} == {"REQUIRED_DOCUMENTATION"}
    assert output.issues[0].evidence.source == "documents[0]"


def test_missing_procedure_date_is_reported_once(submission_payload) -> None:
    """The window rules defer to the missing-data check rather than piling on."""

    submission_payload["procedure"]["procedure_date"] = None

    output = triage(submission_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [i.category for i in output.issues] == ["MISSING_REQUIRED_DATA"]
    assert output.issues[0].evidence.source == "procedure.procedure_date"


def test_unrecognised_risk_does_not_escalate_past_follow_up(submission_payload) -> None:
    submission_payload["procedure"]["procedure_risk"] = "MEDIUM"

    output = triage(submission_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.category for i in output.issues} == {"MISSING_REQUIRED_DATA"}


def test_active_anticoagulant_without_a_plan(submission_payload) -> None:
    submission_payload["medications"] = [{"name": "apixaban", "active": True}]

    output = triage(submission_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.category for i in output.issues} == {"ANTICOAGULATION_MANAGEMENT"}


def test_anticoagulant_with_unknown_status_is_reported_not_assumed(
    submission_payload,
) -> None:
    """A null `active` is not read as taking, but is not swallowed either."""

    submission_payload["medications"] = [{"name": "warfarin", "active": None}]

    output = triage(submission_payload)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.category for i in output.issues} == {"MISSING_REQUIRED_DATA"}
    assert output.issues[0].evidence.source == "medications[0]"


def test_unidentifiable_medication_fails_toward_follow_up(submission_payload) -> None:
    """A drug nobody can classify must not silently clear the patient."""

    submission_payload["medications"] = [{"name": "zzzaban", "active": True}]

    # Classifier returns nothing for the name, as a refusal or omission would.
    output = triage(submission_payload, medications=lambda names: [])

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert {i.category for i in output.issues} == {"MISSING_REQUIRED_DATA"}


def test_safety_exclusion_outranks_other_findings(submission_payload) -> None:
    submission_payload["vitals"][1]["value_f"] = 101.0
    submission_payload["documents"][0]["date"] = "2025-12-01"

    output = triage(submission_payload)

    assert output.decision == "NOT_CLEARED"
    assert {i.category for i in output.issues} == {
        "REQUIRED_DOCUMENTATION",
        "ACUTE_SAFETY_EXCLUSION",
    }


def test_explanation_lists_every_issue_in_rule_order(submission_payload) -> None:
    submission_payload["procedure"]["procedure_date"] = None
    submission_payload["medications"] = [{"name": "apixaban", "active": True}]

    output = triage(submission_payload)

    assert output.explanation == (
        "MISSING_REQUIRED_DATA: Missing procedure date | "
        "ANTICOAGULATION_MANAGEMENT: Missing perioperative anticoagulation plan"
    )


def test_out_of_range_refs_are_discarded(submission_payload) -> None:
    """A label for a ref outside the batch cannot reach the rules."""

    def liar(documents):
        return [ClassifiedDocumentModel(ref=99, role="H_AND_P")]

    output = triage(submission_payload, documents=liar)

    # Every real document falls back to OTHER, so both requirements go missing.
    assert output.decision == "NEEDS_FOLLOW_UP"
    descriptions = {i.description for i in output.issues}
    assert descriptions == {
        "History and Physical document missing",
        "Signed surgical consent missing",
    }
