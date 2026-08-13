"""LLM-backed classification of the free-text parts of a submission.

Two things in a submission resist deterministic interpretation:

`documents[].type` is free text produced by many source systems ("PREOP - H and
P - signed", "Scanned Hist & Phys (H&P) (external)", "Imported: Consult H&P",
...). Treating it as an enum is brittle, so the whole document list for a case
goes to the model in one call, which labels each `doc_id` with the policy role
it serves.

Medication names are an open vocabulary with no reference data in this repo.
Those are resolved against the curated list in `rules.medications` first, and
only names it does not know reach the model -- see `resolve_medications`.

In both cases the model returns nothing but small enum labels keyed by an
identifier we gave it, never an index and never prose that reaches an issue.
Returned keys are validated against the real input before use, so a
hallucinated id cannot corrupt an evidence path, and every rule's phrasing
stays deterministic.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from core import Document, Medication
from rules import medications as medication_reference
from rules.base import (
    ClassifiedDocument,
    ClassifiedMedication,
    DocumentRole,
    MedicationClass,
)
from rules.prompts import render

CLASSIFIER_MODEL = "gpt-4.1-mini"

SYSTEM_TEMPLATE = "document_classifier_system.j2"
USER_TEMPLATE = "document_classifier_user.j2"
MEDICATION_SYSTEM_TEMPLATE = "medication_classifier_system.j2"
MEDICATION_USER_TEMPLATE = "medication_classifier_user.j2"

#: Document bodies are short, but cap them so one pathological record can't
#: blow up the prompt.
MAX_TEXT_CHARS = 2000


class ClassificationRefusedError(RuntimeError):
    """The model declined to produce a classification."""


class ClassifiedDocumentModel(BaseModel):
    """One document's classification as returned by the model."""

    doc_id: str
    role: Literal["H_AND_P", "SURGICAL_CONSENT", "ANTICOAG_PLAN", "OTHER"]
    signed: bool | None = None
    plan_is_clear: bool | None = None


class DocumentClassificationResult(BaseModel):
    classifications: list[ClassifiedDocumentModel] = Field(default_factory=list)


class ClassifiedMedicationModel(BaseModel):
    """One medication name's classification as returned by the model."""

    name: str
    medication_class: Literal["ANTICOAGULANT", "OTHER"]


class MedicationClassificationResult(BaseModel):
    classifications: list[ClassifiedMedicationModel] = Field(default_factory=list)


ResultT = TypeVar("ResultT", bound=BaseModel)


def _structured_call(
    *,
    model: str,
    instructions: str,
    prompt: str,
    text_format: type[ResultT],
) -> ResultT:
    """One Structured Outputs call, returning a validated instance.

    `responses.parse` with `text_format` is genuine Structured Outputs: the SDK
    derives a strict JSON schema from the model (every property required,
    additionalProperties false) and the API constrains decoding to it, so a
    schema violation is impossible rather than merely unlikely.
    """

    # Import lazily so this module stays importable without the openai package
    # installed (e.g. in unit tests that inject a fake classifier).
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        instructions=instructions,
        temperature=0,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text_format=text_format,
    )

    result = response.output_parsed
    if result is None:
        # Only reachable if the model refused. There is no partial output worth
        # salvaging, and guessing would be worse than failing.
        raise ClassificationRefusedError(
            f"Model did not return a {text_format.__name__} "
            f"(status={response.status!r})"
        )
    return result


def build_classifier_payload(documents: list[Document]) -> list[dict[str, object]]:
    """Model-facing view of the documents.

    Deliberately omits `date`: recency is the rules' job, and withholding dates
    keeps the model from quietly deciding a document is "too old to count"
    instead of just naming what it is.
    """

    return [
        {
            "doc_id": doc.doc_id,
            "type": doc.type,
            "author": doc.author,
            "text": (doc.text or "")[:MAX_TEXT_CHARS],
        }
        for doc in documents
    ]


def build_classifier_system_prompt() -> str:
    return render(SYSTEM_TEMPLATE)


def build_classifier_user_prompt(documents: list[Document]) -> str:
    return render(USER_TEMPLATE, documents=build_classifier_payload(documents))


class DocumentClassifier(Protocol):
    """Classifies a case's documents by policy role.

    Implementations must be deterministic given the same input (temperature=0
    or equivalent) so that repeated evaluation of one submission is stable.
    """

    def __call__(self, documents: list[Document]) -> list[ClassifiedDocumentModel]: ...


class LLMDocumentClassifier:
    """Default classifier: a single structured-output call per case.

    Results are memoized on a hash of the exact model input, so re-evaluating
    the same submission (as the determinism harness does) costs one API call
    and returns an identical labelling every time.
    """

    def __init__(self, *, model: str = CLASSIFIER_MODEL) -> None:
        self.model = model
        self._cache: dict[str, list[ClassifiedDocumentModel]] = {}

    def __call__(self, documents: list[Document]) -> list[ClassifiedDocumentModel]:
        if not documents:
            return []

        prompt = build_classifier_user_prompt(documents)
        cache_key = hashlib.sha256(
            f"{self.model}\n{prompt}".encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = _structured_call(
            model=self.model,
            instructions=build_classifier_system_prompt(),
            prompt=prompt,
            text_format=DocumentClassificationResult,
        )
        self._cache[cache_key] = result.classifications
        return result.classifications


def classify_documents(
    documents: list[Document],
    *,
    classifier: DocumentClassifier,
) -> list[ClassifiedDocument]:
    """Run `classifier` and reconcile its output against the real document list.

    Any `doc_id` the model returns that we didn't hand it is dropped (a
    hallucinated id, not a hallucinated role). Any document the model fails to
    classify falls back to OTHER rather than being dropped, so every input
    document is always represented in the output, at its original index.
    """

    if not documents:
        return []

    known_ids = {doc.doc_id for doc in documents}

    labels: dict[str | None, tuple[DocumentRole, bool | None, bool | None]] = {}
    for item in classifier(documents):
        if item.doc_id not in known_ids:
            continue
        # Each judgement is only meaningful for the role it belongs to; forcing
        # the others to None stops a confused model from attaching a signature
        # verdict to an H&P, or plan clarity to a consent.
        signed = item.signed if item.role == "SURGICAL_CONSENT" else None
        plan_is_clear = item.plan_is_clear if item.role == "ANTICOAG_PLAN" else None
        labels[item.doc_id] = (item.role, signed, plan_is_clear)

    classified: list[ClassifiedDocument] = []
    for index, doc in enumerate(documents):
        role, signed, plan_is_clear = labels.get(doc.doc_id, ("OTHER", None, None))
        classified.append(
            ClassifiedDocument(
                index=index,
                document=doc,
                role=role,
                signed=signed,
                plan_is_clear=plan_is_clear,
            )
        )
    return classified


def build_medication_system_prompt() -> str:
    return render(MEDICATION_SYSTEM_TEMPLATE)


def build_medication_user_prompt(names: list[str]) -> str:
    return render(MEDICATION_USER_TEMPLATE, names=names)


class MedicationClassifier(Protocol):
    """Classifies medication names the reference list does not recognise."""

    def __call__(self, names: list[str]) -> list[ClassifiedMedicationModel]: ...


class LLMMedicationClassifier:
    """Fallback classifier for medication names absent from the reference list.

    Caching is keyed on the individual name rather than the whole prompt, so a
    drug seen in one submission is not re-classified for the next. Over a full
    dataset run this collapses to one call per distinct unknown name.
    """

    def __init__(self, *, model: str = CLASSIFIER_MODEL) -> None:
        self.model = model
        self._cache: dict[str, MedicationClass] = {}

    def __call__(self, names: list[str]) -> list[ClassifiedMedicationModel]:
        if not names:
            return []

        pending = sorted({name for name in names if name not in self._cache})
        if pending:
            result = _structured_call(
                model=self.model,
                instructions=build_medication_system_prompt(),
                prompt=build_medication_user_prompt(pending),
                text_format=MedicationClassificationResult,
            )
            requested = set(pending)
            for item in result.classifications:
                if item.name in requested:
                    self._cache[item.name] = item.medication_class

        # A name the model declined to return is left unresolved rather than
        # defaulted to OTHER; `resolve_medications` turns that into UNKNOWN.
        return [
            ClassifiedMedicationModel(name=name, medication_class=self._cache[name])
            for name in names
            if name in self._cache
        ]


def resolve_medications(
    medications: list[Medication],
    *,
    classifier: MedicationClassifier | None = None,
) -> list[ClassifiedMedication]:
    """Resolve each medication's class, reference list first.

    The curated list in `rules.medications` is authoritative: a name it
    recognises never reaches the model, so the common path is deterministic and
    auditable. Only genuinely unknown names are sent to `classifier`.

    A name that neither the list nor the model resolves -- including when no
    classifier is supplied at all -- comes back UNKNOWN, never OTHER. Silently
    clearing a drug nobody could identify is the one failure this rule must not
    have.
    """

    if not medications:
        return []

    unknown: list[str] = []
    known: dict[int, MedicationClass] = {}
    for index, medication in enumerate(medications):
        verdict = medication_reference.lookup(medication.name)
        if verdict is True:
            known[index] = "ANTICOAGULANT"
        elif verdict is False:
            known[index] = "OTHER"
        elif medication.name:
            unknown.append(medication.name)

    fallback: dict[str, MedicationClass] = {}
    if unknown and classifier is not None:
        for item in classifier(unknown):
            fallback[item.name] = item.medication_class

    resolved: list[ClassifiedMedication] = []
    for index, medication in enumerate(medications):
        medication_class = known.get(index) or fallback.get(medication.name or "", "UNKNOWN")
        resolved.append(
            ClassifiedMedication(
                index=index,
                medication=medication,
                medication_class=medication_class,
            )
        )
    return resolved
