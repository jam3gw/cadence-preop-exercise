"""LLM-backed classification of the free-text parts of a submission.

Two things resist deterministic interpretation. `documents[].type` is free text
from many source systems, so a case's whole document list goes to the model in
one call. Medication names are an open vocabulary with no reference data here,
so they are resolved against `rules.medications` first and only unknown names
reach the model.

In both cases the model returns small enum labels keyed by a short identifier
we supplied, never prose that reaches an issue. Keys are kept short on purpose:
a model asked to echo a 36-character UUID will eventually mistype one, and a
mistyped key is indistinguishable from a hallucinated one.
"""

from __future__ import annotations

from core import (
    ClassifiedDocument,
    ClassifiedMedication,
    Document,
    DocumentRole,
    Medication,
    MedicationClass,
)

import hashlib
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field

from rules import medications as medication_reference
from rules.model_config import (
    ModelConfig,
    document_classifier_config,
    medication_classifier_config,
)
from rules.prompts import render

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

    ref: int
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
    config: ModelConfig,
    instructions: str,
    prompt: str,
    text_format: type[ResultT],
) -> ResultT:
    """One Structured Outputs call, returning a validated instance.

    `responses.parse` with `text_format` has the SDK derive a strict schema and
    the API constrain decoding to it, so a schema violation is impossible
    rather than merely unlikely.
    """

    # Import lazily so this module stays importable without the openai package
    # installed (e.g. in unit tests that inject a fake classifier).
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=config.model,
        instructions=instructions,
        **config.request_kwargs(),
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

    Keyed by a short sequential `ref` rather than `doc_id`: a model asked to
    copy back a 36-character UUID mistyped one in testing (`...5eed-aee7-...`
    returned as `...5eed-ae7a-...`), and the reconciler discarded the correct
    classification as a hallucinated id.

    Omits `date` so the model cannot decide a document is "too old to count"
    instead of naming what it is. Recency is the rules' job.
    """

    return [
        {
            "ref": index + 1,
            "type": doc.type,
            "author": doc.author,
            "text": (doc.text or "")[:MAX_TEXT_CHARS],
        }
        for index, doc in enumerate(documents)
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

    def __init__(self, *, config: ModelConfig | None = None) -> None:
        self.config = config or document_classifier_config()
        self._cache: dict[str, list[ClassifiedDocumentModel]] = {}

    def __call__(self, documents: list[Document]) -> list[ClassifiedDocumentModel]:
        if not documents:
            return []

        prompt = build_classifier_user_prompt(documents)
        cache_key = hashlib.sha256(
            f"{self.config}\n{prompt}".encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = _structured_call(
            config=self.config,
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

    Refs are 1-based positions; anything out of range is discarded and any
    unclassified document falls back to OTHER. That fallback is safe in one
    direction -- an unlabelled document can only make a requirement look unmet,
    never clear a patient -- but it is silent, which is why short refs matter.
    """

    if not documents:
        return []

    labels: dict[int, tuple[DocumentRole, bool | None, bool | None]] = {}
    for item in classifier(documents):
        index = item.ref - 1
        if not 0 <= index < len(documents):
            continue
        # Each judgement is only meaningful for the role it belongs to; forcing
        # the others to None stops a confused model from attaching a signature
        # verdict to an H&P, or plan clarity to a consent.
        signed = item.signed if item.role == "SURGICAL_CONSENT" else None
        plan_is_clear = item.plan_is_clear if item.role == "ANTICOAG_PLAN" else None
        labels[index] = (item.role, signed, plan_is_clear)

    classified: list[ClassifiedDocument] = []
    for index, doc in enumerate(documents):
        role, signed, plan_is_clear = labels.get(index, ("OTHER", None, None))
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

    def __init__(self, *, config: ModelConfig | None = None) -> None:
        self.config = config or medication_classifier_config()
        self._cache: dict[str, MedicationClass] = {}

    @staticmethod
    def _echo_key(name: str) -> str:
        """Key for matching a returned name back to the one we asked about.

        The model normalises whitespace and casing when echoing names, so raw
        string matching would drop valid answers as unclassifiable.
        """

        return " ".join(name.split()).casefold()

    def __call__(self, names: list[str]) -> list[ClassifiedMedicationModel]:
        if not names:
            return []

        pending = sorted({name for name in names if self._echo_key(name) not in self._cache})
        if pending:
            result = _structured_call(
                config=self.config,
                instructions=build_medication_system_prompt(),
                prompt=build_medication_user_prompt(pending),
                text_format=MedicationClassificationResult,
            )
            requested = {self._echo_key(name) for name in pending}
            for item in result.classifications:
                key = self._echo_key(item.name)
                if key in requested:
                    self._cache[key] = item.medication_class

        # A name the model declined to return is left unresolved rather than
        # defaulted to OTHER; `resolve_medications` turns that into UNKNOWN.
        return [
            ClassifiedMedicationModel(
                name=name, medication_class=self._cache[self._echo_key(name)]
            )
            for name in names
            if self._echo_key(name) in self._cache
        ]


def resolve_medications(
    medications: list[Medication],
    *,
    classifier: MedicationClassifier | None = None,
) -> list[ClassifiedMedication]:
    """Resolve each medication's class, reference list first.

    A name `rules.medications` recognises never reaches the model, so the
    common path is deterministic and auditable. A name neither resolves comes
    back UNKNOWN, never OTHER -- silently clearing an unidentified drug is the
    one failure this must not have.
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
