"""LLM-backed document classification.

`documents[].type` in submissions is free text produced by many source systems
("PREOP - H and P - signed", "Scanned Hist & Phys (H&P) (external)", "Imported:
Consult H&P", ...). Treating it as an enum is brittle, so instead we send the
full document list (type + author + text) for a case to a model in one call and
ask it to label each `doc_id` with the policy role it actually serves.

The model never sees or invents an index and never produces the prose that ends
up in an issue -- it only returns `{doc_id, role, signed}` tuples, which we
validate against the real document ids before using them. That keeps token
output small, keeps hallucinated ids from silently corrupting evidence, and
keeps every rule's phrasing deterministic.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from core import Document
from rules.base import ClassifiedDocument, DocumentRole
from rules.prompts import render

CLASSIFIER_MODEL = "gpt-4.1-mini"

SYSTEM_TEMPLATE = "document_classifier_system.j2"
USER_TEMPLATE = "document_classifier_user.j2"

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


class DocumentClassificationResult(BaseModel):
    classifications: list[ClassifiedDocumentModel] = Field(default_factory=list)


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

        classifications = self._request(prompt)
        self._cache[cache_key] = classifications
        return classifications

    def _request(self, prompt: str) -> list[ClassifiedDocumentModel]:
        # Import lazily so this module stays importable without the openai
        # package installed (e.g. in unit tests that inject a fake classifier).
        from openai import OpenAI

        client = OpenAI()
        # `responses.parse` with `text_format` is genuine Structured Outputs:
        # the SDK derives a strict JSON schema from the model (every property
        # required, additionalProperties false) and the API constrains decoding
        # to it, so a schema violation is impossible rather than merely
        # unlikely. It also hands back a validated instance directly.
        response = client.responses.parse(
            model=self.model,
            instructions=build_classifier_system_prompt(),
            temperature=0,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            text_format=DocumentClassificationResult,
        )

        result = response.output_parsed
        if result is None:
            # Only reachable if the model refused; there is no partial output
            # worth salvaging, and guessing roles would be worse than failing.
            raise ClassificationRefusedError(
                "Model did not return a document classification "
                f"(status={response.status!r})"
            )
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

    labels: dict[str | None, tuple[DocumentRole, bool | None]] = {}
    for item in classifier(documents):
        if item.doc_id not in known_ids:
            continue
        # Only consent documents carry a meaningful signature judgement.
        signed = item.signed if item.role == "SURGICAL_CONSENT" else None
        labels[item.doc_id] = (item.role, signed)

    classified: list[ClassifiedDocument] = []
    for index, doc in enumerate(documents):
        role, signed = labels.get(doc.doc_id, ("OTHER", None))
        classified.append(
            ClassifiedDocument(index=index, document=doc, role=role, signed=signed)
        )
    return classified
