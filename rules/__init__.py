"""Policy rule evaluators for pre-op triage.

Each module implements one rule from the Cadence pre-operative scheduling
policy as a pure function over a `RuleContext`. Model-backed interpretation
(currently document role classification) happens once per submission, before
any rule runs.
"""

from rules.base import (
    ClassifiedDocument,
    DocumentRole,
    Rule,
    RuleContext,
    build_issue,
    most_recent,
    parse_date,
)
from rules.classifiers import (
    ClassificationRefusedError,
    ClassifiedDocumentModel,
    DocumentClassifier,
    LLMDocumentClassifier,
    classify_documents,
)
from rules.documentation import evaluate as evaluate_required_documentation

__all__ = [
    "ClassificationRefusedError",
    "ClassifiedDocument",
    "ClassifiedDocumentModel",
    "DocumentClassifier",
    "DocumentRole",
    "LLMDocumentClassifier",
    "Rule",
    "RuleContext",
    "build_issue",
    "classify_documents",
    "evaluate_required_documentation",
    "most_recent",
    "parse_date",
]
