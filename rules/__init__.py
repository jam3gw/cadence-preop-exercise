"""Policy rule evaluators for pre-op triage.

Each module implements one rule from the Cadence pre-operative scheduling
policy as a pure function over a `RuleContext`. Model-backed interpretation
(document roles, and medication names the reference list does not know)
happens once per submission, before any rule runs.
"""

from rules.base import (
    ClassifiedDocument,
    ClassifiedMedication,
    DocumentRole,
    LabRef,
    MedicationClass,
    Rule,
    RuleContext,
    build_issue,
    most_recent,
    most_recent_lab,
    parse_date,
)
from rules.classifiers import (
    ClassificationRefusedError,
    ClassifiedDocumentModel,
    ClassifiedMedicationModel,
    DocumentClassifier,
    LLMDocumentClassifier,
    LLMMedicationClassifier,
    MedicationClassifier,
    classify_documents,
    resolve_medications,
)
from rules.anticoagulation import evaluate as evaluate_anticoagulation
from rules.documentation import evaluate as evaluate_required_documentation
from rules.required_testing import evaluate as evaluate_required_testing

__all__ = [
    "ClassificationRefusedError",
    "ClassifiedDocument",
    "ClassifiedDocumentModel",
    "ClassifiedMedication",
    "ClassifiedMedicationModel",
    "DocumentClassifier",
    "DocumentRole",
    "LLMDocumentClassifier",
    "LLMMedicationClassifier",
    "LabRef",
    "MedicationClass",
    "MedicationClassifier",
    "Rule",
    "RuleContext",
    "build_issue",
    "classify_documents",
    "evaluate_anticoagulation",
    "evaluate_required_documentation",
    "evaluate_required_testing",
    "most_recent",
    "most_recent_lab",
    "parse_date",
    "resolve_medications",
]
