"""Policy rule evaluators for pre-op triage.

Each module implements one rule from the Cadence pre-operative scheduling
policy as a pure function over a `RuleContext`. Model-backed interpretation
(document roles, and medication names the reference list does not know)
happens once per submission, before any rule runs.
"""

from core import (
    ClassifiedDocument,
    ClassifiedMedication,
    DocumentRole,
    LabRef,
    MedicationClass,
    Rule,
    RuleContext,
    VitalRef,
    build_issue,
    most_recent,
    most_recent_lab,
    most_recent_vital,
    parse_date,
    parse_timestamp,
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
from rules.acute_safety import evaluate as evaluate_acute_safety
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
    "VitalRef",
    "build_issue",
    "classify_documents",
    "evaluate_acute_safety",
    "evaluate_anticoagulation",
    "evaluate_required_documentation",
    "evaluate_required_testing",
    "most_recent",
    "most_recent_lab",
    "most_recent_vital",
    "parse_date",
    "parse_timestamp",
    "resolve_medications",
]
