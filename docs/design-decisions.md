# Key design decisions

## Use an LLM only where the data is genuinely unstructured

I am a strong advocate for being deliberate about when to reach for an LLM in a
high-impact workflow. Two reasons: LLM calls cost meaningfully more than static
code, and their probabilistic nature — even with the best evals in the world —
introduces inconsistency into decisions being made on a patient's behalf. In a
pipeline determining whether someone is cleared for surgery, the goal is to
minimise inconsistency and maximise determinism.

So a model is used in exactly two places, both of which are genuine
unstructured-to-structured conversions:

**Document role classification.** `documents[].type` is free text from many
source systems. The 50-case sample contains 107 distinct type strings, 92 of
them appearing exactly once — prefixes (`PREOP -`, `Imported:`, `Scanned`),
suffixes (`(scanned)`, `[PDF]`, `- signed`), abbreviations (`H&P`, `H and P`,
`H+P`, `H/P`, `Hx & Physical`), and at least one misspelling
(`History & Phsyical`). Matching this with rules is brittle by construction.

**Medication classification, as a fallback only.** Discussed below.

Everything else is deterministic code. Rules 2 and 4 involve no model at all:
`labs[].code` and `vitals[].type` are machine-generated values from controlled
vocabularies, so they are matched directly.

## Keep the model's job as small as possible

Where a model is used, it returns nothing but small enum labels keyed by an
identifier we supplied. It never sees dates, never produces prose that reaches
an issue, and never decides an outcome.

This has three consequences worth stating:

- **Recency stays in code.** The classifier is not told document dates, so it
  cannot quietly decide something is "too old to count" instead of naming what
  it is. Whether a document is in window is Rule 1's decision.
- **All issue prose is templated in Python.** Dates, day counts and thresholds
  are formatted deterministically, which is why the same submission produces
  byte-identical output across runs.
- **A wrong label degrades rather than corrupts.** The worst a bad label can do
  is make a requirement look unmet. It cannot invent an evidence path or fabricate
  a measurement.

## Identifiers round-tripped through a model must be short

Documents are keyed to the model by a short sequential `ref`, not by their
`doc_id`. This is not cosmetic. Asking a model to echo back a 36-character
UUID produced a real transcription error in testing —
`...5eed-aee7-...` returned as `...5eed-ae7a-...` — which is indistinguishable
from a hallucinated id, so a correct classification was discarded. The
medication classifier had the same problem in a different form: names came back
with whitespace and casing normalised, and exact-string matching dropped them.

The general rule: if a model has to copy an identifier, make it short enough
that copying it is trivial, and normalise on both sides when it cannot be.

## Deterministic reference data first, model second

For Rule 3, the ideal is a database of drug classifications keyed by name,
identifier or SKU. No such data exists in this environment, so the approach is a
curated reference list of anticoagulants covering the agents in routine use by
generic and brand name — organised by the pharmacological classes in ATC group
B01A — with a stronger model as fallback for names the list does not recognise.

This is a deliberate trade. There is inherent risk in trusting a model here, but
the task is drug classification rather than a clinical judgement about a
specific patient, and the fallback only runs for names the curated list could
not settle. In practice it never runs on the sample data: every medication
present resolves from the list, so the common path costs nothing and is fully
auditable.

Antiplatelets (aspirin, clopidogrel, ticagrelor and similar) are recorded
explicitly as *not* anticoagulants rather than simply omitted. They are the
genuinely confusable case — a model asked "is this a blood thinner?" may well
say yes — and the policy speaks specifically of anticoagulant medication. If
the center intends antiplatelets to require a perioperative plan too, that is
one set moving between two constants in `rules/medications.py`.

## Fail toward follow-up, never toward cleared

The two possible errors are not symmetric. A spurious follow-up costs a phone
call; a missed anticoagulation plan is a patient safety event. So:

- A medication neither the list nor the model can identify resolves to
  `UNKNOWN` and is reported, never silently treated as harmless.
- A document the classifier fails to label falls back to `OTHER`, which can only
  make a requirement look unmet.
- An unreadable vital is reported as missing data rather than treated as an
  absent exclusion — the rule can only rule an exclusion *out* by reading the
  measurement.
- A lab or H&P with an unparseable date counts as missing, because a result that
  cannot be placed in time cannot be established as the most recent.

The one place this is uncomfortable is that the `OTHER` fallback is *silent*.
That is what let the UUID bug hide for several rounds, and it is now documented
where the fallback lives.

## Severity is derived from category, not asserted by rules

A rule reports findings; it does not decide the outcome. The assembler maps
categories to a decision: an acute safety exclusion means `NOT_CLEARED`, any
other outstanding issue means `NEEDS_FOLLOW_UP`, nothing outstanding means
`READY`.

I briefly built a mechanism for rules to escalate a decision directly, then
removed it when it turned out nothing needed it. Category is sufficient, and
one way of deciding severity is better than two.

## Per-task model configuration

The two classifiers do not share a model. Document classification is largely a
naming problem over an inconsistent vocabulary; medication classification can
require real pharmacological reasoning and is the judgement where being wrong
matters most. They are configured separately, and every model and effort level
is overridable by environment variable.

Reasoning effort for document classification was raised from medium to high on
evidence rather than intuition. Over three runs of the 165-case fixture suite,
medium scored 163/160/162 and high scored 165/164/164 — better accuracy, and a
fifth of the run-to-run spread. The end-to-end score could not resolve the
difference between them, so the fixture suite was the deciding measurement.

## Prompts are versioned as their own artifact

Prompt text lives in Jinja2 templates under `rules/templates/` rather than in
string literals, so prompt changes are readable in diffs and editable without
touching call-site code. The environment uses `StrictUndefined`, so a typo'd
template variable raises instead of silently rendering an empty string into a
prompt.

## Structured Outputs, properly

Both classifiers use `responses.parse` with a Pydantic `text_format`, which is
genuine Structured Outputs: the SDK derives a strict JSON schema and the API
constrains decoding to it, so a schema violation is impossible rather than
merely unlikely. The starter used `strict: False`, which shows the model a
schema without enforcing it.

Nullable fields survive this. `signed` and `plan_is_clear` remain expressible as
"not determinable" via `anyOf`, which is usually what pushes people back to
non-strict mode unnecessarily.
