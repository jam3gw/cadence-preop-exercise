# Assumptions

Assumptions made while building this, with a note on where each one shows up in
the code. Several are things I would confirm with the team before shipping.

## Domain assumptions

### Lab types should be an enum

`labs[].code` is treated as a controlled vocabulary rather than free text.
These values should be coming from devices and systems we control, so requiring
a fixed set is reasonable — and unlike document titles, the sample data bears
this out.

*In the code:* Rule 2 matches `code` directly, and deliberately ignores
`labs[].display`, which **is** free text (`Complete Blood Count w/
Differential`, `CBC (Complete Blood Count)`, `Comprehensive Metabolic Panel w/
Magnesium`). No model is involved.

*One wrinkle:* the vocabulary arrives in two dialects. `CBC` and `LAB-CBC` are
the same test, as are `CMP` and `LAB-CMP`. Codes are normalised by stripping the
namespace prefix. This is not cosmetic — in 19 of the 50 sample cases the most
recent CBC is coded `LAB-CBC`, and in 9 of those an exact match on `"CBC"` finds
nothing and reports a false "CBC missing".

### Document IDs should never be null

Document IDs come from an internal database — we create the ID when a document
is uploaded to our system or the EHR — so they should always be present.

*In the code:* this is no longer load-bearing. Documents are keyed to the
classifier by a short sequential `ref`, not by `doc_id`, after a model
mistyped a UUID and a correct classification was discarded as a hallucinated
id. `doc_id` remains on the schema but nothing in the triage path matches on
it. The assumption stands; the implementation no longer depends on it.

### Only LOW, MODERATE or HIGH procedure risk is accepted

*In the code:* Rule 2 validates `procedure.procedure_risk` against exactly those
three values. Anything else — absent, blank, or an unrecognised string like
`"MEDIUM"` — is reported as `MISSING_REQUIRED_DATA`, which means
`NEEDS_FOLLOW_UP`.

Note this required loosening `ProcedureInfo.procedure_risk` from a `Literal` to
`str` in the schema. Under the stricter typing, a submission carrying `"MEDIUM"`
raised a `ValidationError` during parsing and killed the whole submission before
triage ran, making the unrecognised-risk branch unreachable. A malformed enum
value from upstream is a triage finding, not a parse failure.

### All labs appearing in the record are complete

There are no incomplete or preliminary labs in a patient's input file. It would
be odd for Cadence to receive a report for an incomplete lab.

*In the code:* no filter on `labs[].status`. All 150 labs in the sample are
`"final"`, consistent with the assumption. If preliminary results ever appear,
this is a gap — they would currently be treated as usable.

### Vitals types are an enum

`blood_pressure` and `temperature` are the two types.

*In the code:* Rule 4 locates each required vital by matching `type` directly,
case- and whitespace-insensitively. No model is involved.

Measurement fields are read with `getattr` rather than by branching on the
`Vital` union type, because a reading missing its values resolves to
`GenericVital`, and a rule should see that as an absent measurement rather than
crash on the wrong branch.

### A null `active` flag means the medication is inactive

Two reasons. First, in health systems this field should be standardised to a
non-nullable boolean. Second, it is plausible that clearing the flag is how some
systems express "no longer taking" — the physician simply deleted the field.

*In the code:* `is_active` requires an explicit `True`.

*Important nuance:* this does not mean the unknown status is ignored. Rule 3
still reports an anticoagulant with `active: null` as `MISSING_REQUIRED_DATA`
("Unknown anticoagulant active status"), so the patient cannot reach `READY` and
a human confirms the medication. The assumption governs only whether we
*additionally* demand a perioperative plan for a row we cannot confirm.

I tested the more cautious reading — treating anything other than an explicit
`False` as active — and it produces a second issue per affected case, demanding
a plan for every stale medication row. The resulting noise is its own safety
problem, and the unknown status is surfaced either way.

### Anticoagulant identification would ideally come from a dataset

In an ideal world there is a database of every anticoagulant by medication name,
identifier or SKU, and we use that to classify a medication. That data was not
available here.

*In the code:* a curated deterministic list based on the drug classes in routine
use (`rules/medications.py`), with a fallback to a stronger model
(`gpt-5.6-terra`) for names the list does not recognise. There is inherent risk
in trusting a model here, but the task is drug classification rather than a
clinical judgement about a specific patient, so it seems a reasonable trade for
this exercise.

Building the list in lieu of a dataset also serves determinism and cost: every
medication in the sample data resolves from the list, so the fallback never runs
on this input. A name that neither the list nor the model resolves comes back
`UNKNOWN` and is reported — never silently cleared.

## Technical assumptions

### Any model may be used, including different models for different calls

*In the code:* `rules/model_config.py` configures a model and reasoning effort
per task — document classification, medication classification, and the eval
judge — and every one is overridable by environment variable.

## Assumptions I would want confirmed

- **`case_00002`'s expected output looks wrong.** The document type is
  misspelled (`'History & Phsyical'`), the expected output misses it and falls
  back to an older H&P, and this implementation reads it correctly. I chose not
  to reproduce that blind spot, which costs one case on the category metric.
- **Consent must be for the surgery itself.** Anesthesia consent, blood
  transfusion consent and similar are classified `OTHER` however clearly they
  are signed, on the reading that the policy requires consent for the procedure.
- **An H&P dated after the procedure date is flagged rather than accepted**, as
  is a lab resulted after the procedure date. Both are data anomalies, and the
  policy's default for anything unclear is follow-up.
- **A retained or superseded H&P is still an H&P.** Whether it is recent enough
  is Rule 1's decision, not the classifier's. The expected outputs agree —
  `case_00002` selects exactly such a retained document as the H&P.
