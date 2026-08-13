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
namespace prefix.

### Document IDs should never be null

Document IDs should come from an internal database (if this were a real production system). We would create the ID when a document
is uploaded to our system or the EHR — so they should always be present.

### Only LOW, MODERATE or HIGH procedure risk is accepted

*In the code:* Rule 2 validates `procedure.procedure_risk` against exactly those
three values. Anything else — absent, blank, or an unrecognised string like
`"MEDIUM"` — is reported as `MISSING_REQUIRED_DATA`, which means
`NEEDS_FOLLOW_UP`.

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

### A null `active` flag means the medication is inactive

Two reasons. First, in health systems this field should be standardised to a
non-nullable boolean. Second, it is plausible that clearing the flag is how some
systems express "no longer taking" — the physician simply deleted the field.

*In the code:* `is_active` requires an explicit `True`.

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
`UNKNOWN` and is reported.

## Technical assumptions

### Any model may be used, including different models for different calls

*In the code:* `rules/model_config.py` configures a model and reasoning effort
per task — document classification, medication classification, and the eval
judge — and every one is overridable by environment variable.

## Assumptions I would want confirmed

- **Consent must be for the surgery itself.** Anesthesia consent, blood
  transfusion consent and similar are classified `OTHER` however clearly they
  are signed, on the reading that the policy requires consent for the procedure.
- **An H&P dated after the procedure date is flagged rather than accepted**, as
  is a lab resulted after the procedure date. Both are data anomalies, and the
  policy's default for anything unclear is follow-up.
