# Approach

## High-level steps

**1. Understand the service before writing anything.** The goal, the logic
behind each rule, and — most importantly — how each rule connects to the data
actually present in the input model. Several rules read differently once you
see what the submission really contains.

**2. Build the skeleton one rule at a time.** For each rule, the question I
asked first was: what level of determinism does this need, and is there
genuinely unstructured information here that requires an LLM to turn into
structured data? That was the case for Rules 1 and 3. It was not the case for
Rules 2 and 4, which are pure comparison logic.

**3. One PR per rule, in isolation.** Each PR dug into the edge cases I could
identify myself, and used Claude to surface gaps I had missed. I documented
assumptions as I went — in the code, in commit messages, and in PR
descriptions — so that decisions stayed tracked rather than becoming folklore.

**4. Rule-level testing before integration.** A classifier eval suite combining
static expected outputs with LLM-as-a-judge for the genuinely ambiguous cases.
I then used those results to iterate on the logic and prompts behind Rules 1
and 3, rather than guessing at what to improve.

**5. Wire everything up and run the supplied end-to-end tests.** Only at this
point does the implementation actually get used by the harness. The results
drove another round of changes — mostly around surfacing evidence in the final
output, which I had deliberately deferred. Evidence quality is easy to enrich
late; rule correctness is not.

**6. Edge cases, cost, and cleanup.** Handling the remaining edge cases,
reducing unnecessary model calls, and tidying the codebase.

## Why this order

Building rules in isolation before integration meant each one could be tested
without an API key and without the others. That paid off repeatedly: when the
integrated pipeline disagreed with expectations, I could tell within minutes
whether the fault was in a rule, in a classifier, or in the assembly.

Deferring evidence quality was a deliberate sequencing call. Getting a decision
right is a correctness problem; explaining it well is a presentation problem.
Doing them in that order meant the presentation work landed on rules that were
already known-good.

## What the evals caught

Several defects were found by the eval suite rather than by reading code, which
is the main argument for building it before integration rather than after:

- A prompt instruction that mislabelled 39 documents. It told the model that a
  document "retaining a prior H&P for chart context" was not an H&P, which is
  wrong: a retained H&P is still an H&P, and whether it is recent enough is
  Rule 1's job. Document accuracy went from 75.8% to 98.2%.
- Consent scoping that was too broad once counselling notes were admitted,
  pulling in anesthesia and blood transfusion consents.
- Two identifier round-tripping bugs, described below.

## The bug worth reading about

`case_00004` intermittently reported a missing surgical consent for a case that
plainly had one. It looked like a classification weakness for several rounds.

It was not. The model classified the document correctly every single time, and
mistyped the id it was asked to echo back:

```
sent:     6bf7813f-59e1-5eed-aee7-a4ac6c47ae39
returned: 6bf7813f-59e1-5eed-ae7a-a4ac6c47ae39
```

A four-character transposition mid-UUID. The reconciler could not distinguish
that from a hallucinated id, dropped the label, and fell back to `OTHER` —
discarding a correct answer and surfacing it as a missing document.

The fix was to stop asking: documents are keyed by a short sequential `ref`
rather than a 36-character UUID, mapped back to position in code.

Two things made this instructive. First, it was the second instance of one root
problem — the medication classifier had the same bug, where the model
normalised whitespace and casing in echoed drug names. Any point where an
identifier round-trips through a model is a place to look. Second, it was
invisible to the fixture suite and only appeared in the pipeline, because the
two batch documents differently. An eval suite that does not reproduce
production's input shape will miss things, however good its coverage looks.

## Results

Against the naive single-call baseline on the same 50 cases, with the naive
implementation re-run rather than quoted from memory:

| Metric | Naive | This implementation |
| --- | ---: | ---: |
| `json_schema_valid` | 100.00 | 100.00 |
| `decision_match_oracle` | 90.00 | **100.00** |
| `issue_categories_match_oracle` | 78.00 | **98.00** |
| `issues_value_grounding` | 70.00 | **100.00** |
| **aggregate** | **86.57** | **99.43** |

Determinism is 100% on decision stability, JSON format and exact output match
across ten runs. Those are ten genuine API calls — `triage_submission` builds a
fresh classifier per invocation, so nothing is served from cache.

One case fails, `case_00002`, and it is a limitation of the expected outputs
rather than of the implementation. The document type there is misspelled
(`'History & Phsyical'`), the expected output misses it and falls back to an
older H&P, and this implementation reads it correctly and finds it in window.
Matching would mean reproducing the blind spot, so I left it.
