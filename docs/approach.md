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

**6. Edge cases and cleanup.** Handling the remaining edge cases and tidying the codebase.

## Why this order

Building rules in isolation before integration meant each one could be tested
without an API key and without the others. That paid off repeatedly: when the
integrated pipeline disagreed with expectations, I could tell very quickly
whether the fault was in a rule, in a classifier, or in the assembly.

Deferring evidence quality was a deliberate sequencing call. Getting a decision
right is a correctness problem; explaining it well is a presentation problem.
Doing them in that order meant the presentation work landed on rules that were
already known to be correct.
