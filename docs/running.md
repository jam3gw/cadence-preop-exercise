# Running this project

The workflow is the same shape as the starter repo's, with two additions and a
couple of behavioural differences. If you have run the original, skim
[What changed from the starter](#what-changed-from-the-starter) and skip the rest.

## Setup

1. Confirm `uv` is installed.

```bash
uv --version
```

2. Provide an OpenAI API key, either exported or in a `.env` file at the repo
   root. `.env` is gitignored.

```bash
export OPENAI_API_KEY="<your_api_key>"
```

`run_baseline.py` and the classifier eval runner read `.env` themselves.
`run_evals.py` does not, so `make evals`, `make determinism` and `make score`
need the key exported into the shell. Exporting it covers everything.

No other install step is needed. Every entry point is a `uv` script with its
dependencies declared inline.

## The main workflow

```bash
make baseline      # run triage over the sample data
make evals         # score the outputs against the expected outputs
make score         # print the aggregate score
make determinism   # replay one case 10x and check for drift
make report        # interactive TUI over the eval report
```

`make all` runs baseline, evals, determinism and score in sequence.

Expect `make baseline` to take a few minutes. It makes two model calls per
submission and runs at high reasoning effort.

## Tests

```bash
make test
```

Unit tests inject both classifiers, so they make no network calls and need no
API key. An autouse fixture fails the test if anything attempts a live call.

## Classifier evals

The two model-backed classifiers have their own eval suite, separate from the
end-to-end scoring above. It runs them against labelled fixtures with real
model calls, and uses a judge model to arbitrate the genuinely ambiguous cases.

```bash
make classifier-evals
```

Or directly, for more control:

```bash
uv run evals/run_classifier_evals.py --suite documents --workers 12
uv run evals/run_classifier_evals.py --suite medications --no-judge
uv run evals/run_classifier_evals.py --suite all --limit 20
```

Writes `data/classifier_eval_report.json` and exits non-zero if any case fails.

The document fixtures are generated from the repo's own sample data, so they
are the exact strings the classifier meets in production. To regenerate after
changing the labelling heuristic:

```bash
make fixtures
```

Review the diff when you do. That script is a labelling heuristic, not an
oracle.

## Configuration

Make variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `INPUT` | `data/patients_sample_50.jsonl` | Input cases |
| `OUTPUT` | `data/baseline_outputs.jsonl` | Baseline outputs |
| `REPORT` | `data/eval_report.json` | Eval report |
| `DETERMINISM_REPORT` | `data/determinism_report.json` | Determinism report |
| `CLASSIFIER_REPORT` | `data/classifier_eval_report.json` | Classifier eval report |
| `MODEL` | *(empty)* | Overrides both classifiers with one model id |
| `EVAL_WORKERS` | `12` | Concurrency for classifier evals |
| `EVAL_BATCH_SIZE` | `8` | Fixtures per model call |

`MODEL` is empty by default, which means "use the per-task configuration in
`rules/model_config.py`". Setting it forces both classifiers onto a single
model, which is mostly useful for comparing against a baseline.

Models are configured per task and each is overridable by environment
variable, so you can swap a model or change an effort level without editing
code:

| Task | Variable | Default |
| --- | --- | --- |
| Document classification | `PREOP_DOCUMENT_CLASSIFIER_MODEL` / `_EFFORT` | `gpt-5.6-luna`, high |
| Medication classification | `PREOP_MEDICATION_CLASSIFIER_MODEL` / `_EFFORT` | `gpt-5.6-terra`, high |
| Eval judge | `PREOP_JUDGE_MODEL` / `_EFFORT` | `gpt-5.6-sol`, medium |

Setting an effort to `none` disables reasoning and pins `temperature=0`
instead, so a non-reasoning model can be substituted cleanly. The two are
mutually exclusive: reasoning models reject an explicit temperature.

```bash
PREOP_DOCUMENT_CLASSIFIER_EFFORT=medium make baseline
PREOP_MEDICATION_CLASSIFIER_MODEL=gpt-4.1-mini PREOP_MEDICATION_CLASSIFIER_EFFORT=none make baseline
```

## What changed from the starter

- **`make baseline` runs submissions concurrently** (`--workers`, default 8).
  Serial execution was impractical once each submission made two calls at
  reasoning effort. Output rows are still written in input order, so the file
  stays byte-stable across runs.
- **`MODEL` defaults to empty** rather than `gpt-4.1-mini`, because models are
  now configured per task.
- **`make classifier-evals` and `make fixtures` are new.**
- **`run_baseline.py` reads `.env`.** `run_evals.py` does not.

## Layout

```
core.py                    schemas, rule primitives, and the assembler
rules/
  documentation.py         Rule 1 -- required documentation
  required_testing.py      Rule 2 -- required testing by risk level
  anticoagulation.py       Rule 3 -- anticoagulation management
  acute_safety.py          Rule 4 -- acute safety exclusions
  classifiers.py           the two model-backed classifiers
  medications.py           curated anticoagulant reference list
  model_config.py          per-task model and reasoning effort
  prompts.py, templates/   Jinja2 prompt templates
evals/
  run_classifier_evals.py  classifier eval runner
  build_fixtures.py        derives document fixtures from the sample data
  judge.py, templates/     LLM judge
  data/                    labelled fixtures
docs/                      this documentation
```
