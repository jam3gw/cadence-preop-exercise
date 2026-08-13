INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
# Empty uses the per-task models in rules/model_config.py.
MODEL ?=

CLASSIFIER_REPORT ?= data/classifier_eval_report.json
EVAL_WORKERS ?= 12
EVAL_BATCH_SIZE ?= 8

.PHONY: baseline evals determinism score report test all clean classifier-evals fixtures

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model "$(MODEL)"

evals:
	uv run run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

determinism:
	uv run run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model "$(MODEL)" \
		--report $(DETERMINISM_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run \
		--with 'jinja2>=3.1.0' \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		python -m pytest tests

# Evaluates the document and medication classifiers against labelled fixtures
# with real model calls. Ambiguous fixtures and any mismatch are arbitrated by
# the judge model; everything else is a plain assertion.
classifier-evals:
	uv run evals/run_classifier_evals.py \
		--suite all \
		--report $(CLASSIFIER_REPORT) \
		--workers $(EVAL_WORKERS) \
		--batch-size $(EVAL_BATCH_SIZE)

# Regenerates the document fixtures from the sample data. Review the diff:
# this is a labelling heuristic, not an oracle.
fixtures:
	uv run evals/build_fixtures.py

all: baseline evals determinism score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
