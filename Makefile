PY ?= .venv/bin/python
NPM ?= npm

.PHONY: venv install train smoke frontend run health clean \
        stage0 stage2 stage4 gen-packet ingest-gen check-pairs cross-gen

venv:
	python3 -m venv .venv

install:
	$(PY) -m pip install -e ".[all]"
	cd frontend && $(NPM) install

# One-time: download data, fine-tune model, precompute HITL pool + examples.
train:
	$(PY) -m training.train

# Fast smoke-training on a tiny subset to verify the pipeline.
smoke:
	$(PY) -m training.train --sample 2000 --epochs 1 --max_len 128

# Zero-shot machine-text signal baseline + preprocessing ablation + a check of
# how far the shipped sentiment-proxy model is from the real task.
stage0:
	$(PY) -m detect.stage0

# Build the Stage 1 packet an external LLM consumes to write machine reviews.
# Override the model label with: make gen-packet MODEL=claude-sonnet-5
MODEL ?= claude-opus-5
gen-packet:
	$(PY) -m scripts.make_gen_packet --model $(MODEL)

# Fold generator output back in. Usage:
#   make ingest-gen RESULTS=results/claude-opus-5.jsonl GENERATOR=claude-opus-5
ingest-gen:
	$(PY) -m scripts.ingest_gen_results --results $(RESULTS) --generator $(GENERATOR)

# Health-check a generator run before trusting it: template collapse, whether
# rewrites are bound to their source, opinion preservation, truncation.
#   make check-pairs RESULTS=data/gen_results/<tag> GENERATOR=<tag>
check-pairs:
	$(PY) -m scripts.check_pairs --results $(RESULTS) --generator $(GENERATOR)

# Score genuine output from other generators with the trained detector. Needs no
# second generation run: evaluation uses only the text, not packet alignment.
cross-gen:
	$(PY) -m scripts.cross_generator --generator $(GENERATORS)

# Train + evaluate the machine-generated-review detector on the Stage 1 pairs.
# The default is the only run verified to be genuine model output — do NOT point
# this at mimo-v2.5-full, which came from scripts/gen_reviews.py and would train
# the detector to separate template text from human text.
#   make stage2 GENERATORS="mimo-v2.5-rerun <second-run>"
GENERATORS ?= mimo-v2.5-rerun
stage2:
	$(PY) -m training.stage2 --generators $(GENERATORS)

# Stage 4: precompute the HITL pool and Examples page for the current detector.
# Both serve data built from the detector, so re-run this whenever the detector
# changes (a new generator, a retrained model).
stage4:
	$(PY) -m training.hitl_pool --generator $(GENERATORS)
	$(PY) -m training.examples --generator $(GENERATORS)

# Build the React + Ant Design frontend into frontend/dist (served by the API).
frontend:
	cd frontend && ([ -d node_modules ] || $(NPM) install) && $(NPM) run build

run: frontend
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

health:
	curl -s http://127.0.0.1:8000/api/health

clean:
	rm -rf data models frontend/dist
