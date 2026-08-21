PY ?= .venv/bin/python

.PHONY: venv install train smoke run health clean

venv:
	python3 -m venv .venv

install:
	$(PY) -m pip install -e ".[all]"

# One-time: download data, fine-tune model, precompute HITL pool + examples.
train:
	$(PY) -m training.train

# Fast smoke-training on a tiny subset to verify the pipeline.
smoke:
	$(PY) -m training.train --sample 2000 --epochs 1 --max_len 128

run:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

health:
	curl -s http://127.0.0.1:8000/api/health

clean:
	rm -rf data models
