# Make targets (Linux/macOS/CI). On Windows run the python commands shown in each recipe.
PY ?= python
CONFIG ?= smoke
EVAL_CONFIG ?= configs/eval.yaml
export KERAS_BACKEND ?= jax

.PHONY: setup data train eval report app test lint format deploy-space push-model clean

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[train,eval,app,dev]" tensorflow-cpu safetensors sentencepiece

data:
	$(PY) -m recipe_gemma.data.build --config configs/data.yaml  # also builds the flavor graph

train:
	$(PY) -m recipe_gemma.train --config configs/train_$(CONFIG).yaml

eval:
	$(PY) -m recipe_gemma.eval --config $(EVAL_CONFIG)
	$(PY) -m recipe_gemma.eval.report --config $(EVAL_CONFIG)

report:
	$(PY) -m recipe_gemma.eval.report --config $(EVAL_CONFIG)

app:
	$(PY) app/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

push-model:
	$(PY) scripts/push_adapter.py --config configs/train_$(CONFIG).yaml

deploy-space:
	$(PY) scripts/deploy_space.py

clean:
	rm -rf checkpoints runs .pytest_cache .ruff_cache
