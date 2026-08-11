PY := .venv/bin/python

.PHONY: eval eval-fast forge test

## eval: full Truth Engine run — forensics + decision + VLM extraction + CI gates
eval:
	$(PY) -m tools.eval.run_eval --check

## eval-fast: same, without the VLM tier (no Ollama required — CI-safe)
eval-fast:
	$(PY) -m tools.eval.run_eval --no-extraction --check

## forge: regenerate the synthetic + tamper datasets from their seeds
forge:
	$(PY) -m tools.eval.run_eval --regen --no-extraction

## test: unit test suite
test:
	$(PY) -m pytest tests/unit -q
