# tinyGroot — common tasks. Run `make help` for the list.
# Local runs go through speedrun_mps.sh (Apple-Silicon / CPU, laptop-sized defaults).
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install dev doctor quickstart pretokenize train sft rl eval sample lint format test build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install core deps into a uv-managed .venv
	uv sync

dev: ## Install with dev + all extras (ruff, pytest, modal, wandb, gpu)
	uv sync --extra dev --extra all

doctor: ## Check the local environment (Python, torch, device)
	bash speedrun_mps.sh doctor

quickstart: ## Tiny end-to-end pretrain on this machine (MPS/CPU)
	bash speedrun_mps.sh train

pretokenize: ## Build a small local token shard + tokenizer
	bash speedrun_mps.sh pretokenize

train: ## Local pretrain (override e.g. MAX_STEPS=200 SEQ_LEN=512)
	bash speedrun_mps.sh train

sft: ## Local chat SFT (set SFT_CHECKPOINT=...)
	bash speedrun_mps.sh sft

rl: ## Local GSM8K RL (set RL_CHECKPOINT=...)
	bash speedrun_mps.sh rl

eval: ## Local eval (set EVAL_CHECKPOINT=...)
	bash speedrun_mps.sh eval

sample: ## Sample from a checkpoint dir: make sample CKPT=runs/my-run
	uv run tinygroot-sample --checkpoint-dir $(CKPT)

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

build: ## Build sdist + wheel into dist/
	uv build

clean: ## Remove caches and build artifacts
	rm -rf dist build *.egg-info .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
