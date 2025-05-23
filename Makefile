.PHONY: test lint

CUDA_AVAILABLE := $(shell command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0)
PORT ?= 9998

test:
	uv run pytest

lint:
	uvx ruff check
	uvx black src/emevo tests scripts --check
	uvx isort src/emevo tests scripts --check


format:
	uvx black src/emevo tests scripts
	uvx isort src/emevo tests scripts


publish:
	uv build
	uvx twine upload dist/*


register:
	uv run ipython kernel install --user --name=emevo-lab


jupyter:
	uv run jupyter lab --port=$(PORT) --no-browser


sync:
ifeq ($(CUDA_AVAILABLE),1)
	uv sync --extra=analysis --extra=cuda12
else
	uv sync --extra=analysis
endif


all: test lint
