.PHONY: test lint

CUDA_AVAILABLE := $(shell command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0)

test:
	uv run pytest

lint:
	uvx ruff check
	uvx black src/emevo tests --check
	uvx isort src/emevo tests --check


format:
	uvx black src/emevo tests
	uvx isort src/emevo tests


publish:
	uv build
	uvx twine upload dist/*


register:
	uv run ipython kernel install --user --name=emevo-lab


jupyter:
	uv run --with jupyter jupyter lab --port=9998 --no-browser


sync:
ifeq ($(CUDA_AVAILABLE),1)
	uv sync --extra=analysis --extra=cuda12
else
	uv sync --extra=analysis
endif


all: test lint
