test:
	python -m pytest tests/*
fmt:
	black src/*
	isort src/* --virtual-env=.emevo-venv
