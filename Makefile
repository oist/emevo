test:
	source .emevo-venv/bin/activate && python -m pytest tests/*
fmt:
	black src/*
	black tests/*
	isort src/* --virtual-env=.emevo-venv
	isort tests/* --virtual-env=.emevo-venv
