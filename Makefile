test:
	python -m pytest $(ARGS)
fmt:
	black src/*
	isort src/*
