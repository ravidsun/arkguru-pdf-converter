PY ?= python
CFG ?= config/config.yaml
install:
	pip install -r requirements.txt
test:
	pip install -e "../arkguru-common[test]"
	$(PY) -m pytest tests -q
phase2: ; $(PY) -m phase2_web.pipeline --config $(CFG)
init-db:
	$(PY) -m phase2_web.pipeline --config $(CFG) --init-db
worker:
	$(PY) -m phase2_web.worker --once
.PHONY: install test phase2 init-db worker
