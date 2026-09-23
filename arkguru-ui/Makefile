PY ?= ../.venv/bin/python
install:
	$(PY) -m pip install -r requirements.txt
	cd frontend && npm install
dev:
	bash scripts/start.sh
api:
	$(PY) -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --reload
ui:
	cd frontend && npm run dev
test:
	$(PY) -m pytest tests -q
.PHONY: install dev api ui test
