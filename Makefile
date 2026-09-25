PY ?= .venv/bin/python
ui:
	$(PY) -m uvicorn backend.app:app --app-dir arkguru-ui --host 127.0.0.1 --port 8080
.PHONY: ui
