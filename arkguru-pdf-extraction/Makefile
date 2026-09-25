PY ?= python
CFG ?= config/config.yaml
install:
	pip install -r requirements.txt
test:
	pip install -e "../arkguru-common[test]"
	$(PY) -m pytest ../arkguru-common/tests -q
sample-pdf: ; $(PY) scripts/make_sample_pdf.py
phase1: ; $(PY) -m phase1_pdf.pipeline --config $(CFG)
init-db: ## Create datastore tables (chunks, chunk_embeddings) if missing
	$(PY) -m phase1_pdf.pipeline --config $(CFG) --init-db
worker: ## Autonomous worker: watch data/raw_pdfs and ingest new PDFs
	$(PY) -m phase1_pdf.worker --interval 30
.PHONY: install test sample-pdf phase1 init-db worker
