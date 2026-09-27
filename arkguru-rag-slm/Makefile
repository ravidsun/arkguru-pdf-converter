PY ?= python
CFG ?= config/config.yaml
install: ; pip install -r requirements.txt
combine: ; $(PY) -m phase3_rag.prepare_dataset --inputs data/processed/pdf_chunks.jsonl data/processed/web_chunks.jsonl
finetune: ; $(PY) -m phase3_rag.finetune_qlora --config $(CFG)
index: ; $(PY) -m phase3_rag.index --config $(CFG)
embed-db: ## Fill chunk_embeddings (bge-m3). REEMBED=1 SCHEMA=public DEVICE=cuda
	$(PY) -m phase3_rag.embed_datastore --model BAAI/bge-m3 --dim 1024 \
		$(if $(REEMBED),--reembed) $(if $(DEVICE),--device $(DEVICE)) \
		$(if $(SCHEMA),--schema $(SCHEMA)) $(if $(BATCH),--batch $(BATCH)) \
		$(if $(ALLOW_HASHING),--allow-hashing --embedder hashing)
serve: ; $(PY) -m phase3_rag.serve --config $(CFG)
eval: ; $(PY) -m phase3_rag.eval_ragas --config $(CFG)
golden: ## Rebuild the stratified golden set (Gemini when GEMINI_API_KEY is set, else offline)
	$(PY) -m phase3_rag.build_golden --per-stratum $(or $(PER),25)
eval-retrieval: ## Retrieval-only metrics per stratum / block_type / source_type (no Ollama, no RAGAS)
	$(PY) -m phase3_rag.eval_retrieval --config $(CFG) $(if $(EMB),--embedder $(EMB)) $(if $(OUT),--out $(OUT)) $(if $(BASE),--compare $(BASE)) $(if $(ALLOW_MISMATCH),--allow-model-mismatch)
test: ; $(PY) -m pytest tests -q
backup: ## Dump chunks + embeddings if the DB watermark moved
	$(PY) -m phase3_rag.backup --once
.PHONY: install combine finetune index embed-db serve eval golden eval-retrieval test from-pdfs worker backup orchestrate orchestrate-once e2e
from-pdfs: ## PDFs -> Phase1 -> Phase3 ingest (PDFS=dir [EMB=sentence_transformer] [ASK="q"] [SINK=postgres])
	$(PY) -m phase3_rag.run_pdfs --pdfs $(PDFS) $(if $(EMB),--embedder $(EMB)) $(if $(ASK),--ask "$(ASK)") $(if $(filter postgres,$(SINK)),--sink postgres) $(if $(ALLOW_HASHING),--allow-hashing)
worker: ## Autonomous worker: embed chunks that lack vectors
	$(PY) -m phase3_rag.worker --interval 60
orchestrate: ## Run the full pipeline (ingest -> crawl -> index) on a schedule
	$(PY) orchestrator.py --interval 300
orchestrate-once: ## Run one full pipeline pass and exit
	$(PY) orchestrator.py --once
e2e: ## End-to-end on a PDF, offline/no-DB (ingest -> answer)
	$(PY) orchestrator.py --once --config orchestrator.filemode.yaml
