#!/usr/bin/env bash
#
# Set up the arkguru development environment.
#
# This repo is the only checkout. Phase folders (arkguru-pdf-extraction,
# arkguru-web-scraping, arkguru-rag-slm) and arkguru-common live nested here.
# They import the shared `common` package from ./arkguru-common.
#
# This script is idempotent and safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ARKGURU_VENV:-$REPO_ROOT/.venv}"

PHASES=(arkguru-pdf-extraction arkguru-web-scraping arkguru-rag-slm)

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# --- Locate nested phase folders -------------------------------------------
find_checkout() {
  local name="$1"
  local marker="$2"
  local candidate="$REPO_ROOT/$name"
  if [ -f "$candidate/$marker" ]; then
    echo "$candidate"
    return 0
  fi
  return 1
}

find_phase() { find_checkout "$1" "requirements.txt"; }

find_common() { find_checkout "arkguru-common" "pyproject.toml"; }

ensure_venv_packages() {
  if python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    log "ERROR: python3 venv/ensurepip is missing and apt-get is not available."
    return 1
  fi
  log "Installing python3-venv and python3-pip (ensurepip is missing)"
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-pip
}

ensure_ocr_packages() {
  if command -v tesseract >/dev/null 2>&1 && command -v gs >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    log "WARNING: tesseract/ghostscript missing and apt-get is not available. Scanned-PDF OCR will be skipped."
    return 0
  fi
  log "Installing tesseract-ocr, tesseract-ocr-eng, and ghostscript for ocrmypdf"
  sudo apt-get update -qq
  sudo apt-get install -y tesseract-ocr tesseract-ocr-eng ghostscript
}

venv_usable() {
  [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1
}

# --- 1. Python virtualenv ---------------------------------------------------
log "Creating virtualenv at $VENV_DIR"
ensure_venv_packages
ensure_ocr_packages
if ! venv_usable; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# --- 2. Shared package -------------------------------------------------------
if ! COMMON_DIR="$(find_common)"; then
  log "ERROR: could not find $REPO_ROOT/arkguru-common."
  exit 1
fi
log "Installing arkguru-common from $COMMON_DIR (schema/tokenizer/chunking/datastore/worker/rrf)"
pip install -e "${COMMON_DIR}[parquet,postgres,test]"

# --- 3. Phase dependencies ---------------------------------------------------
# The phase requirements pin `-e ../arkguru-common`; we install that path
# explicitly above, so we strip the line to avoid a second, conflicting checkout.
install_reqs() {
  local repo="$1"; shift
  local extra=("$@")
  log "Installing requirements for $(basename "$repo")"
  grep -viE '(^|[[:space:]])-e[[:space:]]+\.\./arkguru-common' "$repo/requirements.txt" \
    > /tmp/arkguru-reqs.txt || true
  pip install -r /tmp/arkguru-reqs.txt
  if [ "${#extra[@]}" -gt 0 ]; then
    pip install "${extra[@]}"
  fi
}

for name in "${PHASES[@]}"; do
  if repo="$(find_phase "$name")"; then
    case "$name" in
      # Phase 1 needs reportlab to generate the built-in sample PDF fixture,
      # plus psycopg/pgvector for --sink postgres (also pulled by common[postgres]),
      # plus ocrmypdf/pytesseract/pillow even if an old extraction checkout still
      # comments those lines.
      arkguru-pdf-extraction) install_reqs "$repo" "reportlab>=4.0" "psycopg[binary]>=3.2" "pgvector>=0.3" "ocrmypdf>=16.0" "pytesseract>=0.3.10" "pillow>=10.0" ;;
      # Phase 3's full requirements.txt pulls a heavy ML/serving stack
      # (transformers, sentence-transformers, FlagEmbedding, ragas, llama-index,
      # ...). For a lean, offline-capable dev environment we install only the
      # core needed to run the pipeline end-to-end with the `hashing` embedder +
      # BM25 retrieval. Install the full requirements.txt for GPU/production work.
      arkguru-rag-slm) pip install "numpy>=1.24" "rank-bm25>=0.2.2" ;;
      *) install_reqs "$repo" ;;
    esac
  else
    log "WARNING: could not find $REPO_ROOT/$name (skipping)."
  fi
done

log "Done. Activate with:  source \"$VENV_DIR/bin/activate\""
