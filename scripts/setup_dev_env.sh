#!/usr/bin/env bash
#
# Set up the arkguru development environment.
#
# Phase repos (arkguru-pdf-extraction, optional arkguru-web-scraping,
# arkguru-rag-slm) import the shared `common` package from arkguru-common.
# Prefer a sibling checkout of arkguru-common when present (first-class GitHub
# repo); fall back to the copy vendored under ./arkguru-common.
#
# This script is idempotent and safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
VENV_DIR="${ARKGURU_VENV:-$REPO_ROOT/.venv}"

PHASES=(arkguru-pdf-extraction arkguru-web-scraping arkguru-rag-slm)

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# --- Locate sibling or nested checkouts ------------------------------------
# Cloud Agents check out repos as flat siblings; local clones may nest them
# under this umbrella repo. Support both.
find_checkout() {
  local name="$1"
  local marker="$2"
  for candidate in "$PARENT_DIR/$name" "$REPO_ROOT/$name"; do
    if [ -f "$candidate/$marker" ]; then
      echo "$candidate"
      return 0
    fi
  done
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

venv_usable() {
  [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1
}

# --- 1. Python virtualenv ---------------------------------------------------
log "Creating virtualenv at $VENV_DIR"
ensure_venv_packages
if ! venv_usable; then
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# --- 2. Shared package -------------------------------------------------------
if ! COMMON_DIR="$(find_common)"; then
  log "ERROR: could not find arkguru-common (sibling checkout or ./arkguru-common)."
  exit 1
fi
log "Installing arkguru-common from $COMMON_DIR (schema/tokenizer/chunking/datastore/worker/rrf)"
pip install -e "${COMMON_DIR}[parquet,test]"

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
      # Phase 1 needs reportlab to generate the built-in sample PDF fixture.
      arkguru-pdf-extraction) install_reqs "$repo" "reportlab>=4.0" ;;
      # Phase 3's full requirements.txt pulls a heavy ML/serving stack
      # (transformers, sentence-transformers, FlagEmbedding, ragas, llama-index,
      # ...). For a lean, offline-capable dev environment we install only the
      # core needed to run the pipeline end-to-end with the `hashing` embedder +
      # BM25 retrieval. Install the full requirements.txt for GPU/production work.
      arkguru-rag-slm) pip install "numpy>=1.24" "rank-bm25>=0.2.2" ;;
      *) install_reqs "$repo" ;;
    esac
  else
    log "WARNING: could not find $name (skipping). Clone it as a sibling of this repo."
  fi
done

log "Done. Activate with:  source \"$VENV_DIR/bin/activate\""
