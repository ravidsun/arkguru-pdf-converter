#!/usr/bin/env bash
#
# Set up the arkguru development environment.
#
# The three phase repos (arkguru-pdf-extraction, arkguru-web-scraping,
# arkguru-rag-slm) all import the shared `common` package, which is provided by
# `arkguru-common`. That package used to be a separate (never-published) repo;
# its source now lives here, vendored under ./arkguru-common, and is installed
# into the virtualenv so `import common` works everywhere without depending on a
# fragile `../arkguru-common` sibling checkout.
#
# This script is idempotent and safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
COMMON_DIR="$REPO_ROOT/arkguru-common"
VENV_DIR="${ARKGURU_VENV:-$REPO_ROOT/.venv}"

PHASES=(arkguru-pdf-extraction arkguru-web-scraping arkguru-rag-slm)

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# --- Locate each phase repo -------------------------------------------------
# On this platform the repos are checked out as flat siblings of this repo, but
# the project can also be laid out with them nested underneath it. Support both.
find_phase() {
  local name="$1"
  for candidate in "$PARENT_DIR/$name" "$REPO_ROOT/$name"; do
    if [ -f "$candidate/requirements.txt" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# --- 1. Python virtualenv ---------------------------------------------------
log "Creating virtualenv at $VENV_DIR"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# --- 2. Shared package -------------------------------------------------------
log "Installing arkguru-common (shared schema/tokenizer/chunking/datastore/worker/rrf)"
pip install -e "${COMMON_DIR}[parquet,test]"

# --- 3. Phase dependencies ---------------------------------------------------
# The phase requirements pin `-e ../arkguru-common`; we install that path
# explicitly above, so we strip the line to avoid depending on a sibling layout.
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
