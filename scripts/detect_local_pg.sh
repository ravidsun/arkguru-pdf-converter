#!/usr/bin/env bash
#
# Probe native Postgres (host 5432) vs local Docker Postgres (host 5433)
# and publish PG_DSN to gitignored .env files. Never prints the DSN.
#
# Does not start servers. If nothing answers:
#   docker compose up -d
#   # or native: Homebrew/apt, or arkguru-common/scripts/start_services.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '[detect_local_pg] %s\n' "$*"; }

find_common() {
  local candidate="$REPO_ROOT/arkguru-common"
  if [ -f "$candidate/common/local_pg.py" ]; then
    echo "$candidate"
    return 0
  fi
  return 1
}

find_python() {
  if [ -x "${ARKGURU_VENV:-$REPO_ROOT/.venv}/bin/python" ]; then
    echo "${ARKGURU_VENV:-$REPO_ROOT/.venv}/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

write_local_env() {
  local dest="$1"
  umask 077
  {
    echo "# Written by scripts/detect_local_pg.sh. Do not commit."
    echo "# origin=${DSN_ORIGIN}"
    printf 'PG_DSN=%s\n' "$PG_DSN"
  } > "$dest"
}

if ! COMMON_DIR="$(find_common)"; then
  echo "[detect_local_pg] could not find $REPO_ROOT/arkguru-common." >&2
  exit 1
fi
if ! PY="$(find_python)"; then
  echo "[detect_local_pg] python3 not found." >&2
  exit 1
fi

EXPORT="$(mktemp)"
cleanup() { rm -f "$EXPORT"; }
trap cleanup EXIT

if ! PYTHONPATH="${COMMON_DIR}${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m common.local_pg --export-env "$EXPORT"; then
  exit 1
fi

# shellcheck disable=SC1090
set -a
# The file is KEY=VALUE only (DSN_ORIGIN, PG_DSN).
. "$EXPORT"
set +a

if [ -z "${PG_DSN:-}" ] || [ -z "${DSN_ORIGIN:-}" ]; then
  echo "[detect_local_pg] export file missing PG_DSN or DSN_ORIGIN." >&2
  exit 1
fi

PUBLISH="${COMMON_DIR}/scripts/publish_pg_dsn.sh"
if [ -x "$PUBLISH" ] || [ -f "$PUBLISH" ]; then
  REPOS_PARENT="${REPOS_PARENT:-$REPO_ROOT}"
  export PG_DSN DSN_ORIGIN REPOS_PARENT
  bash "$PUBLISH"
else
  write_local_env "${REPO_ROOT}/.env"
  log "wrote .env in this repo (origin=${DSN_ORIGIN}; publish_pg_dsn.sh not found)"
fi
