#!/usr/bin/env bash
#
# Load the portable rag corpus into local Postgres + pgvector.
#
# The complete snapshot (2026-09-15) has 60 PDF sources, 109163 chunks, and
# 109163 embeddings. Native Windows / apt installs create an empty `rag`
# database; this script fills it. Does not target a remote PG_DSN unless you
# pass --force-remote.
#
#   bash scripts/restore_rag_dump.sh
#   bash scripts/restore_rag_dump.sh --dump /path/to/arkguru_rag_complete_20260915.dump
#   bash scripts/restore_rag_dump.sh --from-dsn "$OTHER_PG_DSN"
#
set -euo pipefail

DUMP_URL="${DUMP_URL:-https://filebin.net/arkguru-complete/arkguru_rag_complete_20260915.dump}"
DUMP_SHA256="90b21d209c211b48378cf951e349c8ea954cd65f4804385d654659c4d222968c"
EXPECTED_CHUNKS=109163
EXPECTED_EMBEDDINGS=109163
EXPECTED_SOURCES=60

HOST="127.0.0.1"
PORT="5432"
USER_NAME="rag"
PASSWORD="change-me"
DB_NAME="rag"
SUPER_USER="postgres"
SUPER_PASSWORD=""
DUMP_PATH=""
FROM_DSN=""
FORCE_REMOTE=0
SKIP_DOWNLOAD=0
TARGET_DSN=""

usage() {
  cat <<'EOF'
Load the portable rag dump (or a live --from-dsn snapshot) into local Postgres.

  --dump PATH       Existing custom-format dump (skips download)
  --from-dsn URI    pg_dump that URI, then restore locally (not the portable dump)
  --dsn URI         Restore target (default: postgresql://rag:change-me@127.0.0.1:5432/rag)
  --host HOST       Target host (default 127.0.0.1)
  --port PORT       Target port (default 5432; Docker compose uses 5433)
  --user NAME       App role (default rag)
  --password PASS   App password (default change-me)
  --db NAME         Database (default rag)
  --super-user NAME Superuser for CREATE EXTENSION / pg_restore (default postgres)
  --super-password  Superuser password (peer/socket auth if empty)
  --force-remote    Allow a non-loopback restore target
  -h, --help        Show this help
EOF
}

log() { printf '[restore_rag_dump] %s\n' "$*"; }
die() { printf '[restore_rag_dump] %s\n' "$*" >&2; exit 1; }

is_loopback_dsn() {
  local dsn="$1"
  case "$dsn" in
    *'@127.0.0.1:'*|*'@localhost:'*|*'@[::1]:'*|*'@127.0.0.1/'*|*'@localhost/'*) return 0 ;;
    *) return 1 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dump) DUMP_PATH="${2:-}"; shift 2 ;;
    --from-dsn) FROM_DSN="${2:-}"; shift 2 ;;
    --dsn) TARGET_DSN="${2:-}"; shift 2 ;;
    --host) HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --user) USER_NAME="${2:-}"; shift 2 ;;
    --password) PASSWORD="${2:-}"; shift 2 ;;
    --db) DB_NAME="${2:-}"; shift 2 ;;
    --super-user) SUPER_USER="${2:-}"; shift 2 ;;
    --super-password) SUPER_PASSWORD="${2:-}"; shift 2 ;;
    --force-remote) FORCE_REMOTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

command -v pg_restore >/dev/null || die "pg_restore not found. Install postgresql-client-16."
command -v psql >/dev/null || die "psql not found. Install postgresql-client-16."

if [ -z "$TARGET_DSN" ]; then
  TARGET_DSN="postgresql://${USER_NAME}:${PASSWORD}@${HOST}:${PORT}/${DB_NAME}"
fi

if ! is_loopback_dsn "$TARGET_DSN" && [ "$FORCE_REMOTE" -ne 1 ]; then
  die "refusing to restore into a remote DSN (would overwrite hosted data). Pass --force-remote if you mean it."
fi

workdir="${TMPDIR:-/tmp}/arkguru-rag-restore"
mkdir -p "$workdir"

if [ -n "$FROM_DSN" ]; then
  DUMP_PATH="${workdir}/from-dsn.dump"
  log "dumping source database to ${DUMP_PATH}"
  pg_dump -Fc --no-owner --no-acl --dbname="$FROM_DSN" --file="$DUMP_PATH"
  SKIP_DOWNLOAD=1
fi

if [ -z "$DUMP_PATH" ]; then
  for candidate in \
    "/opt/cursor/artifacts/arkguru_rag_complete_20260915.dump" \
    "${PWD}/arkguru_rag_complete_20260915.dump" \
    "${workdir}/arkguru_rag_complete_20260915.dump"
  do
    if [ -f "$candidate" ]; then
      DUMP_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "$DUMP_PATH" ]; then
  DUMP_PATH="${workdir}/arkguru_rag_complete_20260915.dump"
fi

if [ ! -f "$DUMP_PATH" ]; then
  log "downloading portable dump"
  if command -v curl >/dev/null; then
    curl -fL --retry 3 -o "$DUMP_PATH" "$DUMP_URL"
  else
    die "dump not found and curl is missing. Pass --dump PATH."
  fi
fi

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  actual="$(sha256sum "$DUMP_PATH" | awk '{print $1}')"
  if [ "$actual" != "$DUMP_SHA256" ]; then
    die "sha256 mismatch for ${DUMP_PATH}: got ${actual}, expected ${DUMP_SHA256}"
  fi
  log "dump sha256 ok (${DUMP_SHA256})"
fi

super_dsn="postgresql://${SUPER_USER}@${HOST}:${PORT}/${DB_NAME}"
restore_env=()
if [ -n "$SUPER_PASSWORD" ]; then
  restore_env=(env "PGPASSWORD=${SUPER_PASSWORD}")
fi

log "ensuring vector extension on ${HOST}:${PORT}/${DB_NAME}"
if [ "${#restore_env[@]}" -gt 0 ]; then
  "${restore_env[@]}" psql "$super_dsn" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;" \
    || "${restore_env[@]}" psql "$TARGET_DSN" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
else
  if command -v sudo >/dev/null && id postgres >/dev/null 2>&1; then
    sudo -u postgres psql -d "$DB_NAME" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
  else
    psql "$TARGET_DSN" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
  fi
fi

log "restoring ${DUMP_PATH}"
if command -v sudo >/dev/null && id postgres >/dev/null 2>&1 && [ -z "$SUPER_PASSWORD" ]; then
  sudo -u postgres pg_restore --no-owner --no-acl --clean --if-exists -d "$DB_NAME" "$DUMP_PATH"
else
  "${restore_env[@]}" pg_restore --no-owner --no-acl --clean --if-exists --dbname="$super_dsn" "$DUMP_PATH" \
    || "${restore_env[@]}" pg_restore --no-owner --no-acl --clean --if-exists --dbname="$TARGET_DSN" "$DUMP_PATH"
fi

verify_sql='
SELECT count(*) AS chunks FROM chunks;
SELECT count(*) AS embeddings FROM chunk_embeddings;
SELECT count(DISTINCT source_id) AS sources FROM chunks;
'
log "verifying row counts"
counts="$(psql "$TARGET_DSN" -v ON_ERROR_STOP=1 -A -t -c \
  "SELECT count(*)::text || ' ' || (SELECT count(*) FROM chunk_embeddings)::text || ' ' || (SELECT count(DISTINCT source_id) FROM chunks)::text FROM chunks;")"
read -r got_chunks got_emb got_src <<<"$counts"
log "chunks=${got_chunks} embeddings=${got_emb} sources=${got_src}"

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  if [ "$got_chunks" != "$EXPECTED_CHUNKS" ] || [ "$got_emb" != "$EXPECTED_EMBEDDINGS" ] || [ "$got_src" != "$EXPECTED_SOURCES" ]; then
    die "count mismatch. expected chunks=${EXPECTED_CHUNKS} embeddings=${EXPECTED_EMBEDDINGS} sources=${EXPECTED_SOURCES}"
  fi
  log "portable dump counts match"
else
  log "live --from-dsn restore; not comparing to portable dump counts"
fi

psql "$TARGET_DSN" -c "$verify_sql"
log "done. PG_DSN=${TARGET_DSN}"
