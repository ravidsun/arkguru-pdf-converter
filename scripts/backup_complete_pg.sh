#!/usr/bin/env bash
#
# Complete backup of the Postgres database (schema + data + extensions +
# indexes + constraints), plus cluster roles via pg_dumpall --globals-only.
#
# This is not the Phase 3 watermark dump, which copies only chunks +
# chunk_embeddings when created_at moved.
#
#   bash scripts/backup_complete_pg.sh
#   bash scripts/backup_complete_pg.sh --docker
#   bash scripts/backup_complete_pg.sh --out-dir /path/to/dir
#
# Restore the custom dump into an existing database:
#
#   pg_restore --no-owner --no-acl --dbname="$PG_DSN" data/backups/<file>.dump
#
# Recreate the database (connects to postgres, then CREATE DATABASE):
#
#   pg_restore --create --no-owner --no-acl --dbname=postgresql://rag:change-me@127.0.0.1:5433/postgres \
#     data/backups/<file>.dump
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_DOCKER=0
OUT_DIR="${ARKGURU_BACKUP_DIR:-${REPO_ROOT}/data/backups}"

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \?//'
}

log() { printf '[backup_complete_pg] %s\n' "$*"; }

die() { printf '[backup_complete_pg] %s\n' "$*" >&2; exit 1; }

load_dotenv() {
  local env_file="${REPO_ROOT}/.env"
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
}

parse_dsn() {
  python3 - "$1" <<'PY'
from urllib.parse import unquote, urlparse
import sys
raw = sys.argv[1].strip()
u = urlparse(raw)
if u.scheme not in ("postgres", "postgresql"):
    raise SystemExit("PG_DSN must be a postgresql:// URI")
host = u.hostname or "127.0.0.1"
port = str(u.port or 5432)
user = unquote(u.username or "")
password = unquote(u.password or "")
db = unquote((u.path or "").lstrip("/") or "postgres")
print(host)
print(port)
print(user)
print(password)
print(db)
PY
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --docker)
      FORCE_DOCKER=1
      shift
      ;;
    --out-dir)
      [ $# -ge 2 ] || die "--out-dir needs a directory"
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      die "unknown argument: $1 (try --help)"
      ;;
  esac
done

command -v pg_dump >/dev/null || die "pg_dump not on PATH. Install PostgreSQL client tools."
command -v pg_dumpall >/dev/null || die "pg_dumpall not on PATH. Install PostgreSQL client tools."
command -v pg_restore >/dev/null || die "pg_restore not on PATH. Install PostgreSQL client tools."
command -v psql >/dev/null || die "psql not on PATH. Install PostgreSQL client tools."
command -v python3 >/dev/null || die "python3 not on PATH."
command -v sha256sum >/dev/null || die "sha256sum not on PATH."

if [ "$FORCE_DOCKER" -eq 1 ]; then
  unset PG_DSN || true
  PG_DSN="postgresql://rag:change-me@127.0.0.1:${ARKGURU_PG_PORT:-5433}/rag"
  log "using local Docker defaults (--docker)"
elif [ -n "${PG_DSN:-}" ]; then
  log "using already-exported PG_DSN"
else
  load_dotenv
  if [ -z "${PG_DSN:-}" ]; then
    PG_DSN="postgresql://rag:change-me@127.0.0.1:${ARKGURU_PG_PORT:-5433}/rag"
    log "PG_DSN unset; using local Docker defaults"
  else
    log "using PG_DSN from .env"
  fi
fi

mapfile -t DSN_PARTS < <(parse_dsn "$PG_DSN")
HOST="${DSN_PARTS[0]}"
PORT="${DSN_PARTS[1]}"
USER_NAME="${DSN_PARTS[2]}"
PASSWORD="${DSN_PARTS[3]}"
DB_NAME="${DSN_PARTS[4]}"
[ -n "$USER_NAME" ] || die "PG_DSN has no username"
export PGPASSWORD="$PASSWORD"

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
PREFIX="${OUT_DIR}/arkguru-${DB_NAME}-${STAMP}"
DUMP_PATH="${PREFIX}.dump"
GLOBALS_PATH="${PREFIX}.globals.sql"
MANIFEST_PATH="${PREFIX}.manifest.txt"
TOC_PATH="${PREFIX}.toc.txt"

cleanup_partial() {
  rm -f "$DUMP_PATH" "$GLOBALS_PATH" "$MANIFEST_PATH" "$TOC_PATH"
}
trap cleanup_partial ERR

log "checking ${HOST}:${PORT}/${DB_NAME}"
psql --dbname="$PG_DSN" -v ON_ERROR_STOP=1 -c 'SELECT 1' >/dev/null

DB_SIZE="$(psql --dbname="$PG_DSN" -At -c "SELECT pg_size_pretty(pg_database_size(current_database()));")"
EXTENSIONS="$(psql --dbname="$PG_DSN" -At -c "SELECT extname || ' ' || extversion FROM pg_extension ORDER BY 1;" | paste -sd, -)"
TABLES="$(psql --dbname="$PG_DSN" -At -c "
SELECT format('%I.%I', n.nspname, c.relname)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1;
")"
COUNTS=""
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  n="$(psql --dbname="$PG_DSN" -At -c "SELECT count(*) FROM ${rel}")"
  COUNTS+="${rel} ${n}"$'\n'
done <<< "$TABLES"
if psql --dbname="$PG_DSN" -At -c "SELECT to_regclass('public.chunks')" | grep -q chunks; then
  COUNTS+="distinct_sources $(psql --dbname="$PG_DSN" -At -c "SELECT count(DISTINCT source_id) FROM chunks;")"$'\n'
fi

log "dumping complete database ${DB_NAME} (${DB_SIZE}) -> ${DUMP_PATH}"
pg_dump \
  --format=custom \
  --compress=6 \
  --create \
  --blobs \
  --verbose \
  --dbname="$PG_DSN" \
  --file="$DUMP_PATH"

log "dumping roles/globals -> ${GLOBALS_PATH}"
pg_dumpall \
  --globals-only \
  --host="$HOST" \
  --port="$PORT" \
  --username="$USER_NAME" \
  --file="$GLOBALS_PATH"

pg_restore --list "$DUMP_PATH" > "$TOC_PATH"
DUMP_SHA="$(sha256sum "$DUMP_PATH" | awk '{print $1}')"
GLOBALS_SHA="$(sha256sum "$GLOBALS_PATH" | awk '{print $1}')"
DUMP_BYTES="$(wc -c < "$DUMP_PATH" | tr -d ' ')"

{
  echo "taken_at_utc=${STAMP}"
  echo "source=${HOST}:${PORT}/${DB_NAME}"
  echo "pg_dump=$(pg_dump --version)"
  echo "db_size=${DB_SIZE}"
  echo "extensions=${EXTENSIONS}"
  echo "user_tables=${TABLES//$'\n'/, }"
  echo "row_counts:"
  printf '%s\n' "$COUNTS"
  echo "dump=${DUMP_PATH}"
  echo "dump_bytes=${DUMP_BYTES}"
  echo "dump_sha256=${DUMP_SHA}"
  echo "globals=${GLOBALS_PATH}"
  echo "globals_sha256=${GLOBALS_SHA}"
  echo "toc:"
  cat "$TOC_PATH"
} > "$MANIFEST_PATH"

trap - ERR
rm -f "$TOC_PATH"
chmod 600 "$DUMP_PATH" "$GLOBALS_PATH" "$MANIFEST_PATH"

log "wrote ${DUMP_PATH} (${DUMP_BYTES} bytes, sha256=${DUMP_SHA})"
log "wrote ${GLOBALS_PATH}"
log "wrote ${MANIFEST_PATH}"
log "do not git-commit these files"
echo "$DUMP_PATH"
