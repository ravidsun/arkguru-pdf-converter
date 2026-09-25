#!/usr/bin/env bash
#
# One-click local Docker Postgres + pgvector.
# Pulls pgvector/pgvector:pg16, starts compose.yaml, writes PG_DSN to .env,
# and prints the connection card.
#
#   bash scripts/setup_docker_pg.sh
#
# Does not install Docker Engine. Does not overwrite a remote PG_DSN in the
# current shell; .env is written for later dotenv loads. An already-exported
# PG_DSN still wins over .env (common.datastore_config.resolve_dsn).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/compose.yaml"
HOST="127.0.0.1"
PORT="${ARKGURU_PG_PORT:-5433}"
USER_NAME="rag"
PASSWORD="change-me"
DB_NAME="rag"
PG_DSN="postgresql://${USER_NAME}:${PASSWORD}@${HOST}:${PORT}/${DB_NAME}"
DSN_ORIGIN="docker"

log() { printf '[setup_docker_pg] %s\n' "$*"; }

die() { printf '[setup_docker_pg] %s\n' "$*" >&2; exit 1; }

find_common() {
  local candidate="$REPO_ROOT/arkguru-common"
  if [ -f "$candidate/scripts/publish_pg_dsn.sh" ] || [ -d "$candidate/common" ]; then
    echo "$candidate"
    return 0
  fi
  return 1
}

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" "$@"
  else
    return 1
  fi
}

write_local_env() {
  local dest="$1"
  umask 077
  {
    echo "# Written by scripts/setup_docker_pg.sh. Do not commit."
    echo "# origin=${DSN_ORIGIN}"
    printf 'PG_DSN=%s\n' "$PG_DSN"
  } > "$dest"
}

print_connection_card() {
  cat <<EOF

Connection (local Docker pgvector)
  host:      ${HOST}
  port:      ${PORT}
  database:  ${DB_NAME}
  user:      ${USER_NAME}
  password:  ${PASSWORD}
  PG_DSN:    ${PG_DSN}
  psql:      PGPASSWORD=${PASSWORD} psql -h ${HOST} -p ${PORT} -U ${USER_NAME} -d ${DB_NAME}

Wrote gitignored .env (origin=docker). To use it in this shell:
  set -a && . ${REPO_ROOT}/.env && set +a

If PG_DSN is already exported (hosted Supabase/Neon), it wins over .env.
Unset it to use Docker:  unset PG_DSN
EOF
}

[ -f "$COMPOSE_FILE" ] || die "missing ${COMPOSE_FILE}"

if ! command -v docker >/dev/null 2>&1; then
  die "Docker is not installed. Install Docker Desktop or Engine, then re-run.
  https://docs.docker.com/get-docker/"
fi
if ! docker info >/dev/null 2>&1; then
  die "Docker daemon is not running. Start Docker, then re-run."
fi
if ! compose version >/dev/null 2>&1 && ! compose --version >/dev/null 2>&1; then
  die "docker compose is not available. Install the Compose plugin, then re-run.
  https://docs.docker.com/compose/install/"
fi

log "pulling pgvector/pgvector:pg16"
compose pull

log "starting arkguru-pg on host port ${PORT}"
if ! compose up -d --wait; then
  log "--wait not supported; starting and polling health"
  compose up -d
  for _ in $(seq 1 36); do
    if docker exec arkguru-pg pg_isready -U "$USER_NAME" -d "$DB_NAME" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  if ! docker exec arkguru-pg pg_isready -U "$USER_NAME" -d "$DB_NAME" >/dev/null 2>&1; then
    die "container started but Postgres is not ready. Check: docker logs arkguru-pg"
  fi
fi

COMMON_DIR=""
if COMMON_DIR="$(find_common)"; then
  PUBLISH="${COMMON_DIR}/scripts/publish_pg_dsn.sh"
  if [ -f "$PUBLISH" ]; then
    REPOS_PARENT="${REPOS_PARENT:-$REPO_ROOT}"
    export PG_DSN DSN_ORIGIN REPOS_PARENT
    bash "$PUBLISH"
  else
    write_local_env "${REPO_ROOT}/.env"
    log "wrote ${REPO_ROOT}/.env (publish_pg_dsn.sh not found)"
  fi
else
  write_local_env "${REPO_ROOT}/.env"
  log "wrote ${REPO_ROOT}/.env"
fi

print_connection_card
