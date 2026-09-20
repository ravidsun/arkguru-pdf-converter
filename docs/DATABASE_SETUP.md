# Database setup — PostgreSQL + pgvector

The pipeline stores data in **two Postgres tables** (`chunks` = text + metadata,
`chunk_embeddings` = vectors) using the **pgvector** extension. You do **not**
write DDL by hand — `ensure_schema()` in `common/datastore.py` creates the
tables and indexes. Provide:

1. PostgreSQL 14+ (16 recommended) with pgvector 0.5+,
2. a database + login role,
3. a connection string in **`PG_DSN`**.

The app is one client. Switch hosts by changing `PG_DSN` only. There is no
provider-specific backend.

| Where | Host port | Typical `PG_DSN` |
| --- | --- | --- |
| Native local (Homebrew / apt / `start_services.sh`) | **5432** | `postgresql://arkguru:arkguru@127.0.0.1:5432/arkguru` or `postgresql://rag:change-me@127.0.0.1:5432/rag` |
| Local Docker ([compose.yaml](../compose.yaml)) | **5433** | `postgresql://rag:change-me@127.0.0.1:5433/rag` |
| Hosted Supabase | session pooler **5432** | `postgresql://postgres.PROJECT:PASS@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require` |
| Other hosted pgvector (Neon, Crunchy, RDS, Aiven, …) | vendor port | session/direct URI; `sslmode=require` is added if omitted |

Loopback hosts (`localhost`, `127.0.0.1`, `::1`) are left without TLS. Remote
hosts get `sslmode=require` when the URI omitted it. Override with `PG_SSLMODE`
or an explicit `?sslmode=`. Prefer a **session or direct** URI — transaction
poolers break the named cursor in `iter_missing_embeddings`.

`.env` is gitignored. Copy [`.env.example`](../.env.example) → `.env`, or run
the one-click Docker script / detect helper below. Never commit a password.

---

## One-click local Docker

From this repo (requires Docker Engine or Desktop already installed):

```bash
bash scripts/setup_docker_pg.sh
```

That pulls `pgvector/pgvector:pg16`, starts [compose.yaml](../compose.yaml) on
host port **5433**, writes gitignored `.env`, and prints:

| Field | Value |
| --- | --- |
| host | `127.0.0.1` |
| port | `5433` (`ARKGURU_PG_PORT` to override) |
| database | `rag` |
| user | `rag` |
| password | `change-me` |
| `PG_DSN` | `postgresql://rag:change-me@127.0.0.1:5433/rag` |
| `psql` | `PGPASSWORD=change-me psql -h 127.0.0.1 -p 5433 -U rag -d rag` |

An already-exported `PG_DSN` (hosted Supabase/Neon) still wins over `.env`.
`unset PG_DSN` to use Docker. Use `bash scripts/detect_local_pg.sh` when a
server is already running and you only need `.env` written.

---

## Hybrid retrieve

When `PG_DSN` is set, Phase 3 retrieve is one SQL function, `search_chunks()`,
created by `ensure_schema()` in `common/datastore.py` (not a hand-written SQL
file). It fuses HNSW cosine on `chunk_embeddings` with GIN `ts` on `chunks`
using reciprocal rank fusion (`is_parent = false` on both legs). Python still
embeds the query, reranks, and expands parents. File/npz mode keeps Python RRF.

---

## Detect local native vs Docker

If a native cluster and Docker compose can both exist on one machine, they use
**different host ports** (5432 vs 5433). From this repo:

```bash
# start one or both, then:
bash scripts/detect_local_pg.sh
```

`python -m common.local_pg` probes, in order:

1. Existing `PG_DSN` whose host is **not** loopback (injected hosted URI) — keep it.
2. Existing `PG_DSN` if it already connects.
3. Docker on `127.0.0.1:5433` (`ARKGURU_PG_PORT` overrides the port).
4. Native candidates on `127.0.0.1:5432` (`arkguru` / `rag` / `$USER`).

The CLI prints `origin=` and `host:port/db` only (never the password). The
shell helper writes gitignored `.env` files (mode `0600`). It does **not** start
Postgres. If nothing answers:

```bash
# native
#   macOS: brew services start postgresql@16
#   Linux: sudo pg_ctlcluster 16 main start
#   Cloud: bash ../arkguru-common/scripts/start_services.sh
# Docker
docker compose up -d
```

Then re-run `bash scripts/detect_local_pg.sh`.

---

## Option A — Docker (local or VPS)

Image `pgvector/pgvector:pg16`. Host port **5433** so native 5432 can coexist.
Override with `ARKGURU_PG_PORT`.

```bash
# from arkguru-pdf-converter — pull image, start, write .env, print DSN
bash scripts/setup_docker_pg.sh
```

Manual equivalent:

```bash
docker compose up -d
export PG_DSN=postgresql://rag:change-me@127.0.0.1:5433/rag
```

Equivalent one-liner:

```bash
docker run -d --name arkguru-pg \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=change-me \
  -e POSTGRES_DB=rag \
  -p 5433:5432 \
  -v arkguru_pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

The app creates the `vector` extension on first write. By hand:

```bash
docker exec -it arkguru-pg psql -U rag -d rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

---

## Option B — Native local install

### macOS (Homebrew)

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb rag
psql rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
export PG_DSN="postgresql://$(whoami)@127.0.0.1:5432/rag"
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y postgresql postgresql-16-pgvector   # match your PG major
sudo -u postgres psql <<'SQL'
CREATE ROLE rag LOGIN PASSWORD 'change-me';
CREATE DATABASE rag OWNER rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
SQL
export PG_DSN="postgresql://rag:change-me@127.0.0.1:5432/rag"
```

If `postgresql-16-pgvector` is missing from apt:

```bash
sudo apt install -y build-essential postgresql-server-dev-16 git
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

A fresh native install (Windows `setup_windows_pg.ps1`, apt, or Homebrew) has
**schema-ready but empty** tables until you ingest PDFs or restore the portable
dump. Hosted `PG_DSN` (Supabase / Neon) can also be a **smaller subset**. The
complete 2026-09-15 snapshot is 60 PDF sources / 109163 chunks /
109163 embeddings (`vector` + HNSW + GIN).

```bash
# Linux / macOS / Cloud Agent — default target 127.0.0.1:5432/rag
bash scripts/restore_rag_dump.sh
# Docker compose on 5433:
# bash scripts/restore_rag_dump.sh --port 5433
```

```powershell
# Windows native (after setup_windows_pg.cmd so pgvector exists).
# The .cmd fetches restore_rag_dump.ps1 when you only downloaded the launcher.
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ravidsun/arkguru-pdf-converter/refs/heads/cursor/restore-rag-dump-e7c4/scripts/restore_rag_dump.cmd" -OutFile .\restore_rag_dump.cmd
.\restore_rag_dump.cmd
```

The script downloads
`https://filebin.net/arkguru-complete/arkguru_rag_complete_20260915.dump`
when the file is not already on disk, checks
`sha256=90b21d209c211b48378cf951e349c8ea954cd65f4804385d654659c4d222968c`,
runs `pg_restore --no-owner --no-acl`, and fails if counts do not match.
Windows must use `curl.exe` (not `Invoke-WebRequest`): Filebin returns an HTML
page to the PowerShell user-agent. EDB Postgres has no `vector.control`; the
restore script copies the unofficial Windows pgvector 0.8.6 zip into
`C:\Program Files\PostgreSQL\16` (needs an elevated PowerShell).
It refuses a non-loopback target unless you pass `--force-remote`. To copy a
**live** other database instead of the portable dump:

```bash
bash scripts/restore_rag_dump.sh --from-dsn "$OTHER_PG_DSN"
```

Expected after a portable restore:

```bash
psql "$PG_DSN" -c "SELECT count(*) FROM chunks;"
psql "$PG_DSN" -c "SELECT count(*) FROM chunk_embeddings;"
psql "$PG_DSN" -c "SELECT count(DISTINCT source_id) FROM chunks;"
# 109163 / 109163 / 60
```

Cloud Agent VMs can use `arkguru-common/scripts/start_services.sh`, which
creates role/db `arkguru` on **5432**:

```
postgresql://arkguru:arkguru@127.0.0.1:5432/arkguru
```

---

## Option C — VPS (Ubuntu, remote access)

Use this when Phases 1/2 run in the cloud and write to a shared database, or
when the NUC (Phase 3) connects to a remote DB. You can also run
[compose.yaml](../compose.yaml) on the VPS (still host port 5433 unless you
set `ARKGURU_PG_PORT`).

**1. Install Postgres + pgvector** (Ubuntu steps in Option B, or Docker).

**2. Create the database and role**

```bash
sudo -u postgres psql <<'SQL'
CREATE ROLE rag LOGIN PASSWORD 'use-a-strong-password';
CREATE DATABASE rag OWNER rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

**3. Allow remote connections** only if you cannot use an SSH tunnel:

```bash
# /etc/postgresql/16/main/postgresql.conf
listen_addresses = '*'

# /etc/postgresql/16/main/pg_hba.conf  -- restrict to YOUR client IP, use scram
host  rag  rag  <your.client.ip>/32  scram-sha-256

sudo systemctl restart postgresql
```

**4. Firewall — never open 5432 to the world**

```bash
sudo ufw allow from <your.client.ip> to any port 5432
sudo ufw enable
```

**5. Connection string (TLS for remote)**

```bash
export PG_DSN="postgresql://rag:use-a-strong-password@<vps.host>:5432/rag?sslmode=require"
```

### Safer: SSH tunnel

Keep Postgres on localhost on the VPS and tunnel:

```bash
ssh -N -L 5432:localhost:5432 user@<vps.host>
export PG_DSN="postgresql://rag:use-a-strong-password@127.0.0.1:5432/rag"
```

---

## Option D — Supabase (hosted)

The app still talks **Postgres over `PG_DSN`** (`psycopg` + pgvector). There is
no REST/JS vector client.

Cloud Agents are **IPv4-only**. The Direct URI (`db.<ref>.supabase.co:5432`) is
IPv6-only and fails with `Network is unreachable`. Use **Session pooler**
(Supavisor session mode), not Direct and not transaction-mode port **6543**.

**What is already on the project.** Tables `chunks` and `chunk_embeddings`
exist. The `vector` extension is in schema `extensions`. RLS is **on** with no
anon/authenticated policies, so the Data API cannot read the corpus. The
`postgres` role in `PG_DSN` still has full table access. Do not use the
publishable (anon) key for ingest or retrieve.

**Connection string.** Dashboard: **Settings → Database** / **Connect →
Session pooler**. Port **5432**. User is `postgres.<project-ref>`. Append
`sslmode=require`:

```
postgresql://postgres.PROJECT_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
```

Store it as a Cursor **Runtime Secret** or **Environment Variable** named
`PG_DSN` (not a Build Secret). The detect helper will **not** overwrite a
non-loopback `PG_DSN`.

**`.env` (never commit secrets).** Copy [`.env.example`](../.env.example) →
`.env` in this repo, or `arkguru-rag-slm/.env.example` → `.env`. Phase 1 loads
`.env` from the cwd or a parent up to the git root via `resolve_dsn`.

PowerShell for one session:

```powershell
$env:PG_DSN="postgresql://postgres.PROJECT_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require"
```

**Two tables.** Phase 1 `--sink postgres` fills **`chunks`** only
(`chunk_index` is 0-based). Phase 3 `embed_datastore` fills
**`chunk_embeddings`**. In the Table Editor, inspect `chunks.chunk_index` (a
`0` can look blank).

```bash
cd arkguru-pdf-extraction
python -m phase1_pdf.pipeline --input data/raw_pdfs --sink postgres

cd arkguru-rag-slm
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
python -m phase3_rag.serve
```

`phase3_rag.serve` is **fail-closed**: if `PG_DSN` is set and connect fails, it
**raises**. Retrieve is SQL `search_chunks()`. JSONL retrieve is used only when
no DSN is configured.

**Daily backup** (custom-format dump of both tables when the `created_at`
watermark moved). Requires `pg_dump`:

```bash
cd arkguru-rag-slm
python -m phase3_rag.backup --once
# or: make backup
```

Dumps: `data/backups/arkguru-YYYYMMDD-HHMMSS.dump` (gitignored; last 7 kept).

```bash
pg_restore --clean --if-exists -d "$PG_DSN" data/backups/<file>.dump
```

```bash
cd arkguru-rag-slm
python -m phase3_rag.sources    # PDF source_id vs embedded counts
```

---

## Option E — Other hosted Postgres + pgvector

Any managed PostgreSQL 14+ that allows `CREATE EXTENSION vector` is a drop-in:
Neon, Crunchy Bridge, Timescale / Tiger Cloud, Aiven, AWS RDS / Aurora, Google
Cloud SQL / AlloyDB, Azure Database for PostgreSQL, DigitalOcean, Railway,
Render. Use a **session or direct** URI.

```
postgresql://user:password@YOUR_HOST:5432/YOUR_DB?sslmode=require
```

Self-hosted Supabase is optional if you want their dashboard; the app still
connects with `PG_DSN`.

---

## Point the pipeline at the database

Datastore table names and embedding `dim` live in each phase’s
`config/datastore.yaml`. The connection string comes from `PG_DSN`.

```bash
# this repo
cp .env.example .env
# or: bash scripts/detect_local_pg.sh

# Phase 3 example file also works; resolve_dsn walks parents to a git root
# cp ../arkguru-rag-slm/.env.example ../arkguru-rag-slm/.env
```

To use different table names or dimension, edit `chunks_table`,
`vectors_table`, and `dim`. No code changes.

---

## Verify

```bash
psql "$PG_DSN" -c "SELECT version();"
psql "$PG_DSN" -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"

python - <<'PY'
from common.datastore_config import open_chunk_store
s = open_chunk_store()
s.ensure_schema()
print("connected OK. chunks:", s.count(), "| vectors:", s.count_vectors())
PY
```

Expected: tables `chunks` and `chunk_embeddings` exist. First write also
creates them. Explicit create:

```bash
python -m phase1_pdf.pipeline --init-db
```

---

## Operating notes

- **Tables are auto-managed.** `ensure_schema()` is idempotent on every write path
  and installs `search_chunks()` (hybrid retrieve). Do not put that function in a
  hand-written schema file.
- **Re-embed with a new model:** `TRUNCATE chunk_embeddings;` then re-run
  `phase3_rag.embed_datastore`. If `dim` changes, update `config/datastore.yaml`
  and drop/recreate `chunk_embeddings` (vector width is fixed at create).
- **Backups:** `python -m phase3_rag.backup --once` / `make backup`, or
  `pg_dump "$PG_DSN"`.
- **Sizing:** each embedding is `dim * 4` bytes (bge-m3 1024-d ≈ 4 KB/row) plus
  the HNSW index; chunk text is usually larger. 100k chunks fits a small VPS.
- **VPS security:** strong password + `scram-sha-256`, restrict `pg_hba.conf`,
  `sslmode=require` for remote clients, keep 5432 off the public internet.

---

## Related

- [compose.yaml](../compose.yaml) — local Docker pgvector
- [scripts/setup_docker_pg.sh](../scripts/setup_docker_pg.sh) — one-click pull, start, write `PG_DSN`
- [scripts/detect_local_pg.sh](../scripts/detect_local_pg.sh) — pick native vs Docker if already running
- [scripts/restore_rag_dump.sh](../scripts/restore_rag_dump.sh) — load the portable 60-source dump (Windows: `restore_rag_dump.cmd`)
- [LOCAL_RUN.md](LOCAL_RUN.md) — full local pipeline
- [CLOUD_RUN.md](CLOUD_RUN.md) — Cloud Agent + hosted `PG_DSN`
