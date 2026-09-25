# Database Setup — PostgreSQL + pgvector

The pipeline stores data in **two Postgres tables** (`chunks` = text + metadata,
`chunk_embeddings` = vectors), using the **pgvector** extension. You do **not** write any DDL yourself — `ensure_schema()` in `common/datastore.py`
creates the tables, indexes, and the `search_chunks()` hybrid retrieve function. All you need to provide is:

1. a running PostgreSQL with the `vector` extension available,
2. a database + login role,
3. a connection string in `PG_DSN`.

Everything below is one-time server setup. Pick **Docker** (fastest, local or
VPS), a **native local install**, or a **VPS**.

Requirements: PostgreSQL 14+ (16 recommended) and pgvector 0.5+.

---

## Hybrid retrieve

When `PG_DSN` is set, Phase 3 retrieve is one SQL function, `search_chunks()`,
created by `ensure_schema()` in `common/datastore.py` (not a hand-written SQL
file). It fuses HNSW cosine on `chunk_embeddings` with GIN `ts` on `chunks`
using reciprocal rank fusion (`is_parent = false` on both legs). Python still
embeds the query, reranks, and expands parents. File/npz mode keeps Python RRF.

---

## Option A — Docker (fastest; works local or on a VPS)

The official pgvector image already bundles the extension:

```bash
docker run -d --name arkguru-pg \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=change-me \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  -v arkguru_pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

Connection string:

```bash
export PG_DSN="postgresql://rag:change-me@localhost:5432/rag"
```

The `vector` extension is enabled automatically by the app on first run; to do it
by hand: `docker exec -it arkguru-pg psql -U rag -d rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'`

---

## Option B — Native local install

### macOS (Homebrew)
```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb rag
psql rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
export PG_DSN="postgresql://$(whoami)@localhost:5432/rag"
```

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y postgresql postgresql-16-pgvector   # pkg name matches your PG major version
sudo -u postgres psql <<'SQL'
CREATE ROLE rag LOGIN PASSWORD 'change-me';
CREATE DATABASE rag OWNER rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
SQL
export PG_DSN="postgresql://rag:change-me@localhost:5432/rag"
```
If `postgresql-16-pgvector` isn't in your apt sources, build it:
```bash
sudo apt install -y build-essential postgresql-server-dev-16 git
git clone --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

---

## Option C — VPS server (Ubuntu 22.04, remote access)

Use this when Phases 1/2 run in the cloud and write to a shared database, or when
your NUC (Phase 3) connects to a remote DB.

**1. Install Postgres + pgvector** (see Ubuntu steps above).

**2. Create the database and role**
```bash
sudo -u postgres psql <<'SQL'
CREATE ROLE rag LOGIN PASSWORD 'use-a-strong-password';
CREATE DATABASE rag OWNER rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
SQL
```

**3. Allow remote connections** (only if you cannot use an SSH tunnel — see below)
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

**5. Connection string (require TLS for remote)**
```bash
export PG_DSN="postgresql://rag:use-a-strong-password@<vps.host>:5432/rag?sslmode=require"
```

### Safer alternative: SSH tunnel (recommended)
Instead of exposing 5432, keep Postgres bound to localhost on the VPS and tunnel:
```bash
ssh -N -L 5432:localhost:5432 user@<vps.host>
# then, locally:
export PG_DSN="postgresql://rag:use-a-strong-password@localhost:5432/rag"
```
This needs no `listen_addresses`, no firewall hole, and no public 5432.

---

## Option D — Supabase (Session pooler)

Cloud Agents are **IPv4-only**. The **Direct** URI (`db.<ref>.supabase.co:5432`) is IPv6-only and will fail with `Network is unreachable`. Use **Session pooler** (Supavisor session mode):

- Host: `aws-<region>.pooler.supabase.com` (not `db.*.supabase.co`)
- Port: **5432** (not 6543 — transaction mode often breaks DDL)
- User: `postgres.<project-ref>` (not plain `postgres`)
- Query: `sslmode=require`

```bash
export PG_DSN="postgresql://postgres.YOUR_REF:YOUR_PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
```

Copy the string from the project dashboard: **Connect → Session pooler**. Store it as a Cursor **Runtime Secret** or **Environment Variable** named `PG_DSN` (not a Build Secret).

**Two tables:** Phase 1 `--sink postgres` fills **`chunks`** (`chunk_index` is 0-based on that table). It does **not** insert into **`chunk_embeddings`**. Fill vectors in Phase 3:

```bash
python -m phase3_rag.embed_datastore --embedder hashing --dim 1024
# or: make embed-db
# or: python -m phase3_rag.index --embed-only --embedder hashing
```

In the Table Editor, inspect `chunks.chunk_index` (0, 1, 2, …). Opening `chunk_embeddings` will not show `chunk_index`. A `0` in `chunk_index` can look blank in the editor — confirm with SQL:

```sql
SELECT chunk_id, chunk_index, source_id, page, section, token_count
FROM chunks ORDER BY source_id, chunk_index;
SELECT count(*) FROM chunk_embeddings;
```

---

## Point the pipeline at your database

All datastore settings live in **`config/datastore.yaml`**; the connection string
is read from the `PG_DSN` env var by default. Either export it, or copy
`.env.example` → `.env` and fill it in:

```bash
cp .env.example .env
# edit .env:  PG_DSN=postgresql://rag:...@host:5432/rag
```

To use different table names or a different DB, edit `config/datastore.yaml`
(`chunks_table`, `vectors_table`, `dim`). No code changes needed.

---

## Verify it works

```bash
# server + extension present
psql "$PG_DSN" -c "SELECT version();"
psql "$PG_DSN" -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"

# app can connect and create the two tables
python - <<'PY'
from common.datastore_config import open_chunk_store
s = open_chunk_store()          # reads config/datastore.yaml + PG_DSN
s.ensure_schema()               # creates tables + search_chunks() if missing
print("connected OK. chunks:", s.count(), "| vectors:", s.count_vectors())
PY
```

Expected: the two tables `chunks` and `chunk_embeddings` now exist.

You usually don't even need this: the tables are created automatically the first
time any phase writes to the database. To create them explicitly, Phase 1 offers:

```bash
python -m phase1_pdf.pipeline --init-db     # or: make init-db  /  python run_phase1.py --init-db
```

---

## Operating notes

- **Tables are auto-managed.** `ensure_schema()` is idempotent on every write path
  and installs `search_chunks()` (hybrid retrieve). Do not put that function in a
  hand-written schema file.
- **Re-embed with a new model:** `TRUNCATE chunk_embeddings;` (chunk rows in
  `chunks` are untouched), then re-run `phase3_rag.embed_datastore`. If the new
  model's dimension differs, update `dim` in `config/datastore.yaml` first and
  drop/recreate `chunk_embeddings` (the vector column dimension is fixed at
  creation).
- **Backups:** `pg_dump "$PG_DSN" > backup.sql`.
- **Sizing:** each embedding is `dim * 4` bytes (bge-m3 1024-d ≈ 4 KB/row) plus
  the HNSW index; chunk text is usually the larger cost. 100k chunks is well
  within a small VPS.
- **Security checklist (VPS):** strong password + `scram-sha-256`, restrict
  `pg_hba.conf` to specific IPs (or use an SSH tunnel), `sslmode=require` for
  remote clients, keep 5432 off the public internet.
