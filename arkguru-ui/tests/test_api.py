from fastapi.testclient import TestClient

from backend.app import app
from backend.paths import discover

client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "phase1" in body
    assert "phase2" in body
    assert "phase3" in body
    assert body["phase2"]["found"] is True
    assert "pg_dsn_set" in body
    # never leak the DSN itself
    assert "PG_DSN" not in r.text
    assert "postgresql://" not in r.text


def test_defaults_and_corpus():
    defaults = client.get("/api/defaults")
    assert defaults.status_code == 200
    body = defaults.json()
    assert "phase1_input" in body
    assert "phase2_seeds" in body
    corpus = client.get("/api/corpus")
    assert corpus.status_code == 200
    assert "phase1" in corpus.json()


def test_phase2_requires_seeds():
    r = client.post("/api/jobs/phase2", json={"seeds": []})
    assert r.status_code == 400


def test_chat_requires_question():
    r = client.post("/api/chat", json={"question": "   "})
    assert r.status_code == 400


def test_phase1_missing_input_dir():
    r = client.post(
        "/api/jobs/phase1",
        json={"input_dir": "/this/path/does/not/exist-arkguru-ui"},
    )
    assert r.status_code == 400


def test_index_html_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Run Phase 1" in r.text
    assert "Run Phase 2" in r.text


def test_unified_phase2_requires_seeds():
    r = client.post("/api/jobs", json={"phase": "phase2", "seeds": []})
    assert r.status_code == 400


def test_current_job_empty():
    r = client.get("/api/jobs/current")
    assert r.status_code == 200
    assert "job" in r.json()


def test_unknown_job_404():
    r = client.get("/api/jobs/not-a-real-job")
    assert r.status_code == 404


def test_layout_matches_health():
    layout = discover()
    body = client.get("/api/health").json()
    assert body["phase2"]["found"] is (layout.phase2 is not None)
