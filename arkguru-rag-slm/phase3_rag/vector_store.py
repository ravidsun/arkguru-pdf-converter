"""
Local, incremental vector store -- no Postgres required.

Persists to two files next to `path`:
  <path>.npz   : float32 matrix of L2-normalized embeddings
  <path>.jsonl : one metadata record per row (chunk_id, text, section, ...)

Key property for your iterative workflow: `add()` is **incremental and
idempotent**. It skips any chunk whose `chunk_id` is already stored, so you can
process a couple of PDFs today, more next week, and just call `add()` again --
only the new chunks get embedded and appended. Nothing is recomputed.

For production/scale, swap this for pgvector (see index.py) behind the same
add()/search() interface.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class LocalVectorStore:
    def __init__(self, path: str = "data/store/index"):
        self.path = Path(path)
        self.vecs: np.ndarray | None = None
        self.meta: list[dict] = []
        self._ids: set[str] = set()
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self):
        npz, jsonl = self.path.with_suffix(".npz"), self.path.with_suffix(".jsonl")
        if npz.exists() and jsonl.exists():
            self.vecs = np.load(npz)["vecs"].astype("float32")
            self.meta = [json.loads(l) for l in jsonl.open() if l.strip()]
            self._ids = {m["chunk_id"] for m in self.meta}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path.with_suffix(".npz"),
                            vecs=self.vecs if self.vecs is not None
                            else np.zeros((0, 0), "float32"))
        with self.path.with_suffix(".jsonl").open("w") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # -- mutation ----------------------------------------------------------
    def add(self, chunks: list[dict], vectors: np.ndarray) -> int:
        """Append only chunks whose chunk_id is new. Returns #added."""
        keep_rows, keep_meta = [], []
        for i, c in enumerate(chunks):
            cid = c["chunk_id"]
            if cid in self._ids:
                continue
            self._ids.add(cid)
            keep_rows.append(vectors[i])
            keep_meta.append(c)
        if not keep_rows:
            return 0
        new = np.asarray(keep_rows, dtype="float32")
        self.vecs = new if self.vecs is None or self.vecs.size == 0 \
            else np.vstack([self.vecs, new])
        self.meta.extend(keep_meta)
        return len(keep_rows)

    # -- query -------------------------------------------------------------
    def search(self, qvec: np.ndarray, k: int = 20) -> list[tuple[dict, float]]:
        if self.vecs is None or len(self.meta) == 0:
            return []
        sims = self.vecs @ np.asarray(qvec, dtype="float32")  # cosine (normalized)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.meta[i], float(sims[i])) for i in idx]

    def __len__(self):
        return len(self.meta)
