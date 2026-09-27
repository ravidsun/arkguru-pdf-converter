"""
Embedding abstraction with two backends:

  - "sentence_transformer" : REAL embeddings (bge-m3 / e5 / MiniLM) via
                             sentence-transformers. This is the default and
                             the only backend allowed to write the production
                             Postgres datastore unless ``--allow-hashing`` is
                             set. Downloads the model from Hugging Face on
                             first run.
  - "hashing"              : dependency-free (numpy only) hashing embedder
                             with a FIXED dimension. No downloads,
                             deterministic. For tests/dev only — never the
                             silent default, and never written to the
                             production datastore without an explicit allow.

Both return L2-normalized float32 vectors, so cosine == dot product everywhere.
``Embedder.label`` is the string that must be stored in
``chunk_embeddings.model`` (the Hugging Face id for sentence_transformer,
``hashing`` for the test backend). Callers must not invent a different label.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

import numpy as np

log = logging.getLogger("phase3.embedder")

DEFAULT_BACKEND = "sentence_transformer"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024
DEFAULT_MAX_SEQ_LENGTH = 512
# bge-m3 at seq 512 in fp16 fits a 6 GB RTX 3050 with headroom. Raise on CPU
# (32–64) or a larger GPU; lower to 8 if you still OOM.
DEFAULT_BATCH_SIZE = 16
HASHING_LABEL = "hashing"
HASHING_DATASTORE_ERROR = (
    "Refusing to write hashing embeddings to the Postgres datastore. "
    "The production store must use sentence_transformer (default model "
    f"{DEFAULT_MODEL}). Pass --embedder hashing --allow-hashing only for "
    "tests/dev."
)

_WORD = re.compile(r"[a-z0-9]+")


class HashingDatastoreForbidden(RuntimeError):
    """Raised when hashing would write to the production datastore."""


class EmbeddingModelMismatch(RuntimeError):
    """Stored ``chunk_embeddings.model`` does not match the query embedder."""


def resolve_device(explicit: str | None = None) -> str:
    """Return ``cuda`` when a GPU is available, otherwise ``cpu``.

    ``explicit`` of ``None``, ``""``, or ``"auto"`` triggers detection.
    Any other value is returned as-is (``cpu``, ``cuda``, ``cuda:0``, …).
    """
    if explicit and explicit not in ("auto",):
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def assert_datastore_backend_allowed(
    backend: str, *, allow_hashing: bool = False
) -> None:
    """Block hashing writes to Postgres unless the caller opted in."""
    if backend == "hashing" and not allow_hashing:
        raise HashingDatastoreForbidden(HASHING_DATASTORE_ERROR)


def is_invalid_vector(vec) -> bool:
    """True for empty, non-finite, or exact-zero vectors."""
    arr = np.asarray(vec, dtype="float32")
    if arr.size == 0:
        return True
    if not np.isfinite(arr).all():
        return True
    return float(np.linalg.norm(arr)) == 0.0


def filter_valid_vectors(
    ids: Sequence[str], vectors
) -> tuple[list[str], np.ndarray, list[str]]:
    """Drop zero/NaN/Inf rows. Returns (kept_ids, kept_vecs, skipped_ids)."""
    arr = np.asarray(vectors, dtype="float32")
    keep_ids: list[str] = []
    keep_rows: list[np.ndarray] = []
    skipped: list[str] = []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for cid, row in zip(ids, arr):
        if is_invalid_vector(row):
            skipped.append(cid)
            continue
        keep_ids.append(cid)
        keep_rows.append(row)
    if not keep_rows:
        dim = int(arr.shape[1]) if arr.ndim == 2 and arr.shape[1] else 0
        return keep_ids, np.zeros((0, dim), dtype="float32"), skipped
    return keep_ids, np.stack(keep_rows, axis=0), skipped


def write_embeddings(store, ids: Sequence[str], texts: Sequence[str],
                     emb: "Embedder") -> tuple[int, list[str]]:
    """Encode ``texts`` and upsert valid vectors under ``emb.label``.

    Invalid (zero / NaN / Inf) vectors are never stored. Returns
    ``(rows_written, skipped_chunk_ids)``. Skipped rows stay missing (or keep
    their previous vector on ``--reembed``) so a later run retries them.
    """
    if not ids:
        return 0, []
    vecs = emb.encode(list(texts))
    good_ids, good_vecs, skipped = filter_valid_vectors(ids, vecs)
    wrote = 0
    if good_ids:
        wrote = store.update_embeddings(good_ids, good_vecs, model=emb.label)
    if skipped:
        log.warning(
            "skipped %d invalid (zero/NaN/Inf) vector(s); not stored. "
            "example chunk_id=%s",
            len(skipped), skipped[0],
        )
    return wrote, skipped


def check_stored_embedding_model(
    store,
    expected: str,
    *,
    allow_mismatch: bool = False,
) -> list[str]:
    """Fail (or warn) when stored models are blank or not ``expected``.

    An empty vectors table is not a mismatch — there is nothing to compare.
    """
    raw = list(store.distinct_embedding_models())
    labels = [(m or "").strip() for m in raw]
    unique = sorted(set(labels))
    if not raw:
        log.info("no rows in chunk_embeddings; skipping model-match check")
        return []
    mismatches = [lab if lab else "<blank>" for lab in unique if lab != expected]
    if not mismatches:
        return unique
    msg = (
        f"Stored embedding model(s) {unique!r} do not match the query "
        f"embedder {expected!r}. Re-embed with the same model "
        f"(python -m phase3_rag.embed_datastore --reembed --model {expected}) "
        f"or pass --allow-model-mismatch / retrieval.allow_model_mismatch."
    )
    if allow_mismatch:
        log.warning("allow_model_mismatch: %s", msg)
        return unique
    raise EmbeddingModelMismatch(msg)


class Embedder:
    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        model_name: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        device: str | None = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if backend not in ("sentence_transformer", "hashing"):
            raise ValueError(f"unknown embedder backend: {backend!r}")
        self.backend = backend
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = resolve_device(device)
        self._model = None
        if backend == "sentence_transformer":
            self._load_sentence_transformer()

    def _load_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model.max_seq_length = self.max_seq_length
        if self.device.startswith("cuda"):
            self._model.half()
        self.dim = self._model.get_sentence_embedding_dimension()

    @property
    def label(self) -> str:
        """Value written to ``chunk_embeddings.model``. Never a guessed alias."""
        if self.backend == "hashing":
            return HASHING_LABEL
        return self.model_name

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        bs = self.batch_size if batch_size is None else batch_size
        if self.backend == "sentence_transformer":
            v = self._model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(v, dtype="float32")
        if self.backend == "hashing":
            return self._encode_hashing(texts)
        never: str = self.backend
        raise ValueError(f"unhandled embedder backend: {never!r}")

    def _encode_hashing(self, texts: list[str]) -> np.ndarray:
        """Hashed bag-of-words with sub-linear TF weighting, L2-normalized.

        Deterministic and fixed-dim: token -> bucket via md5, value += 1+log(tf).
        ``_WORD`` is ASCII-only, so non-ASCII-only text yields a zero vector
        which ``write_embeddings`` / ``filter_valid_vectors`` will refuse.
        """
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            counts: dict[int, float] = {}
            for w in _WORD.findall(t.lower()):
                b = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                counts[b] = counts.get(b, 0.0) + 1.0
            for b, c in counts.items():
                out[i, b] = 1.0 + np.log(c)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out
