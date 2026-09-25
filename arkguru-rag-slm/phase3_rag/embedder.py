"""
Embedding abstraction with two backends:

  - "sentence_transformer" : REAL embeddings (bge-m3 / e5 / MiniLM) via
                             sentence-transformers. Use on your NUC. Downloads
                             the model from Hugging Face on first run.
  - "hashing"              : dependency-free (numpy only) hashing embedder with
                             a FIXED dimension. No downloads, deterministic,
                             perfectly incremental. Meant for OFFLINE TESTING of
                             the pipeline plumbing -- not for production quality.

Both return L2-normalized float32 vectors, so cosine == dot product everywhere.
The rest of the system never needs to know which backend produced a vector; only
the dimension must stay consistent within one store.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self, backend: str = "sentence_transformer",
                 model_name: str = "BAAI/bge-m3", dim: int = 1024):
        self.backend = backend
        self.model_name = model_name
        self.dim = dim
        self._model = None
        if backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device="cpu")
            self.dim = self._model.get_sentence_embedding_dimension()

    # -- public ------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.backend == "sentence_transformer":
            v = self._model.encode(texts, batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype="float32")
        return self._encode_hashing(texts)

    # -- offline hashing backend ------------------------------------------
    def _encode_hashing(self, texts: list[str]) -> np.ndarray:
        """Hashed bag-of-words with sub-linear TF weighting, L2-normalized.

        Deterministic and fixed-dim: token -> bucket via md5, value += 1+log(tf).
        Gives a real lexical vector good enough to demonstrate ranking.
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
