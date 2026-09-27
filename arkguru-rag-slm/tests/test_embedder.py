"""Embedder defaults, labels, zero-vector guard, and device selection."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from phase3_rag.embedder import (
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIM,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL,
    HASHING_LABEL,
    Embedder,
    EmbeddingModelMismatch,
    HashingDatastoreForbidden,
    assert_datastore_backend_allowed,
    check_stored_embedding_model,
    filter_valid_vectors,
    is_invalid_vector,
    resolve_device,
    write_embeddings,
)


def test_defaults_are_production_embedder():
    assert DEFAULT_BACKEND == "sentence_transformer"
    assert DEFAULT_MODEL == "BAAI/bge-m3"
    assert DEFAULT_DIM == 1024
    assert DEFAULT_MAX_SEQ_LENGTH == 512
    assert DEFAULT_BATCH_SIZE == 16


def test_hashing_label_is_never_the_hf_model_id():
    emb = Embedder(backend="hashing", model_name="BAAI/bge-m3", dim=32)
    assert emb.label == HASHING_LABEL
    assert emb.label != emb.model_name


def test_sentence_transformer_label_is_the_requested_model(monkeypatch):
    seen = {}

    class FakeST:
        def __init__(self, name, device="cpu"):
            seen["name"] = name
            seen["device"] = device
            self.max_seq_length = 0

        def half(self):
            seen["half"] = True

        def get_sentence_embedding_dimension(self):
            return 1024

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    emb = Embedder(
        backend="sentence_transformer",
        model_name="BAAI/bge-m3",
        device="cpu",
    )
    assert emb.label == "BAAI/bge-m3"
    assert seen["name"] == "BAAI/bge-m3"
    assert seen["device"] == "cpu"
    assert "half" not in seen
    assert emb._model.max_seq_length == 512


def test_resolve_device_auto_cuda(monkeypatch):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_device() == "cuda"
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_cpu(monkeypatch):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_device() == "cpu"


def test_resolve_device_explicit_override(monkeypatch):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_sentence_transformer_uses_cuda_fp16(monkeypatch):
    seen = {}

    class FakeST:
        def __init__(self, name, device="cpu"):
            seen["device"] = device
            self.max_seq_length = 0

        def half(self):
            seen["half"] = True

        def get_sentence_embedding_dimension(self):
            return 1024

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    emb = Embedder(backend="sentence_transformer", model_name="BAAI/bge-m3")
    assert emb.device == "cuda"
    assert seen["device"] == "cuda"
    assert seen["half"] is True


def test_hashing_non_ascii_is_zero_and_invalid():
    emb = Embedder(backend="hashing", dim=32)
    vecs = emb.encode(["你好世界", "॥ शान्तिः"])
    assert vecs.shape == (2, 32)
    assert np.allclose(vecs, 0)
    assert all(is_invalid_vector(v) for v in vecs)


def test_hashing_ascii_is_nonzero():
    emb = Embedder(backend="hashing", dim=32)
    vec = emb.encode(["sub sub lord in KP horary"])[0]
    assert not is_invalid_vector(vec)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(vec))


def test_filter_valid_vectors_skips_zero_and_nan():
    ids = ["a", "b", "c"]
    vecs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [np.nan, 1.0],
    ], dtype="float32")
    kept_ids, kept, skipped = filter_valid_vectors(ids, vecs)
    assert kept_ids == ["b"]
    assert skipped == ["a", "c"]
    assert kept.shape == (1, 2)


def test_write_embeddings_skips_invalid_and_labels_model():
    class Store:
        def __init__(self):
            self.calls = []

        def update_embeddings(self, ids, vectors, model):
            self.calls.append((list(ids), np.asarray(vectors), model))
            return len(ids)

    store = Store()
    emb = Embedder(backend="hashing", model_name="BAAI/bge-m3", dim=16)
    wrote, skipped = write_embeddings(
        store, ["ok", "bad"], ["hello world", "你好"], emb
    )
    assert wrote == 1
    assert skipped == ["bad"]
    assert store.calls[0][0] == ["ok"]
    assert store.calls[0][2] == "hashing"


def test_hashing_datastore_guard():
    assert_datastore_backend_allowed("sentence_transformer", allow_hashing=False)
    with pytest.raises(HashingDatastoreForbidden, match="allow-hashing"):
        assert_datastore_backend_allowed("hashing", allow_hashing=False)
    assert_datastore_backend_allowed("hashing", allow_hashing=True)


def test_mismatch_blank_and_wrong_model():
    class Store:
        def __init__(self, models):
            self._models = models

        def distinct_embedding_models(self):
            return self._models

    with pytest.raises(EmbeddingModelMismatch, match="do not match"):
        check_stored_embedding_model(Store([None]), "BAAI/bge-m3")
    with pytest.raises(EmbeddingModelMismatch, match="do not match"):
        check_stored_embedding_model(Store([""]), "BAAI/bge-m3")
    with pytest.raises(EmbeddingModelMismatch, match="hashing"):
        check_stored_embedding_model(Store(["hashing"]), "BAAI/bge-m3")


def test_mismatch_override_warns(caplog):
    class Store:
        def distinct_embedding_models(self):
            return ["hashing"]

    with caplog.at_level("WARNING"):
        labels = check_stored_embedding_model(
            Store(), "BAAI/bge-m3", allow_mismatch=True
        )
    assert labels == ["hashing"]
    assert "allow_model_mismatch" in caplog.text


def test_matching_model_passes():
    class Store:
        def distinct_embedding_models(self):
            return ["BAAI/bge-m3"]

    assert check_stored_embedding_model(Store(), "BAAI/bge-m3") == ["BAAI/bge-m3"]


def test_empty_store_is_not_a_mismatch():
    class Store:
        def distinct_embedding_models(self):
            return []

    assert check_stored_embedding_model(Store(), "BAAI/bge-m3") == []
