"""Embeddings: turn text into vectors so similar meaning = nearby vectors.

Default: fastembed (ONNX, no PyTorch, no API key) running BAAI/bge-small-en-v1.5 locally.
HashingEmbedder is a tiny deterministic fallback (word overlap only, NOT semantic); it exists
so tests and demos run offline. Every embedder returns L2-normalised float32 vectors, so
cosine similarity is a plain dot product.
"""
import hashlib
import re
from typing import List

import numpy as np

STOPWORDS = set("a an and are as at be by do does for from how i if in is it my of on or the "
                "to was what when where which who will with you your can we our their".split())


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype("float32")


class HashingEmbedder:
    """Bag-of-words feature hashing. Deterministic and dependency-free; lexical, not semantic."""
    name = "hashing-bow (offline fallback, lexical only)"

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in STOPWORDS:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), "float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding   # lazy import; downloads the model on first use
        self.model_name = model_name
        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), "float32")
        fn = getattr(self._model, "passage_embed", self._model.embed)
        return _normalize(np.array(list(fn(texts)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        fn = getattr(self._model, "query_embed", self._model.embed)
        return _normalize(np.array(list(fn([text])), dtype="float32"))[0]


def get_embedder(prefer: str = "auto"):
    """'auto' tries the real model and falls back to the hashing embedder with a clear name."""
    if prefer == "hashing":
        return HashingEmbedder()
    try:
        return FastEmbedEmbedder()
    except Exception as exc:  # noqa: BLE001 - missing package, no network, etc.
        if prefer == "fastembed":
            raise
        emb = HashingEmbedder()
        emb.name = f"hashing-bow FALLBACK ({type(exc).__name__}: real embedder unavailable)"
        return emb
