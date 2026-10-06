"""Retrieval: embed the question, search the vector DB, then improve the results.

Retrieval-quality levers implemented here:
  * top-k          how many passages go to the LLM
  * score threshold drop passages that are not similar enough (avoids feeding noise)
  * MMR            Maximal Marginal Relevance: balance relevance with diversity so the
                   top-k are not near-duplicates of each other
"""
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int
    score: float


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float = 0.6) -> List[int]:
    """Return indices of k candidates chosen by MMR. Vectors must be L2-normalised.
    lambda_mult=1.0 is pure relevance; lower values favour diversity."""
    if len(cand_vecs) == 0:
        return []
    rel = cand_vecs @ query_vec
    chosen = [int(np.argmax(rel))]
    remaining = [i for i in range(len(cand_vecs)) if i != chosen[0]]
    while remaining and len(chosen) < k:
        sims_to_chosen = cand_vecs[remaining] @ cand_vecs[chosen].T          # (remaining, chosen)
        scores = lambda_mult * rel[remaining] - (1 - lambda_mult) * sims_to_chosen.max(axis=1)
        best = remaining[int(np.argmax(scores))]
        chosen.append(best)
        remaining.remove(best)
    return chosen


class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 4, threshold: float = 0.0, use_mmr: bool = False,
                 fetch_k: int = 20, lambda_mult: float = 0.6) -> List[Hit]:
        qv = self.embedder.embed_query(query)
        rows = self.store.search(qv, n=max(fetch_k, k) if use_mmr else k)
        rows = [r for r in rows if r["score"] >= threshold]
        if use_mmr and rows:
            idx = mmr(qv, np.stack([r["vector"] for r in rows]), k, lambda_mult)
            rows = [rows[i] for i in idx]
        return [Hit(r["id"], r["text"], r["metadata"].get("source", "?"),
                    int(r["metadata"].get("page", 0)), r["score"]) for r in rows[:k]]
