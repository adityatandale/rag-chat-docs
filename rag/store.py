"""Vector database layer (ChromaDB): store chunk vectors + text + metadata, search by similarity."""
import uuid
from typing import List, Optional

import chromadb
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, collection: Optional[str] = None, path: Optional[str] = None):
        """path=None -> in-memory; path='.chroma' -> persisted on disk.
        A unique collection name per session keeps users isolated in a shared process."""
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.name = collection or f"docs-{uuid.uuid4().hex[:12]}"
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: List[Chunk], embeddings: np.ndarray, batch: int = 256):
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(ids=[c.id for c in part],
                            documents=[c.text for c in part],
                            embeddings=embeddings[i:i + batch].tolist(),
                            metadatas=[c.metadata for c in part])

    def search(self, query_vec: np.ndarray, n: int = 20):
        """Return rows (id, text, metadata, score, vector) sorted best-first. score = cosine similarity."""
        total = self.col.count()
        if total == 0:
            return []
        res = self.col.query(query_embeddings=[query_vec.tolist()], n_results=min(n, total),
                             include=["documents", "metadatas", "distances", "embeddings"])
        return [{"id": res["ids"][0][i], "text": res["documents"][0][i], "metadata": res["metadatas"][0][i],
                 "score": 1.0 - float(res["distances"][0][i]),
                 "vector": np.asarray(res["embeddings"][0][i], dtype="float32")}
                for i in range(len(res["ids"][0]))]

    def delete_source(self, source: str):
        self.col.delete(where={"source": source})

    def sources(self) -> dict:
        """{filename: number_of_chunks}"""
        counts: dict = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def count(self) -> int:
        return self.col.count()

    def reset(self):
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
