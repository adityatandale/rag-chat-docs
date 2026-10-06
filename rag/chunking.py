"""Text chunking: recursive splitting on natural boundaries, with overlap.

Why chunk? Embedding models and LLM prompts work best on focused passages. Too big and the
embedding blurs several topics; too small and a chunk loses the context needed to answer.
Overlap keeps facts that straddle a boundary retrievable.
"""
import hashlib
from dataclasses import dataclass, field
from typing import List

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_recursive(text: str, size: int, seps: List[str]) -> List[str]:
    """Split text into pieces no longer than `size`, preferring the earliest separator."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            parts = text.split(sep)
            pieces = []
            for j, part in enumerate(parts):
                piece = part + (sep if j < len(parts) - 1 else "")
                if len(piece) > size:
                    pieces.extend(_split_recursive(piece, size, seps[i + 1:]))
                elif piece:
                    pieces.append(piece)
            return pieces
    return [text[k:k + size] for k in range(0, len(text), size)]   # hard cut as a last resort


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Greedy-merge small pieces into chunks of <= chunk_size characters, carrying `overlap`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    pieces = _split_recursive(text.strip(), chunk_size, SEPARATORS)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # start the next chunk at a word boundary inside the overlap window
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = tail + piece
            if len(current) > chunk_size:   # overlap + piece too long: drop the overlap
                current = piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_documents(pages, chunk_size: int = 500, overlap: int = 80) -> List[Chunk]:
    """pages: list of (text, metadata). Returns Chunks with stable ids and metadata."""
    out = []
    for text, meta in pages:
        for i, piece in enumerate(split_text(text, chunk_size, overlap)):
            digest = hashlib.md5(piece.encode("utf-8")).hexdigest()[:8]
            cid = f"{meta.get('source', 'doc')}-p{meta.get('page', 0)}-c{i}-{digest}"
            out.append(Chunk(cid, piece, {**meta, "chunk": i}))
    return out
