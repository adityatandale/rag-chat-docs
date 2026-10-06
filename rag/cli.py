"""Ask questions about a folder of documents from the terminal.

    python -m rag.cli data/sample "How many days of annual leave do I get?"
Works without an API key (prints the retrieved passages instead of a generated answer).
"""
import glob
import os
import sys

from .embeddings import get_embedder
from .llm import ChatError, LLMClient, LLMConfig
from .loaders import SUPPORTED_EXTENSIONS, load_path
from .pipeline import RAGPipeline
from .store import VectorStore


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m rag.cli <docs_folder> <question>")
    folder, question = sys.argv[1], " ".join(sys.argv[2:])
    embedder = get_embedder()
    cfg = LLMConfig.from_env()
    pipe = RAGPipeline(VectorStore(), embedder, LLMClient(cfg) if cfg else None)
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"indexed {os.path.basename(path)}: {pipe.ingest_pages(load_path(path))} chunks")
    try:
        ans = pipe.ask(question, k=4)
    except ChatError as e:
        raise SystemExit(e.user_message)
    print(f"\n[{ans.mode}] {ans.text}\n")
    for i, h in enumerate(ans.hits, 1):
        print(f"[{i}] {h.source} p{h.page} (score {h.score:.2f})\n    {h.text[:200]!r}")


if __name__ == "__main__":
    main()
