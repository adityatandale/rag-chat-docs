# 📚 Chat With Your Documents (RAG)

Upload PDFs, Word files, text or Markdown, then ask questions in plain language. Answers are **grounded in your documents** and come with **numbered citations** and the exact passages used.

```
Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer (+ sources)
 pypdf /    recursive   bge-small    ChromaDB    top-k +      grounded     citations
 docx/md    + overlap   (local)      (cosine)    MMR + min    prompt
                                                 similarity
```

**Live demo:** _optional: paste your Streamlit URL here_

## Why RAG?
LLMs hallucinate: they produce fluent text from patterns, not from your private or recent data. RAG retrieves the relevant passages first and instructs the model to answer **only** from them, and to say "I couldn't find that" otherwise.

## Features
| Stage | Implementation |
|---|---|
| Loading | PDF (per page), DOCX, TXT, MD; clear errors for empty/scanned/unsupported files |
| Chunking | Recursive splitter (paragraph → line → sentence → word) with configurable size and overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, local, no API key, no PyTorch) |
| Vector DB | ChromaDB, cosine similarity, per-session collections, delete-and-replace on re-upload |
| Retrieval quality | top-k, minimum-similarity threshold, MMR for diverse results |
| Generation | Grounded prompt: use only excerpts, cite `[1]`, refuse when unsupported, ignore instructions hidden in documents |
| Robustness | Works **without any API key** (retrieval-only mode); LLM calls retry rate limits/timeouts and show friendly errors |
| Evaluation | `evaluation/eval_retrieval.py` measures Hit@k and MRR across chunk sizes and MMR |

## Run locally
```bash
git clone https://github.com/adityatandale/rag-chat-docs.git
cd rag-chat-docs
pip install -r requirements.txt
streamlit run app.py
```
1. Tick **Also load the sample documents** (or upload your own) and click **Index documents**.
2. Ask: *"How many days of paid annual leave do I get?"*, *"Can I keep restricted data on my phone?"*
3. The first run downloads the embedding model (~130 MB).

**Generated answers (optional):** copy `.env.example` to `.env` and set `LLM_API_KEY` (free key from console.groq.com). Without a key the app shows the retrieved passages instead.

**Terminal version:** `python -m rag.cli data/sample "What is the return window?"`

## Evaluate retrieval quality
```bash
python -m evaluation.eval_retrieval          # writes evaluation/report.md
```
15 labelled questions over the sample documents; a retrieval counts as a hit if a top-k chunk contains the answer-bearing phrase. The report compares chunk sizes (200 / 500 / 1000) with MMR on and off. Add your own questions to `CASES` for your own documents.

## Project layout
| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, settings, chat, source viewer |
| `rag/loaders.py` | PDF/DOCX/TXT/MD → pages with metadata |
| `rag/chunking.py` | Recursive chunking with overlap |
| `rag/embeddings.py` | fastembed embedder + offline hashing fallback |
| `rag/store.py` | ChromaDB wrapper |
| `rag/retriever.py` | Search, threshold filter, MMR |
| `rag/pipeline.py` | Ingestion + question answering, grounded prompt |
| `rag/llm.py` | OpenAI-compatible client with retries |
| `tests/` | 19 offline unit tests + UI smoke test |

```bash
python -m pytest -q            # no network or key needed
python tests/ui_smoke.py
```

## Deploy free (Streamlit Community Cloud)
1. Push to GitHub, then at share.streamlit.io choose repo `adityatandale/rag-chat-docs`, branch `main`, file `app.py`.
2. Optional: under **Advanced settings > Secrets** add `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` (see `.streamlit/secrets.toml.example`).
3. The first load downloads the embedding model, so allow a minute. Uploaded documents live only in the session's memory.

## Limitations
- Follow-up questions are retrieved using the latest message only ("what about sick leave?" works; "and that one?" may not). Next step: rewrite follow-ups into standalone questions before retrieval.
- Scanned PDFs need OCR; tables and images in PDFs are not understood.
- Small chunk sizes can split an answer across chunks; large ones blur the embedding. Use the evaluation script to tune for your data.
- The similarity threshold depends on the embedding model, so tune it rather than trusting a default.
- Sessions are in-memory: indexed documents disappear when the app restarts (use `VectorStore(path=".chroma")` to persist locally).
- Ideas: hybrid keyword + vector search, cross-encoder re-ranking, streaming answers, per-claim citation checking.

## Security notes
API keys come only from environment variables or Streamlit secrets (`.env` is git-ignored). Retrieved text is treated as untrusted data in the prompt to reduce prompt-injection risk, but no defence is perfect.
