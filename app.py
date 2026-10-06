import glob
import os
import uuid

import streamlit as st

from rag import (SUPPORTED_EXTENSIONS, ChatError, LLMClient, LLMConfig, RAGPipeline, VectorStore,
                 get_embedder, load_path)

st.set_page_config(page_title="Chat With Your Documents", page_icon="📚", layout="wide")


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~130 MB)...")
def load_embedder():
    return get_embedder("auto")


@st.cache_resource
def load_llm(api_key, base_url, model):
    return LLMClient(LLMConfig(api_key=api_key, base_url=base_url, model=model))


def get_llm():
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    cfg = LLMConfig.from_env(secrets)
    return load_llm(cfg.api_key, cfg.base_url, cfg.model) if cfg else None


embedder = load_embedder()
llm = get_llm()

ss = st.session_state
ss.setdefault("collection", f"docs-{uuid.uuid4().hex[:12]}")   # isolates each browser session
ss.setdefault("messages", [])
ss.setdefault("settings_sig", None)

# ---------- sidebar: ingestion + retrieval settings ----------
with st.sidebar:
    st.header("📄 1. Documents")
    files = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=[e.strip(".") for e in SUPPORTED_EXTENSIONS],
                             accept_multiple_files=True)
    use_sample = st.checkbox("Also load the sample documents", value=False,
                             help="Fictional company handbook, product FAQ and IT policy for trying the app.")

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 500, 50)
    overlap = st.slider("Overlap (characters)", 0, 300, 80, 10)

    st.header("🔎 2. Retrieval")
    k = st.slider("Passages sent to the LLM (top-k)", 1, 10, 4)
    use_mmr = st.checkbox("Diversify results (MMR)", value=False,
                          help="Avoids near-duplicate passages by balancing relevance and variety.")
    threshold = st.slider("Minimum similarity", 0.0, 0.9, 0.0, 0.05,
                          help="Passages scoring below this are dropped. Raise it to reduce noise.")

    store = VectorStore(collection=ss["collection"])
    pipe = RAGPipeline(store, embedder, llm, chunk_size=chunk_size, overlap=overlap)

    if st.button("⚙️ Index documents", type="primary", use_container_width=True):
        sources = [(f.name, f.getvalue()) for f in (files or [])]
        if use_sample:
            for p in sorted(glob.glob("data/sample/*")):
                with open(p, "rb") as fh:
                    sources.append((os.path.basename(p), fh.read()))
        if not sources:
            st.warning("Upload a file or tick the sample documents first.")
        else:
            with st.spinner("Chunking, embedding and storing..."):
                for name, data in sources:
                    try:
                        n = pipe.ingest_file(name, data)
                        st.success(f"{name}: {n} chunks")
                    except ValueError as e:
                        st.error(str(e))

    indexed = store.sources()
    if indexed:
        st.caption("Indexed: " + ", ".join(f"{s} ({n})" for s, n in indexed.items()))
    if st.button("🗑️ Clear index and chat", use_container_width=True):
        store.reset()
        ss["messages"] = []
        st.rerun()

    st.divider()
    st.caption(f"Embeddings: `{embedder.name}`")
    st.caption("LLM: " + (f"`{llm.model}`" if llm else "none (retrieval-only mode)"))

# ---------- main ----------
st.title("📚 Chat With Your Documents")
st.caption("Document → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer")

if "FALLBACK" in embedder.name:
    st.warning("The real embedding model could not be loaded, so a basic keyword-based fallback is being used. "
               "Answers will be much less accurate.")
if llm is None:
    st.info("No LLM key configured: the app shows the retrieved passages instead of a written answer. "
            "Set `LLM_API_KEY` (see README) to enable generated answers.")
if store.count() == 0:
    st.info("👈 Upload documents (or tick the sample documents) and click **Index documents** to begin.")


def render_hits(hits):
    with st.expander(f"📎 Sources ({len(hits)} passages retrieved)"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h.source}**, page {h.page} · similarity `{h.score:.2f}`")
            st.text(h.text)


for m in ss["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            render_hits(m["hits"])

question = st.chat_input("Ask a question about your documents...", disabled=store.count() == 0)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in ss["messages"]]
    ss["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            try:
                ans = pipe.ask(question, history=history, k=k, threshold=threshold, use_mmr=use_mmr)
            except ChatError as e:
                st.error(e.user_message)
                ss["messages"].pop()
            else:
                ss["messages"].append({"role": "assistant", "content": ans.text, "hits": ans.hits})
                st.rerun()
