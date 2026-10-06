"""Measure retrieval quality on labelled questions, comparing chunk sizes and MMR.

A retrieval is a HIT if any of the top-k chunks contains the phrase that answers the question.
Metrics: hit@1, hit@3, hit@k and MRR (mean reciprocal rank of the first answer-bearing chunk).

    python -m evaluation.eval_retrieval                 # real embeddings (needs internet once)
    python -m evaluation.eval_retrieval --embedder hashing   # offline lexical fallback
"""
import argparse
import datetime
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RAGPipeline, VectorStore, get_embedder, load_path  # noqa: E402

# (question, phrase that must appear in a retrieved chunk)
CASES = [
    ("How many days of paid annual leave do I get?", "24 days of paid annual leave"),
    ("Can I carry over unused vacation days?", "Up to 5 unused days"),
    ("When do I need a doctor's note for sick leave?", "3 consecutive days"),
    ("How many days a week can I work from home?", "up to 3 days per week"),
    ("What is the daily meal allowance when travelling?", "40 USD per day"),
    ("How long is parental leave for primary caregivers?", "16 weeks"),
    ("How long is the warranty on the hiking boots?", "2 years warranty"),
    ("What is the return window?", "within 45 days"),
    ("How waterproof is the Summit Pro jacket?", "20,000 mm"),
    ("How much does express shipping cost?", "costs 15 USD"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("What should I do if I lose my laptop?", "within 1 hour"),
    ("Can I keep restricted data on my phone?", "never be stored on personal devices"),
    ("Do I need to use a VPN at the airport?", "public Wi-Fi"),
    ("Is the learning budget carried over?", "does not roll over"),
]


def evaluate(embedder, data_dir, chunk_size, overlap, use_mmr, k=4):
    store = VectorStore()
    pipe = RAGPipeline(store, embedder, llm=None, chunk_size=chunk_size, overlap=overlap)
    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        pipe.ingest_pages(load_path(path))
    hit1 = hit3 = hitk = 0
    rr = 0.0
    misses = []
    for q, phrase in CASES:
        hits = pipe.retriever.retrieve(q, k=k, use_mmr=use_mmr)
        rank = next((i for i, h in enumerate(hits, 1) if phrase.lower() in h.text.lower()), None)
        if rank:
            rr += 1 / rank
            hit1 += rank == 1
            hit3 += rank <= 3
            hitk += 1
        else:
            misses.append(q)
    n = len(CASES)
    return {"chunks": store.count(), "hit@1": hit1 / n, "hit@3": hit3 / n, f"hit@{k}": hitk / n,
            "mrr": rr / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--embedder", default="auto", choices=["auto", "fastembed", "hashing"])
    ap.add_argument("--out", default="evaluation/report.md")
    args = ap.parse_args()

    embedder = get_embedder(args.embedder)
    print("Embedder:", embedder.name)
    lines = ["# Retrieval Evaluation Report",
             f"\nGenerated {datetime.datetime.now():%Y-%m-%d %H:%M} | Embedder: `{embedder.name}` | "
             f"{len(CASES)} questions | k=4\n",
             "| Chunk size | Overlap | MMR | Chunks | Hit@1 | Hit@3 | Hit@4 | MRR |",
             "|---|---|---|---|---|---|---|---|"]
    all_misses = {}
    for size, overlap in [(200, 40), (500, 80), (1000, 150)]:
        for use_mmr in (False, True):
            r = evaluate(embedder, args.data, size, overlap, use_mmr)
            lines.append(f"| {size} | {overlap} | {'on' if use_mmr else 'off'} | {r['chunks']} | "
                         f"{r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@4']:.0%} | {r['mrr']:.2f} |")
            all_misses[(size, use_mmr)] = r["misses"]
            print(lines[-1])
    lines += ["\n## Questions missed at k=4\n"]
    for (size, use_mmr), misses in all_misses.items():
        if misses:
            lines.append(f"- chunk {size}, MMR {'on' if use_mmr else 'off'}: " + "; ".join(misses))
    lines += ["\n## Analysis (fill in)\n",
              "- Which chunk size gave the best hit@1 and MRR, and why do you think so?",
              "- Did MMR help or hurt on this small corpus?",
              "- Look at the missed questions: is it a chunking problem (answer split across chunks) "
              "or an embedding problem (wrong passage ranked higher)?",
              "- What would you try next (different embedding model, hybrid keyword search, re-ranking)?"]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Report written to", args.out)


if __name__ == "__main__":
    main()
