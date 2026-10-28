"""Ask a question and retrieve the most similar chunks, with book, section and page.

Run 08_embed_chunks.py first, with the same --model you use here.
"""
from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

from lab_common import ROOT, load_jsonl, write_json

embedder = importlib.import_module("08_embed_chunks")
QUERY_PREFIX = "search_query: "


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector)) or 1.0


def load_index(model: str, only: str | None) -> list[dict]:
    index, seen = [], set()
    for path in sorted((ROOT / "data" / "extracted").glob("*/embeddings.jsonl")):
        folder = path.parent
        if only and not folder.name.startswith(Path(only).stem + "-"):
            continue
        vectors = {row["chunk_id"]: row for row in load_jsonl(path) if row["model"] == model}
        if not vectors:
            print(f"Skipping {folder.name}: no vectors for model {model}.")
            continue
        for chunk in load_jsonl(folder / "chunks.jsonl"):
            row = vectors.get(chunk["chunk_id"])
            if row is None or row["text_hash"] != embedder.text_hash(chunk["text"]):
                raise SystemExit(f"{folder.name}: embeddings are stale. Re-run 08_embed_chunks.py.")
            if chunk["chunk_id"] in seen:  # identical PDF copied twice
                continue
            seen.add(chunk["chunk_id"])
            index.append({**chunk, "vector": row["vector"], "_norm": norm(row["vector"])})
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--pdf", help="Only search this PDF (e.g. data/pdfs/pdf4.pdf)")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--model", default="all-minilm:22m")
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()
    index = load_index(args.model, args.pdf)
    if not index:
        raise SystemExit("No embeddings found. Run: python scripts\\08_embed_chunks.py --all")
    query = embedder.embed(embedder.model_inputs([args.query], args.model, QUERY_PREFIX), args.model, args.url)[0]
    if len(query) != len(index[0]["vector"]):
        raise SystemExit("Query and chunk vectors differ in size; use the same --model for 08 and 09.")
    query_norm = norm(query)
    scored = sorted(
        ((sum(a * b for a, b in zip(query, item["vector"])) / (query_norm * item["_norm"]), item) for item in index),
        key=lambda pair: -pair[0],
    )[:max(1, args.top)]
    print(f"Query: {args.query}\nSearched {len(index)} chunks with {args.model}\n")
    results = []
    for rank, (score, item) in enumerate(scored, 1):
        pages = (f"page {item['pdf_page_start']}" if item["pdf_page_start"] == item["pdf_page_end"]
                 else f"pages {item['pdf_page_start']}-{item['pdf_page_end']}")
        print(f"#{rank}  score {score:.4f}  {item['source_file']}  PDF {pages}")
        print(f"    {' > '.join(item['heading_path'])}")
        print(f"    {' '.join(item['text'].split())[:240]}...\n")
        results.append({"rank": rank, "score": round(score, 4), "chunk_id": item["chunk_id"],
                        "source_file": item["source_file"], "heading_path": item["heading_path"],
                        "pdf_page_start": item["pdf_page_start"], "pdf_page_end": item["pdf_page_end"],
                        "text": item["text"]})
    write_json(ROOT / "data" / "demo" / "last-search.json",
               {"query": args.query, "model": args.model, "results": results})
    print("Saved: data/demo/last-search.json")
    print("Scores rank chunks against each other; they are not a confidence or an accuracy measure.")


if __name__ == "__main__":
    main()
