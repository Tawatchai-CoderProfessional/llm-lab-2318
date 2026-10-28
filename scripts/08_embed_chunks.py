"""Embed every chunk of one PDF (or all PDFs) with a local Ollama embedding model.

Run 06_chunk_sections.py first. Vectors are cached by chunk text and model, so a
re-run only embeds chunks that are new or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lab_common import ROOT, doc_dir, load_jsonl, write_json, write_jsonl

DOCUMENT_PREFIX = "search_document: "


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def model_inputs(texts: list[str], model: str, prefix: str) -> list[str]:
    # nomic-embed-text was trained with task prefixes; other models get plain text.
    return [prefix + text for text in texts] if model.startswith("nomic-embed-text") else texts


def embed(texts: list[str], model: str, url: str, timeout: int = 300) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts, "truncate": True}).encode("utf-8")
    request = Request(url.rstrip("/") + "/api/embed", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Ollama HTTP {exc.code}: {detail}\nCheck `ollama list` and the selected model.") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot call Ollama at {url}: {exc}\nOpen the Ollama app, then run 01_check_setup.py.") from exc
    vectors = result.get("embeddings", [])
    if len(vectors) != len(texts) or any(not vector for vector in vectors):
        raise SystemExit(f"Unexpected Ollama response: {list(result)}")
    return vectors


def load_cache(path: Path, model: str) -> dict[tuple[str, str], list[float]]:
    if not path.is_file():
        return {}
    return {(row["chunk_id"], row["text_hash"]): row["vector"]
            for row in load_jsonl(path) if row.get("model") == model}


def embed_document(destination: Path, model: str, url: str, batch_size: int) -> dict:
    chunks = load_jsonl(destination / "chunks.jsonl")
    if not chunks:
        raise SystemExit(f"No chunks in {destination.name}. Run 06_chunk_sections.py first.")
    output = destination / "embeddings.jsonl"
    cache = load_cache(output, model)
    vectors: dict[str, list[float]] = {}
    todo = []
    for chunk in chunks:
        key = (chunk["chunk_id"], text_hash(chunk["text"]))
        if key in cache:
            vectors[chunk["chunk_id"]] = cache[key]
        else:
            todo.append(chunk)
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        result = embed(model_inputs([c["text"] for c in batch], model, DOCUMENT_PREFIX), model, url)
        for chunk, vector in zip(batch, result):
            vectors[chunk["chunk_id"]] = vector
        done = min(start + batch_size, len(todo))
        print(f"  {destination.name}: embedded {done}/{len(todo)} new chunks", end="\r", flush=True)
    if todo:
        print()
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise SystemExit(f"Inconsistent vector dimensions: {sorted(dimensions)}")
    rows = [{
        "chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
        "model": model, "text_hash": text_hash(chunk["text"]),
        "dimensions": len(vectors[chunk["chunk_id"]]), "vector": vectors[chunk["chunk_id"]],
    } for chunk in chunks]
    write_jsonl(output, rows)
    report = {
        "source_file": chunks[0]["source_file"], "document_id": chunks[0]["document_id"],
        "model": model, "dimensions": dimensions.pop(), "chunks": len(chunks),
        "newly_embedded": len(todo), "reused_from_cache": len(chunks) - len(todo),
        "seconds": round(time.perf_counter() - started, 1),
        "note": "truncate=true: text beyond the model's context window is cut by Ollama, not by this script.",
    }
    write_json(destination / "embeddings-report.json", report)
    return report


def all_documents() -> list[Path]:
    """Every extracted folder with chunks, keeping one folder per identical PDF."""
    seen, folders = set(), []
    for folder in sorted((ROOT / "data" / "extracted").glob("*/chunks.jsonl")):
        document_id = folder.parent.name.rsplit("-", 1)[-1]
        if document_id in seen:
            print(f"Skipping duplicate PDF copy: {folder.parent.name}")
            continue
        seen.add(document_id)
        folders.append(folder.parent)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pdf", help="PDF path, relative to project root or absolute")
    target.add_argument("--all", action="store_true", help="Embed every PDF that has chunks.jsonl")
    parser.add_argument("--model", default="all-minilm:22m")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    folders = all_documents() if args.all else [doc_dir(args.pdf)]
    if not folders:
        raise SystemExit("No chunks.jsonl found. Run steps 03-06 first.")
    for folder in folders:
        report = embed_document(folder, args.model, args.url, args.batch_size)
        print(f"{report['source_file']}: {report['chunks']} chunks, {report['dimensions']} dimensions, "
              f"new={report['newly_embedded']}, cached={report['reused_from_cache']}, {report['seconds']} s")
    print("Saved embeddings.jsonl next to each chunks.jsonl. Next: 09_search.py --query \"...\"")


if __name__ == "__main__":
    main()
