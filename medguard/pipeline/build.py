"""Build MedGuard's OWN retrieval index from a public corpus (fresh build code).

MedGuard rebuilds its index from public data (e.g. PubMed derm abstracts) rather than
loading legacy baseline's prebuilt .faiss -- so the whole stack is independent. This embeds
the corpus and writes a FAISS IndexFlatIP + a parallel metadata jsonl.

    python -m medguard.pipeline.build --corpus /path/to/derm_abstracts.jsonl
    # defaults for --index / --meta / --model come from medguard.config

The corpus jsonl needs one record per line with at least a "text" field (any other
fields -- id, source, pmid, title -- are carried into the metadata for citations).
"""
from __future__ import annotations

import argparse

from .. import config
from .retriever import build_index


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the MedGuard FAISS text index")
    ap.add_argument("--corpus", default=config.CORPUS_PATH,
                    help="public jsonl corpus (one record/line, needs 'text')")
    ap.add_argument("--index", default=config.TEXT_INDEX_PATH)
    ap.add_argument("--meta", default=config.TEXT_INDEX_META)
    ap.add_argument("--model", default=config.EMBEDDING_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    print(f"[build] corpus : {args.corpus}")
    print(f"[build] model  : {args.model}")
    print(f"[build] index  -> {args.index}")
    n = build_index(args.corpus, args.index, args.meta, model=args.model,
                    batch_size=args.batch_size)
    print(f"[build] done: {n} vectors indexed; metadata -> {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
