"""Build MedGuard's OWN visual-exemplar atlas + linear probes from public TRAIN images.

Fresh build code (not legacy baseline's scripts): encodes HAM10000 train images with the frozen
exemplar encoder (DermLIP by default; swap via MEDGUARD_EXEMPLAR_ENCODER), builds a FAISS
kNN atlas, and fits the 7-way diagnosis + binary malignant probes. Train-split only.

    python -m medguard.pipeline.build_atlas --train /path/to/vqa_train.jsonl
    # jsonl records need an image path + a diagnosis label (keys autodetected;
    #   override with --image-key / --label-key)
"""
from __future__ import annotations

import argparse

from .. import config
from .image_retriever import build_atlas, fit_probes, make_encoder


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MedGuard exemplar atlas + probes")
    ap.add_argument("--train", required=True, help="train jsonl (image path + label per line)")
    ap.add_argument("--index", default=config.EXEMPLAR_INDEX_PATH)
    ap.add_argument("--meta", default=config.EXEMPLAR_META_PATH)
    ap.add_argument("--malignant", default=config.EXEMPLAR_MALIGNANT_CLF)
    ap.add_argument("--diag", default=config.EXEMPLAR_DIAG_CLF)
    ap.add_argument("--image-key", default="image_path")
    ap.add_argument("--label-key", default="label")
    ap.add_argument("--no-probes", action="store_true", help="atlas only, skip probe fitting")
    args = ap.parse_args()

    enc = make_encoder()
    print(f"[atlas] encoder: {config.EXEMPLAR_ENCODER_MODEL}")
    n = build_atlas(args.train, args.index, args.meta, enc, args.image_key, args.label_key)
    print(f"[atlas] {n} exemplars indexed -> {args.index}")
    if not args.no_probes:
        r = fit_probes(args.train, args.malignant, args.diag, enc, args.image_key, args.label_key)
        print(f"[probes] fit on {r['n']} imgs: {r['diag_classes']} diag classes, "
              f"{r['malignant_pos']} malignant -> {args.diag}, {args.malignant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
