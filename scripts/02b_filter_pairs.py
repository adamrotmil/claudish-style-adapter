#!/usr/bin/env python3
"""Phase 3b: filter generated pairs for meaning preservation.

The teacher translator occasionally drifts (dropped clauses, flipped negations).
This pass embeds each pair with a small sentence encoder and keeps only pairs
where the Claudish text — and, when present, the roundtrip English — stay
semantically close to the original English. Rejected pairs go to a side file
for inspection.

Example:
    python scripts/02b_filter_pairs.py --pairs data/claudish_pairs.jsonl \
        --out data/claudish_pairs.filtered.jsonl
"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/claudish_pairs.jsonl")
    parser.add_argument("--out", default="data/claudish_pairs.filtered.jsonl")
    parser.add_argument("--rejected-out", default="data/claudish_pairs.rejected.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--min-claudish-sim", type=float, default=0.70,
                        help="min cosine sim between english and claudish")
    parser.add_argument("--min-roundtrip-sim", type=float, default=0.70,
                        help="min cosine sim between english and roundtrip english")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    import numpy as np

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"{len(pairs)} pairs loaded")

    model = SentenceTransformer(args.model)

    def embed(texts):
        return model.encode(texts, batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=True)

    eng = embed([p["english"] for p in pairs])
    cla = embed([p["claudish"] for p in pairs])
    sim_claudish = (eng * cla).sum(axis=1)

    has_rt = [bool(p.get("roundtrip_english")) for p in pairs]
    sim_roundtrip = np.ones(len(pairs))
    rt_idx = [i for i, h in enumerate(has_rt) if h]
    if rt_idx:
        rt = embed([pairs[i]["roundtrip_english"] for i in rt_idx])
        sim_roundtrip[rt_idx] = (eng[rt_idx] * rt).sum(axis=1)

    kept = rejected = 0
    with open(args.out, "w", encoding="utf-8") as out, \
         open(args.rejected_out, "w", encoding="utf-8") as rej:
        for i, pair in enumerate(pairs):
            pair["sim_claudish"] = round(float(sim_claudish[i]), 4)
            if has_rt[i]:
                pair["sim_roundtrip"] = round(float(sim_roundtrip[i]), 4)
            ok = (sim_claudish[i] >= args.min_claudish_sim
                  and sim_roundtrip[i] >= args.min_roundtrip_sim)
            target = out if ok else rej
            target.write(json.dumps(pair, ensure_ascii=False) + "\n")
            kept += ok
            rejected += not ok

    print(f"kept {kept} pairs -> {args.out}")
    print(f"rejected {rejected} pairs -> {args.rejected_out}")
    print(f"claudish sim: mean {sim_claudish.mean():.3f}, p10 "
          f"{np.percentile(sim_claudish, 10):.3f}")
    if rt_idx:
        rts = sim_roundtrip[rt_idx]
        print(f"roundtrip sim: mean {rts.mean():.3f}, p10 {np.percentile(rts, 10):.3f}")


if __name__ == "__main__":
    main()
