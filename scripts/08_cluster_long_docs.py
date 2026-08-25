#!/usr/bin/env python3
"""Phase 3e (v3): build coherent long English documents from related short texts.

v2's synthetic long pairs concatenated *unrelated* texts, which taught the model that
long documents change topic mid-stream — it then invented topic changes at inference.
This builds long docs the right way: embed the short texts, cluster them, and only
concatenate texts from the same cluster. Output is English-only docs meant to be
authored into Claudish by scripts/07_author_pairs.py.

Example:
    python scripts/08_cluster_long_docs.py --pairs data/claudish_pairs.all_short.jsonl \
        --out data/english_long_docs.jsonl --n 2500
"""
import argparse
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/claudish_pairs.all_short.jsonl")
    parser.add_argument("--out", default="data/english_long_docs.jsonl")
    parser.add_argument("--n", type=int, default=2500, help="long docs to build")
    parser.add_argument("--min-part-chars", type=int, default=150)
    parser.add_argument("--parts", type=int, default=3, help="max texts per doc")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    with open(args.pairs, encoding="utf-8") as f:
        texts = sorted({json.loads(line)["english"] for line in f if line.strip()})
    texts = [t for t in texts if len(t) >= args.min_part_chars]
    print(f"{len(texts)} usable component texts")

    model = SentenceTransformer(args.model)
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=True)

    # Small clusters (~2x parts) so every doc's components are near neighbors.
    k = max(2, len(texts) // (args.parts * 2))
    labels = KMeans(n_clusters=k, random_state=args.seed, n_init="auto").fit_predict(emb)

    by_cluster = {}
    for text, label in zip(texts, labels):
        by_cluster.setdefault(int(label), []).append(text)

    rng = random.Random(args.seed)
    docs = []
    for members in by_cluster.values():
        rng.shuffle(members)
        for i in range(0, len(members) - 1, args.parts):
            group = members[i:i + args.parts]
            if len(group) < 2:
                continue
            docs.append({"english": "\n\n".join(group),
                         "n_parts": len(group), "clustered": True})
    rng.shuffle(docs)
    docs = docs[: args.n]

    with open(args.out, "w", encoding="utf-8") as out:
        for doc in docs:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    lens = sorted(len(d["english"]) for d in docs)
    print(f"wrote {len(docs)} docs -> {args.out} "
          f"(chars p50={lens[len(lens)//2]}, p90={lens[int(len(lens)*.9)]})")


if __name__ == "__main__":
    main()
