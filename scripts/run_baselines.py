"""Run the study A baselines (no atoms) on the judgment questions.

Systems:
    random       articles in random order: the floor every system must beat
    bm25         keyword search over the article text
    dense        embedding search (default BAAI/bge-m3, which reads up to
                 8,192 tokens, so most facts sections fit whole)
    hybrid       BM25 + dense: sum of standardised scores (z-score fusion)
    hybrid_rrf   BM25 + dense: reciprocal rank fusion (k = 60)
    graph_cite   hybrid_rrf, then scores spread along citations between articles
    graph_tree   hybrid_rrf, then scores spread within sections of the legal tree
    graph_both   hybrid_rrf, then both

The spreading strengths (--beta-*) default to the values chosen on the dev
split by nDCG@10; --sweep re-evaluates a grid of them.

Runs on the dev split by default. The test split is for the final run only,
once every system and setting is fixed, so it needs --final.

Every run writes, under outputs/tables/:

    baselines_<split>_summary.json     configuration + mean metrics per system
    baselines_<split>_per_query.csv    one row per question and system, for
                                       error analysis and paired tests
    graph_sweep_<split>.csv            with --sweep: one row per beta setting

Dense embeddings are cached in data/processed/study_a/cache/ (git-ignored).

Examples (from the repo root):

    python scripts/run_baselines.py
    python scripts/run_baselines.py --systems random bm25
    python scripts/run_baselines.py --sweep
    python scripts/run_baselines.py --split test --final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.graph.propagation import citation_adjacency, spread, tree_adjacency  # noqa: E402
from src.similarity.retrieval import (  # noqa: E402
    article_documents, bm25_scores, dense_scores, evaluate, random_scores, rrf_fusion,
    zscore_fusion,
)

QUERIES = PROJECT_ROOT / "data" / "processed" / "study_a" / "queries.jsonl"
STATUTES = PROJECT_ROOT / "data" / "processed" / "statutes.jsonl"
CACHE = PROJECT_ROOT / "data" / "processed" / "study_a" / "cache"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

SYSTEMS = ["random", "bm25", "dense", "hybrid", "hybrid_rrf",
           "graph_cite", "graph_tree", "graph_both"]
SWEEP_BETAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--final", action="store_true",
                   help="required for --split test: confirms this is the final run")
    p.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    p.add_argument("--model", default="BAAI/bge-m3", help="sentence-transformer for dense")
    p.add_argument("--max-length", type=int, default=8192,
                   help="tokens the dense model reads; longer texts are truncated")
    p.add_argument("--random-draws", type=int, default=100,
                   help="random orders averaged for the random baseline")
    p.add_argument("--beta-cite", type=float, default=0.25,
                   help="spreading strength along citations, graph_cite (chosen on dev)")
    p.add_argument("--beta-tree", type=float, default=1.0,
                   help="spreading strength within sections, graph_tree (chosen on dev)")
    p.add_argument("--beta-both", type=float, nargs=2, default=[0.1, 1.0],
                   metavar=("CITE", "TREE"), help="both strengths for graph_both (chosen on dev)")
    p.add_argument("--sweep", action="store_true",
                   help="also evaluate a grid of spreading strengths -> graph_sweep_<split>.csv")
    return p.parse_args()


def _dense(questions: list[str], docs: list[str], args) -> np.ndarray:
    """Dense scores, cached per model, length limit and input texts."""
    key = hashlib.sha256(json.dumps([args.model, args.max_length, questions, docs]).encode())
    path = CACHE / f"dense_{args.model.replace('/', '_')}_{key.hexdigest()[:12]}.npy"
    if path.exists():
        print(f"dense: cached scores {path.name}")
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = args.max_length
    scores = dense_scores(questions, docs, model)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    return scores


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.final:
        sys.exit("The test split is for the final run only. Add --final if that is now.")

    items = [json.loads(line) for line in QUERIES.open(encoding="utf-8")]
    items = [it for it in items if it["split"] == args.split]
    questions = [it["question"] for it in items]
    answers = [it["answers"] for it in items]
    ids, docs = article_documents(STATUTES)
    print(f"{len(items)} {args.split} questions, {len(ids)} articles")

    computed: dict[str, np.ndarray] = {}
    cite, tree = citation_adjacency(ids), tree_adjacency(ids, STATUTES)

    def scores_of(system: str) -> np.ndarray:
        if system not in computed:
            if system == "bm25":
                computed[system] = bm25_scores(questions, docs)
            elif system == "dense":
                computed[system] = _dense(questions, docs, args)
            elif system == "hybrid":
                computed[system] = zscore_fusion(scores_of("bm25"), scores_of("dense"))
            elif system == "hybrid_rrf":
                computed[system] = rrf_fusion(scores_of("bm25"), scores_of("dense"))
            elif system == "graph_cite":
                computed[system] = spread(scores_of("hybrid_rrf"), (cite, args.beta_cite))
            elif system == "graph_tree":
                computed[system] = spread(scores_of("hybrid_rrf"), (tree, args.beta_tree))
            elif system == "graph_both":
                bc, bt = args.beta_both
                computed[system] = spread(scores_of("hybrid_rrf"), (cite, bc), (tree, bt))
        return computed[system]

    summaries, per_query = {}, []
    for system in args.systems:
        if system == "random":
            runs = [evaluate(random_scores(len(items), len(ids), seed=s), ids, answers)
                    for s in range(args.random_draws)]
            pq = sum(r[0] for r in runs) / len(runs)
            summary = {k: round(float(np.mean([r[1][k] for r in runs])), 4) for k in runs[0][1]}
        else:
            pq, summary = evaluate(scores_of(system), ids, answers)
        summaries[system] = summary
        pq.insert(0, "system", system)
        pq.insert(1, "ecli", [it["ecli"] for it in items])
        pq.insert(2, "n_answers", [len(a) for a in answers])
        per_query.append(pq)
        print(f"{system:>10}: " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        rows = []
        grid = ([("cite", b, 0.0) for b in SWEEP_BETAS] + [("tree", 0.0, b) for b in SWEEP_BETAS]
                + [("both", bc, bt) for bc in (0.1, 0.25) for bt in (0.5, 1.0, 2.0)])
        for links, bc, bt in grid:
            fused = spread(scores_of("hybrid_rrf"), (cite, bc), (tree, bt))
            rows.append({"links": links, "beta_cite": bc, "beta_tree": bt,
                         **evaluate(fused, ids, answers)[1]})
        sweep = pd.DataFrame(rows)
        sweep.to_csv(OUT_DIR / f"graph_sweep_{args.split}.csv", index=False)
        best = sweep.loc[sweep.groupby("links")["ndcg@10"].idxmax()]
        print("best spreading strength per link type (by nDCG@10):")
        print(best[["links", "beta_cite", "beta_tree", "mrr", "recall@10", "ndcg@10"]]
              .to_string(index=False))

    stem = OUT_DIR / f"baselines_{args.split}"
    pd.concat(per_query).to_csv(f"{stem}_per_query.csv", index=False)
    config = {k: v for k, v in vars(args).items() if k not in ("final", "sweep")}
    Path(f"{stem}_summary.json").write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": config, "n_questions": len(items), "n_articles": len(ids),
        "results": summaries,
    }, indent=2), encoding="utf-8")
    print(f"-> {Path(f'{stem}_summary.json').relative_to(PROJECT_ROOT)}, "
          f"{Path(f'{stem}_per_query.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
