"""Run the article-level leave-one-out evaluation and save the results.

Thesis tables should come from files, not from whatever a notebook printed
last. Every run writes, under outputs/tables/:

    <name>_summary.json    the configuration used + the summary metrics
    <name>_per_query.csv   one row per query (atom), for error analysis

With --sweep it instead evaluates every normalize x alpha combination and
writes <name>_sweep.csv (one row per combination) plus the summary JSON.

Examples (from the repo root):

    python scripts/run_eval.py
    python scripts/run_eval.py --name no_semantic --alpha 1.0
    python scripts/run_eval.py --name sweep --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.atom_table import (  # noqa: E402
    NORMALIZE_MODES, build_atom_table, compute_tag_similarities, drop_synthetic,
    evaluate_article_retrieval,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", default="article_retrieval", help="prefix for output files")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--normalize", choices=NORMALIZE_MODES, default="cosine")
    p.add_argument("--agg", choices=["max", "sum"], default="max",
                   help="how atom scores become an article score")
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2",
                   help="sentence-transformer for the semantic channel; 'none' disables it")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="minimum tag similarity for a semantic match")
    p.add_argument("--filter-fields", nargs="*", default=None,
                   help="pre-filter on these fields, e.g. --filter-fields actors acts")
    p.add_argument("--sweep", action="store_true",
                   help="evaluate every normalize x alpha combination")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Filter left no candidates")

    df = drop_synthetic(build_atom_table(PROJECT_ROOT / "data" / "annotations"))
    print(f"{len(df)} atoms across {df['article_id'].nunique()} real articles")

    sim_pairs = None
    if args.encoder != "none":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.encoder)
        sim_pairs = compute_tag_similarities(df, threshold=args.threshold, model=model)
        print(f"{len(sim_pairs)} semantic tag pairs at threshold {args.threshold}")

    config = {
        **vars(args),
        "n_atoms": len(df),
        "n_articles": int(df["article_id"].nunique()),
        "run_at": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        rows = []
        for norm in NORMALIZE_MODES:
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                _, s = evaluate_article_retrieval(
                    df, sim_pairs=sim_pairs, alpha=alpha, normalize=norm,
                    article_agg=args.agg, filter_fields=args.filter_fields,
                )
                rows.append({"normalize": norm, "alpha": alpha, **s})
        table = pd.DataFrame(rows)
        table.to_csv(OUT_DIR / f"{args.name}_sweep.csv", index=False)
        summary = {"sweep_rows": len(rows)}
        print(table[["normalize", "alpha", "mrr", "hit@1", "hit@3", "hit@10"]].to_string(index=False))
    else:
        per_query, summary = evaluate_article_retrieval(
            df, sim_pairs=sim_pairs, alpha=args.alpha, normalize=args.normalize,
            article_agg=args.agg, filter_fields=args.filter_fields,
        )
        per_query.to_csv(OUT_DIR / f"{args.name}_per_query.csv", index=False)
        for k, v in summary.items():
            print(f"  {k:24s} {v}")

    with open(OUT_DIR / f"{args.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "summary": summary}, f, indent=2)
    print(f"Saved to {OUT_DIR.relative_to(PROJECT_ROOT)}/{args.name}_*")


if __name__ == "__main__":
    main()
