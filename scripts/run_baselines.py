"""Run the study A baselines (no atoms) on the judgment questions.

Systems:
    random       articles in random order: the floor every system must beat
    bm25         keyword search over the article text
    dense        embedding search (default BAAI/bge-m3, which reads up to
                 8,192 tokens, so most facts sections fit whole)
    hybrid       BM25 + dense: sum of standardised scores (z-score fusion)
    hybrid_rrf   BM25 + dense: reciprocal rank fusion (k = 60)

Runs on the dev split by default. The test split is for the final run only,
once every system and setting is fixed, so it needs --final.

Every run writes, under outputs/tables/:

    baselines_<split>_summary.json     configuration + mean metrics per system
    baselines_<split>_per_query.csv    one row per question and system, for
                                       error analysis and paired tests

Dense embeddings are cached in data/processed/study_a/cache/ (git-ignored).

Examples (from the repo root):

    python scripts/run_baselines.py
    python scripts/run_baselines.py --systems random bm25
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

from src.similarity.retrieval import (  # noqa: E402
    article_documents, bm25_scores, dense_scores, evaluate, random_scores, rrf_fusion,
    zscore_fusion,
)

QUERIES = PROJECT_ROOT / "data" / "processed" / "study_a" / "queries.jsonl"
STATUTES = PROJECT_ROOT / "data" / "processed" / "statutes.jsonl"
CACHE = PROJECT_ROOT / "data" / "processed" / "study_a" / "cache"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

SYSTEMS = ["random", "bm25", "dense", "hybrid", "hybrid_rrf"]


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
    stem = OUT_DIR / f"baselines_{args.split}"
    pd.concat(per_query).to_csv(f"{stem}_per_query.csv", index=False)
    config = {k: v for k, v in vars(args).items() if k != "final"}
    Path(f"{stem}_summary.json").write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "config": config, "n_questions": len(items), "n_articles": len(ids),
        "results": summaries,
    }, indent=2), encoding="utf-8")
    print(f"-> {Path(f'{stem}_summary.json').relative_to(PROJECT_ROOT)}, "
          f"{Path(f'{stem}_per_query.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
