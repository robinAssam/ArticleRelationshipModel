"""Baseline retrieval over the article corpus, and multi-answer metrics.

Study A compares retrieval systems on real judgments: each question is the
facts section of a judgment, the answers are the corpus articles it cites
(scripts/build_study_a.py). This module holds the systems that need no
atoms, and the metrics every system is scored with:

    article_documents()       one text per article from statutes.jsonl
    bm25_scores(...)          keyword search (BM25 over lowercased words)
    dense_scores(...)         embedding search (cosine of sentence embeddings)
    zscore_fusion(...)        hybrid: sum of standardised scores of several systems
    rrf_fusion(...)           hybrid: reciprocal rank fusion of several systems
    rank_metrics(...)         recall@k, MRR and nDCG@k for one question
    evaluate(...)             metrics for every question, plus the mean

A system is just a matrix of scores (questions x articles); evaluate() turns
it into metrics. That keeps systems comparable: the atom-based systems later
produce the same matrix.

Usage:
    from src.similarity.retrieval import article_documents, bm25_scores, evaluate

    ids, docs = article_documents()
    scores = bm25_scores(questions, docs)
    per_query, mean = evaluate(scores, ids, answers)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STATUTES = Path("data/processed/statutes.jsonl")
KS = (1, 5, 10, 20)
NDCG_K = 10

# Common Dutch function words, plus the masks of the study A questions
_STOPWORDS = set("""
aan al alle alles als dan dat de deze die dit door dus een eens en er ge geen
had heb hebben heeft hem het hier hij hoe hun ik in is ja je kan kon kunnen
maar me meer men met mij mijn moet na naar niet niets nog nu of om omdat ook
op over reeds te tegen toch toen tot u uit uw van veel voor want waren was wat
we wel werd wezen wie wij wil worden wordt zal ze zelf zich zij zijn zo zonder
zou artikel wet
""".split())
_WORD = re.compile(r"[a-zà-ÿ]+(?:-[a-zà-ÿ]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased words without Dutch stopwords or one-letter tokens."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def article_documents(path: Path | str = DEFAULT_STATUTES) -> tuple[list[str], list[str]]:
    """(article ids, article texts) in corpus order. The text is the statute
    text only: no article number (the questions have theirs masked) and no
    headings from the legal tree (that structure is a later system)."""
    ids, docs = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            ids.append(art["article_id"])
            docs.append(art["text"])
    return ids, docs


# ------------------------------------------------------------------- systems

def bm25_scores(questions: list[str], docs: list[str], k1: float = 1.5,
                b: float = 0.75) -> np.ndarray:
    """BM25 (Okapi) score of every article for every question."""
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(d) for d in docs], k1=k1, b=b)
    return np.vstack([bm25.get_scores(tokenize(q)) for q in questions])


def dense_scores(questions: list[str], docs: list[str], model,
                 batch_size: int = 4) -> np.ndarray:
    """Cosine similarity of question and article embeddings.

    ``model`` is a loaded SentenceTransformer; texts longer than its
    ``max_seq_length`` are truncated by the model.
    """
    q = model.encode(questions, batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=True)
    d = model.encode(docs, batch_size=batch_size, normalize_embeddings=True)
    return q @ d.T


def random_scores(n_questions: int, n_docs: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n_questions, n_docs))


def zscore_fusion(*score_matrices: np.ndarray) -> np.ndarray:
    """Sum of per-question standardised scores (equal weights).

    BM25 scores and cosine similarities live on different scales; standardising
    each question's scores (mean 0, sd 1) makes them addable. An article ranks
    high when both systems like it, or one system likes it a lot.
    """
    def z(s: np.ndarray) -> np.ndarray:
        sd = s.std(axis=1, keepdims=True)
        return (s - s.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)

    return sum(z(s) for s in score_matrices)


def rrf_fusion(*score_matrices: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over the systems.

    Uses only each system's ranking, not its scores, so no scaling is needed;
    k = 60 is the usual setting (Cormack et al., 2009). Ties within a system
    share the average rank.
    """
    import scipy.stats

    ranks = [scipy.stats.rankdata(-s, axis=1) for s in score_matrices]
    return sum(1.0 / (k + r) for r in ranks)


# ------------------------------------------------------------------- metrics

def rank_metrics(ranked: list[str], answers: set[str], ks=KS, ndcg_k: int = NDCG_K) -> dict:
    """Metrics for one question, given the articles in ranked order.

    recall@k   share of the answers in the top k
    mrr        1 / rank of the first answer
    ndcg@k     rank-discounted share of answers near the top (1 = all at the top)
    """
    hits = [a in answers for a in ranked]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits[:ndcg_k]) if h)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(answers), ndcg_k)))
    return {
        "first_rank": None if first is None else first + 1,
        "mrr": 0.0 if first is None else 1 / (first + 1),
        **{f"recall@{k}": sum(hits[:k]) / len(answers) for k in ks},
        f"ndcg@{ndcg_k}": dcg / ideal if ideal else 0.0,
    }


def _ranked_ids(row: np.ndarray, ids: list[str], rng: np.random.Generator) -> list[str]:
    """Articles by descending score; ties broken at random."""
    order = np.lexsort((rng.random(len(ids)), -row))
    return [ids[i] for i in order]


def evaluate(scores: np.ndarray, ids: list[str], answers: list[list[str]],
             tie_draws: int = 20, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per-question metrics and their mean for a questions x articles score matrix.

    Articles with equal scores have no meaningful order, so when a question
    has ties the metrics are averaged over ``tie_draws`` random orders of the
    tied articles (an estimate of the expected value). Without ties one order
    is exact.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for row, ans in zip(scores, answers):
        draws = tie_draws if len(np.unique(row)) < len(row) else 1
        ms = [rank_metrics(_ranked_ids(row, ids, rng), set(ans)) for _ in range(draws)]
        mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != "first_rank"}
        firsts = [m["first_rank"] for m in ms if m["first_rank"] is not None]
        mean["first_rank"] = float(np.mean(firsts)) if firsts else None
        rows.append(mean)
    per_query = pd.DataFrame(rows)
    summary = {c: round(float(per_query[c].mean()), 4) for c in per_query if c != "first_rank"}
    return per_query, summary
