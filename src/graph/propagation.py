"""Score spreading over links between articles (study A).

A first-stage system scores every article for a question. Spreading then
lets each article pass part of its score to the articles it is linked to,
so an article that is linked to a strong match moves up even if the
question does not resemble it much. Two kinds of link, neither using atoms:

    citation   explicit cross-references in the statute text (the BWB
               ``intref`` elements in data/processed/statute_references.csv),
               treated as undirected: articles that refer to each other
    tree       articles in the same section (afdeling) of the legal tree,
               the structure G-DSR (Louis et al., 2023) builds on

Spreading: s' = s + beta * A s, with A symmetrically normalised
(D^-1/2 A D^-1/2), the usual choice in graph propagation. It keeps a
weakly linked article from receiving the full score of its single
neighbour, which the plain average of neighbours would do.

Usage:
    from src.graph.propagation import citation_adjacency, tree_adjacency, spread

    A = citation_adjacency(ids)
    scores = spread(first_stage_scores, A, beta=0.1)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_REFERENCES = Path("data/processed/statute_references.csv")
DEFAULT_STATUTES = Path("data/processed/statutes.jsonl")


def citation_adjacency(ids: list[str], path: Path | str = DEFAULT_REFERENCES) -> np.ndarray:
    """1 where one article refers to the other (either direction), else 0."""
    idx = {a: i for i, a in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)))
    with Path(path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s, t = idx.get(r["source_article"]), idx.get(r["target_article"])
            if s is not None and t is not None and s != t:
                A[s, t] = A[t, s] = 1.0
    return A


def tree_adjacency(ids: list[str], path: Path | str = DEFAULT_STATUTES) -> np.ndarray:
    """1 where two articles are in the same section (afdeling), else 0."""
    section = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            section[art["article_id"]] = art["path"].rsplit("/", 1)[0]
    secs = np.array([section[a] for a in ids])
    A = (secs[:, None] == secs[None, :]).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def normalise(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation D^-1/2 A D^-1/2; unlinked articles stay all-zero."""
    d = A.sum(axis=1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    return inv[:, None] * A * inv[None, :]


def minmax(scores: np.ndarray) -> np.ndarray:
    """Scale each question's scores to [0, 1], so beta means the same for every question."""
    lo = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - lo
    return (scores - lo) / np.where(span > 0, span, 1.0)


def spread(scores: np.ndarray, *links: tuple[np.ndarray, float]) -> np.ndarray:
    """One spreading step: s' = s + sum over link types of beta * norm(A) s.

    ``scores`` is questions x articles (scaled to [0, 1] first); each link is
    (adjacency matrix, beta).
    """
    base = minmax(scores)
    out = base.copy()
    for A, beta in links:
        out += beta * base @ normalise(A).T
    return out
