"""Chained reasoning over the atom graph.

Given an anchor query (an atom or atom-shaped dict), walk the graph to
surface related atoms and articles through three complementary channels:

- Cross-article reference edges: explicit statutory citations
- Shared-tag edges: atoms that overlap on annotation tags (IDF-weighted)
- Classifier ranking: semantic and tag-based similarity to the query

Returns a structured trace that records *how* each surfaced atom was
reached, so the result is auditable rather than a black-box list. This is
what makes chained legal reasoning look like reasoning rather than
retrieval — every step names its edge.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.atom_table import classify_atom


def chained_reasoning(
    query: dict,
    df: pd.DataFrame,
    ref_edges: pd.DataFrame,
    edges: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    top_k: int = 3,
    max_hops: int = 1,
    min_edge_strength: float = 2.0,
    max_expansions_per_atom: int = 4,
    sem_lookup: dict | None = None,
) -> dict:
    """Walk the graph from an anchor query to produce a chained trace.

    Parameters
    ----------
    query : dict
        Atom-shaped dict (may be partial — missing fields default to []).
    df : pd.DataFrame
        The full atom table.
    ref_edges : pd.DataFrame
        From build_reference_edges(). Columns: source, target, edge_kind, ref_text.
    edges : pd.DataFrame
        From build_edge_table(). One row per (atom_a, atom_b, field).
    weights, sim_pairs, alpha, top_k
        Passed to classify_atom() for the anchor step.
    max_hops : int
        How many expansion hops to walk. 1 is usually enough for demos.
    min_edge_strength : float
        Ignore tag-overlap edges with weighted_strength below this.
    max_expansions_per_atom : int
        Cap expansions per source atom so the chain stays legible.
    sem_lookup : dict or None
        Prebuilt tag-similarity lookup (see build_semantic_lookup); takes
        precedence over ``sim_pairs`` for the anchor step.

    Returns
    -------
    dict with keys:
        query, primary_matches, chain, articles_touched
    """
    # Anchor: classify the query
    result = classify_atom(
        query, df,
        weights=weights, sim_pairs=sim_pairs, sem_lookup=sem_lookup,
        alpha=alpha, top_k=top_k,
    )
    primary = result["top_matches"]

    visited: set[str] = set(primary["atom_id"].tolist())
    articles: set[str] = set(primary["article_id"].tolist())
    chain: list[dict] = []

    # Hop 0: record the anchors
    for rank, (_, row) in enumerate(primary.iterrows(), start=1):
        atom_id = row["atom_id"]
        atom_row = df[df["atom_id"] == atom_id].iloc[0]
        chain.append({
            "atom_id":    atom_id,
            "article_id": atom_row["article_id"],
            "text":       atom_row["text"],
            "hop":        0,
            "via":        f"classifier rank {rank}",
            "edge_kind":  "classifier",
            "score":      float(row["combined_score"]),
        })

    # Hops 1..max_hops: expand via graph edges
    frontier = list(visited)
    for hop in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for src in frontier:
            expansions = _expand_atom(
                src, df, ref_edges, edges,
                visited=visited,
                min_edge_strength=min_edge_strength,
                limit=max_expansions_per_atom,
            )
            for exp in expansions:
                exp["hop"] = hop
                chain.append(exp)
                visited.add(exp["atom_id"])
                articles.add(exp["article_id"])
                next_frontier.append(exp["atom_id"])
        frontier = next_frontier
        if not frontier:
            break

    return {
        "query":            query,
        "primary_matches":  primary,
        "chain":            chain,
        "articles_touched": sorted(articles),
    }


def _expand_atom(
    source_atom: str,
    df: pd.DataFrame,
    ref_edges: pd.DataFrame,
    edges: pd.DataFrame,
    visited: set[str],
    min_edge_strength: float,
    limit: int,
) -> list[dict]:
    """One-hop expansion from a source atom via all three edge channels."""
    expansions: list[dict] = []

    src_row = df[df["atom_id"] == source_atom]
    if src_row.empty:
        return []
    src_row = src_row.iloc[0]
    src_article = src_row["article_id"]

    # 1) Outgoing reference edges (source atom cites another article)
    if ref_edges is not None and len(ref_edges):
        out = ref_edges[ref_edges["source"] == source_atom]
        for _, ref in out.iterrows():
            target_article = ref["target"]
            for _, ta in df[df["article_id"] == target_article].iterrows():
                if ta["atom_id"] in visited:
                    continue
                expansions.append({
                    "atom_id":    ta["atom_id"],
                    "article_id": target_article,
                    "text":       ta["text"],
                    "via":        f"{source_atom} references '{ref['ref_text']}'",
                    "edge_kind":  "reference_out",
                    "score":      None,
                })

    # 2) Incoming reference edges (another atom cites source's article)
    if ref_edges is not None and len(ref_edges):
        inc = ref_edges[ref_edges["target"] == src_article]
        for _, ref in inc.iterrows():
            citer = ref["source"]
            if citer in visited or citer == source_atom:
                continue
            citer_row = df[df["atom_id"] == citer]
            if citer_row.empty:
                continue
            citer_row = citer_row.iloc[0]
            expansions.append({
                "atom_id":    citer,
                "article_id": citer_row["article_id"],
                "text":       citer_row["text"],
                "via":        f"{citer} references {src_article} ('{ref['ref_text']}')",
                "edge_kind":  "reference_in",
                "score":      None,
            })

    # 3) Strong shared-tag edges. Aggregate multi-field rows per atom pair
    #    so a pair sharing tags across 3 fields is judged on its total, not
    #    on any single field's strength.
    if edges is not None and len(edges):
        strength_col = "weighted_strength" if "weighted_strength" in edges.columns else "strength"
        rel = edges[(edges["atom_a"] == source_atom) | (edges["atom_b"] == source_atom)]
        if len(rel):
            def _partner(row):
                return row["atom_b"] if row["atom_a"] == source_atom else row["atom_a"]
            rel = rel.copy()
            rel["partner"] = rel.apply(_partner, axis=1)
            agg = (
                rel.groupby("partner")
                   .agg(total_strength=(strength_col, "sum"),
                        fields=("field", list),
                        shared_tags=("shared", list))
                   .reset_index()
                   .sort_values("total_strength", ascending=False)
            )
            agg = agg[agg["total_strength"] >= min_edge_strength]
            for _, row in agg.iterrows():
                partner = row["partner"]
                if partner in visited:
                    continue
                partner_row = df[df["atom_id"] == partner]
                if partner_row.empty:
                    continue
                partner_row = partner_row.iloc[0]
                # Flatten shared tags for display
                all_shared = [t for sub in row["shared_tags"] for t in sub]
                expansions.append({
                    "atom_id":    partner,
                    "article_id": partner_row["article_id"],
                    "text":       partner_row["text"],
                    "via":        f"shares {all_shared} with {source_atom}",
                    "edge_kind":  "shared_tags",
                    "score":      float(row["total_strength"]),
                })

    # Deduplicate: same atom might come up via references and shared tags;
    # prefer references (they're stronger legal evidence than tag overlap).
    priority = {"reference_out": 0, "reference_in": 1, "shared_tags": 2}
    expansions.sort(key=lambda e: priority.get(e["edge_kind"], 3))
    seen: set[str] = set()
    deduped: list[dict] = []
    for exp in expansions:
        if exp["atom_id"] in seen:
            continue
        seen.add(exp["atom_id"])
        deduped.append(exp)

    return deduped[:limit]


def explain_chain(result: dict, max_text_len: int = 110) -> str:
    """Render a chained-reasoning result as a human-readable trace."""
    lines: list[str] = []

    # Query
    q = result["query"]
    q_label = q.get("atom_id", "(custom query)") if isinstance(q, dict) else str(q)
    lines.append(f"Query: {q_label}")
    if isinstance(q, dict):
        tags_shown = []
        for field in ["actors", "acts", "legal_relations", "residual"]:
            v = q.get(field)
            if v:
                tags_shown.append(f"{field}={v}")
        if tags_shown:
            lines.append("  tags: " + " · ".join(tags_shown))
    lines.append("")

    # Chain by hop
    chain = result["chain"]
    max_hop = max((c["hop"] for c in chain), default=0)
    for hop in range(max_hop + 1):
        hop_items = [c for c in chain if c["hop"] == hop]
        if not hop_items:
            continue
        if hop == 0:
            lines.append(f"Anchor matches (hop 0, classifier):")
        else:
            lines.append(f"Hop {hop} — graph expansion:")
        for c in hop_items:
            text = c["text"]
            if len(text) > max_text_len:
                text = text[: max_text_len - 1].rstrip() + "…"
            score_str = ""
            if c.get("score") is not None:
                score_str = f"  [{c['edge_kind']}, score={c['score']:.2f}]"
            else:
                score_str = f"  [{c['edge_kind']}]"
            lines.append(f"  {c['atom_id']:14s}  via: {c['via']}{score_str}")
            lines.append(f"                  → {text}")
        lines.append("")

    lines.append(f"Articles surfaced: {', '.join(result['articles_touched'])}")
    return "\n".join(lines)
