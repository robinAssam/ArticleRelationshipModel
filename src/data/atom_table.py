"""Atom table — flat DataFrame view of all annotated atoms across articles.

The loader reads every JSON file in `data/annotations/` and produces a
single pandas DataFrame, one row per atom. List-valued fields (actors,
acts, residual, etc.) remain as Python lists so they can be used directly
for set-intersection relation building downstream.

Usage:
    from src.data.atom_table import build_atom_table, summary_stats, tag_frequencies

    df = build_atom_table()              # default: data/annotations
    print(df.shape)                       # (n_atoms, n_columns)
    print(summary_stats(df))              # one row per article
    print(tag_frequencies(df, 'actors'))  # which actors appear how often
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.annotation.schema import load_atoms


# The 8 list-valued schema fields (Tasks 2–9, minus pre/post which is a string).
LIST_FIELDS = [
    "actors",
    "legal_relations",
    "acts",
    "geographical_domain",
    "temporal",
    "explicit_references",
    "hierarchies",
    "residual",
]

# Preferred column ordering for display.
COLUMN_ORDER = [
    "atom_id", "article_id", "lid", "condition_type",
    "text", "text_nl",
    *LIST_FIELDS,
    "source", "annotator", "annotated_at", "notes",
]


def build_atom_table(annotations_dir: str | Path = "data/annotations") -> pd.DataFrame:
    """Read every JSON in annotations_dir and return a flat atom DataFrame.

    One row per atom. List-valued fields remain Python lists.
    """
    annotations_dir = Path(annotations_dir)
    rows: list[dict] = []
    for json_file in sorted(annotations_dir.glob("*.json")):
        atoms = load_atoms(json_file)
        for atom in atoms:
            rows.append(atom.to_dict())

    if not rows:
        raise FileNotFoundError(
            f"No annotation files found in {annotations_dir.resolve()}"
        )

    df = pd.DataFrame(rows)
    # Reorder columns for readability; keep anything unknown at the end.
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra]


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per article: number of atoms, and per-field population counts."""
    rows = []
    for article, sub in df.groupby("article_id", sort=True):
        row = {
            "article_id": article,
            "n_atoms": len(sub),
            "n_pre": int((sub["condition_type"] == "pre_condition").sum()),
            "n_post": int((sub["condition_type"] == "post_condition").sum()),
        }
        for field in LIST_FIELDS:
            row[field] = int(sub[field].apply(lambda x: len(x) > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def tag_frequencies(df: pd.DataFrame, field: str) -> pd.Series:
    """Frequency of every tag value across atoms in a given field."""
    if field not in LIST_FIELDS:
        raise ValueError(
            f"Field '{field}' is not a list-valued schema field. "
            f"Expected one of: {LIST_FIELDS}"
        )
    return df[field].explode().dropna().value_counts()


def build_edge_table(
    df: pd.DataFrame,
    cross_article_only: bool = True,
    weights: dict | None = None,
) -> pd.DataFrame:
    """For every pair of atoms, list each shared field as one edge.

    Returns one row per (atom_a, atom_b, field) where the two atoms share
    at least one tag value in that field. Multiple rows per atom pair are
    expected — one per overlapping field.

    Parameters
    ----------
    df : pd.DataFrame
        The atom table from build_atom_table().
    cross_article_only : bool
        If True (default), only return edges between atoms in different
        articles. Set False to include within-article edges as well.
    weights : dict or None
        Optional mapping {(field, tag): weight} from compute_tag_weights().
        If provided, each edge gets a ``weighted_strength`` column equal to
        the sum of the weights of its shared tags. If None, weighted_strength
        equals strength (unit weight per shared tag).
    """
    from itertools import combinations

    rows = []
    for (_, a), (_, b) in combinations(df.iterrows(), 2):
        if cross_article_only and a["article_id"] == b["article_id"]:
            continue
        for field in LIST_FIELDS:
            overlap = set(a[field]) & set(b[field])
            if overlap:
                strength = len(overlap)
                if weights is not None:
                    weighted = sum(weights.get((field, tag), 0.0) for tag in overlap)
                else:
                    weighted = float(strength)
                rows.append({
                    "atom_a": a["atom_id"],
                    "atom_b": b["atom_id"],
                    "article_a": a["article_id"],
                    "article_b": b["article_id"],
                    "field": field,
                    "shared": sorted(overlap),
                    "strength": strength,
                    "weighted_strength": round(weighted, 4),
                })
    return pd.DataFrame(rows)


def compute_tag_weights(
    df: pd.DataFrame,
    method: str = "idf",
) -> dict:
    """Compute a discriminative weight for every (field, tag) pair in the corpus.

    Two methods:

    - ``"idf"`` (default): inverse document frequency. A tag's weight is
      ``log((N + 1) / (df + 1))`` where ``N`` is total atoms and ``df`` is
      the number of atoms containing the tag in that field. Common tags
      (appearing in many atoms) get low weight; rare tags get high weight.

    - ``"tf"``: raw document frequency. A tag's weight is the number of
      atoms it appears in. Common tags get high weight; rare tags low.

    Returns
    -------
    dict
        Mapping {(field, tag): weight}.
    """
    import math

    n_atoms = len(df)
    weights: dict = {}

    for field in LIST_FIELDS:
        # Count how many atoms each tag appears in (one count per atom even
        # if the same tag appears twice in that atom's list)
        atom_counts: dict[str, int] = {}
        for tags in df[field]:
            for tag in set(tags):
                atom_counts[tag] = atom_counts.get(tag, 0) + 1

        for tag, df_count in atom_counts.items():
            if method == "idf":
                weights[(field, tag)] = math.log((n_atoms + 1) / (df_count + 1))
            elif method == "tf":
                weights[(field, tag)] = float(df_count)
            else:
                raise ValueError(
                    f"Unknown method {method!r}. Use 'idf' or 'tf'."
                )

    return weights


def show_tag_weights(
    weights: dict,
    field: str | None = None,
    top: int | None = None,
) -> pd.DataFrame:
    """View tag weights as a sorted DataFrame.

    Parameters
    ----------
    weights : dict
        Output of compute_tag_weights().
    field : str or None
        Optional filter — show only tags from this schema field.
    top : int or None
        Optional cap — show only the top-N highest-weighted tags.
    """
    rows = [
        {"field": f, "tag": t, "weight": round(w, 4)}
        for (f, t), w in weights.items()
        if field is None or f == field
    ]
    out = pd.DataFrame(rows).sort_values("weight", ascending=False)
    if top is not None:
        out = out.head(top)
    return out.reset_index(drop=True)


def edges_per_atom(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Augment the atom table with a count of cross-article edges per atom.

    Useful for spotting hub atoms (many edges) and isolated atoms (zero).
    """
    counts_a = edges.groupby("atom_a").size().rename("edges_out")
    counts_b = edges.groupby("atom_b").size().rename("edges_in")
    augmented = df[["atom_id", "article_id", "lid", "condition_type"]].copy()
    augmented = augmented.merge(counts_a, left_on="atom_id",
                                right_index=True, how="left")
    augmented = augmented.merge(counts_b, left_on="atom_id",
                                right_index=True, how="left")
    augmented[["edges_out", "edges_in"]] = (
        augmented[["edges_out", "edges_in"]].fillna(0).astype(int)
    )
    augmented["total_edges"] = augmented["edges_out"] + augmented["edges_in"]
    return augmented.sort_values("total_edges", ascending=False)


def atom_pair_matrix(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long edge table into one row per atom pair.

    Each row shows the two atoms and how many edges of each field type
    connect them. Useful for finding the most tightly connected pairs.
    """
    pivot = (
        edges
        .groupby(["atom_a", "atom_b", "field"])["strength"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)


def add_edge_summary(
    df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-atom connectivity columns to the atom DataFrame.

    Adds three columns:
      ``n_edges``         — total edges this atom participates in
      ``connected_atoms`` — sorted list of atom IDs this atom is connected to
      ``edge_fields``     — sorted list of schema fields producing those edges
    """
    df = df.copy()
    n_edges: list[int] = []
    connected: list[list[str]] = []
    fields: list[list[str]] = []

    for atom_id in df["atom_id"]:
        if len(edges_df) == 0:
            n_edges.append(0)
            connected.append([])
            fields.append([])
            continue

        mask = (edges_df["atom_a"] == atom_id) | (edges_df["atom_b"] == atom_id)
        sub = edges_df[mask]
        n_edges.append(len(sub))

        others: set[str] = set()
        for _, e in sub.iterrows():
            others.add(e["atom_b"] if e["atom_a"] == atom_id else e["atom_a"])
        connected.append(sorted(others))
        fields.append(sorted(sub["field"].unique().tolist()))

    df["n_edges"] = n_edges
    df["connected_atoms"] = connected
    df["edge_fields"] = fields
    return df


def article_connectivity(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate edges to the article level: one row per article pair."""
    if len(edges_df) == 0:
        return pd.DataFrame(columns=["article_a", "article_b", "n_edges", "fields"])

    grouped = (
        edges_df.groupby(["article_a", "article_b"])
        .agg(
            n_edges=("field", "size"),
            fields=("field", lambda x: sorted(set(x))),
        )
        .reset_index()
    )
    return grouped


def show_atom(df: pd.DataFrame, atom_id: str) -> None:
    """Pretty-print a single atom by id."""
    matches = df[df["atom_id"] == atom_id]
    if matches.empty:
        print(f"No atom found with id {atom_id!r}")
        return
    row = matches.iloc[0]
    print()
    print("=" * 70)
    print(f"{row['atom_id']}   [{row['condition_type']}]")
    print("=" * 70)
    print(f"  EN: {row['text']}")
    print(f"  NL: {row['text_nl']}")
    print()
    for field in LIST_FIELDS:
        vals = row[field]
        if vals:
            print(f"  {field:22s} {vals}")
    if row.get("notes"):
        print()
        print(f"  notes: {row['notes']}")
    print()


# ============================================================================
# Semantic similarity layer
# ============================================================================

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def unique_tags_by_field(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return ``{field: sorted_unique_tags}`` for every list-valued field."""
    out: dict[str, list[str]] = {}
    for field in LIST_FIELDS:
        tags: set[str] = set()
        for tags_list in df[field]:
            tags.update(tags_list)
        out[field] = sorted(tags)
    return out


def compute_tag_similarities(
    df: pd.DataFrame,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    threshold: float = 0.75,
    model: object | None = None,
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between unique tag values.

    For each schema field, find pairs of distinct tag values whose embeddings
    have cosine similarity at or above ``threshold``. Exact matches are
    excluded (they're already captured by the exact-match channel).

    Parameters
    ----------
    df : pd.DataFrame
        Atom table from build_atom_table().
    model_name : str
        Hugging Face sentence-transformer checkpoint. Default is a small
        multilingual model (~120 MB) that handles Dutch reasonably well.
        Override with e.g. 'sentence-transformers/paraphrase-multilingual-
        mpnet-base-v2' for stronger results at higher cost.
    threshold : float
        Minimum cosine similarity (0.0–1.0) to include a pair. 0.75 is a
        starting point; tune based on what shows up.
    model : SentenceTransformer or None
        Optionally pass a pre-loaded model to skip re-downloading.

    Returns
    -------
    DataFrame with columns: field, tag_a, tag_b, similarity, sorted descending.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

    tags_by_field = unique_tags_by_field(df)

    # Embed every unique tag once (across all fields), then look up per field
    all_unique = sorted(set().union(*tags_by_field.values()))
    if not all_unique:
        return pd.DataFrame(columns=["field", "tag_a", "tag_b", "similarity"])

    import numpy as np
    embeddings = model.encode(
        all_unique, convert_to_numpy=True, normalize_embeddings=True
    )
    tag_to_emb = {t: embeddings[i] for i, t in enumerate(all_unique)}

    pairs = []
    for field, tags in tags_by_field.items():
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                ta, tb = tags[i], tags[j]
                sim = float(tag_to_emb[ta] @ tag_to_emb[tb])
                if sim >= threshold:
                    pairs.append({
                        "field": field,
                        "tag_a": ta,
                        "tag_b": tb,
                        "similarity": round(sim, 4),
                    })

    out = pd.DataFrame(pairs, columns=["field", "tag_a", "tag_b", "similarity"])
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


def build_semantic_edge_table(
    df: pd.DataFrame,
    sim_pairs: pd.DataFrame,
    cross_article_only: bool = True,
) -> pd.DataFrame:
    """Generate atom-level edges from tag-level semantic similarities.

    For each pair of atoms (A, B) and each field, find every pair of tags
    (tag_a in A's field, tag_b in B's field) that appears in ``sim_pairs``.
    Each atom pair becomes one row per field with at least one semantic match.

    Returns columns: atom_a, atom_b, article_a, article_b, field,
    matched_pairs (list of (tag_a, tag_b, similarity) tuples),
    n_matches, max_similarity, avg_similarity.
    """
    from itertools import combinations

    # Build a lookup: (field, tag) -> list of (other_tag, sim)
    lookup: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for _, row in sim_pairs.iterrows():
        key_a = (row["field"], row["tag_a"])
        key_b = (row["field"], row["tag_b"])
        lookup.setdefault(key_a, []).append((row["tag_b"], row["similarity"]))
        lookup.setdefault(key_b, []).append((row["tag_a"], row["similarity"]))

    rows = []
    for (_, a), (_, b) in combinations(df.iterrows(), 2):
        if cross_article_only and a["article_id"] == b["article_id"]:
            continue
        for field in LIST_FIELDS:
            tags_a = set(a[field])
            tags_b = set(b[field])
            if not tags_a or not tags_b:
                continue

            matches = []
            for ta in tags_a:
                for related_tag, sim in lookup.get((field, ta), []):
                    if related_tag in tags_b:
                        matches.append((ta, related_tag, sim))

            if matches:
                sims = [m[2] for m in matches]
                rows.append({
                    "atom_a": a["atom_id"],
                    "atom_b": b["atom_id"],
                    "article_a": a["article_id"],
                    "article_b": b["article_id"],
                    "field": field,
                    "matched_pairs": matches,
                    "n_matches": len(matches),
                    "max_similarity": round(max(sims), 4),
                    "avg_similarity": round(sum(sims) / len(sims), 4),
                })

    return pd.DataFrame(rows)


# ============================================================================
# Atom classifier
# ============================================================================

def classify_atom(
    query,
    df: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    top_k: int = 5,
) -> dict:
    """Score a query atom against every atom in the corpus and return the top-k.

    The query can be:
      - an Atom dataclass instance
      - a dict matching the atom schema
      - a partial dict containing only some fields (others default to empty)

    For each candidate atom in df, compute two scores:
      - exact_score:    sum of IDF weights of tags the query shares exactly
                        with the candidate, summed across fields
      - semantic_score: sum of best semantic-similarity matches between the
                        query's tags and the candidate's tags (excluding tags
                        already matched exactly)

    The final score is alpha * exact + (1 - alpha) * semantic. The function
    skips any candidate whose atom_id equals the query's atom_id (so you can
    pass an existing atom for self-comparison without it always topping itself).

    Returns
    -------
    dict with:
      'top_matches':       DataFrame of the top_k by combined_score
      'predicted_article': the most common article_id among the top_k
      'scores_by_atom':    the full ranked DataFrame (all candidates)
    """
    if hasattr(query, "to_dict"):
        query = query.to_dict()
    query = dict(query)  # shallow copy

    # Build a (field, tag) -> [(other_tag, sim), ...] lookup for semantic matches
    sem_lookup: dict[tuple[str, str], list[tuple[str, float]]] = {}
    if sim_pairs is not None and len(sim_pairs) > 0:
        for _, row in sim_pairs.iterrows():
            key_a = (row["field"], row["tag_a"])
            key_b = (row["field"], row["tag_b"])
            sem_lookup.setdefault(key_a, []).append((row["tag_b"], row["similarity"]))
            sem_lookup.setdefault(key_b, []).append((row["tag_a"], row["similarity"]))

    query_id = query.get("atom_id")
    rows = []

    for _, cand in df.iterrows():
        if cand["atom_id"] == query_id:
            continue

        exact_score = 0.0
        semantic_score = 0.0
        exact_matches: list[tuple[str, str, float]] = []
        semantic_matches: list[tuple[str, str, str, float]] = []

        for field in LIST_FIELDS:
            q_tags = set(query.get(field, []) or [])
            c_tags = set(cand[field])
            if not q_tags or not c_tags:
                continue

            # Exact overlap
            shared = q_tags & c_tags
            for tag in shared:
                w = weights.get((field, tag), 1.0) if weights else 1.0
                exact_score += w
                exact_matches.append((field, tag, round(w, 4)))

            # Semantic matches — for each query tag not exactly matched, find
            # the best semantically-similar tag on the candidate
            for q_tag in q_tags - shared:
                best_sim = 0.0
                best_match: str | None = None
                for related_tag, sim in sem_lookup.get((field, q_tag), []):
                    if related_tag in c_tags and sim > best_sim:
                        best_sim = sim
                        best_match = related_tag
                if best_match is not None:
                    semantic_score += best_sim
                    semantic_matches.append((field, q_tag, best_match, round(best_sim, 4)))

        combined = alpha * exact_score + (1.0 - alpha) * semantic_score
        rows.append({
            "atom_id": cand["atom_id"],
            "article_id": cand["article_id"],
            "combined_score": round(combined, 4),
            "exact_score": round(exact_score, 4),
            "semantic_score": round(semantic_score, 4),
            "exact_matches": exact_matches,
            "semantic_matches": semantic_matches,
        })

    scores_df = pd.DataFrame(rows).sort_values(
        "combined_score", ascending=False
    ).reset_index(drop=True)

    top = scores_df.head(top_k)
    predicted_article = top["article_id"].mode().iloc[0] if len(top) else None

    return {
        "top_matches": top[["atom_id", "article_id", "combined_score",
                            "exact_score", "semantic_score"]],
        "predicted_article": predicted_article,
        "scores_by_atom": scores_df,
    }


def evaluate_classifier_loo(
    df: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Leave-one-out evaluation of the atom classifier.

    For each atom in df: remove it, classify it against the rest, then check
    whether the correct article appears at rank 1, in the top_k, and where.

    Returns
    -------
    eval_df : DataFrame with per-atom results (atom_id, true_article,
              predicted_article, top1_hit, topk_hit, rank, reciprocal_rank)
    summary : dict with overall top-1 accuracy, top-k accuracy, and MRR
    """
    results = []
    for idx, atom in df.iterrows():
        rest = df.drop(idx)
        out = classify_atom(
            atom.to_dict(), rest,
            weights=weights, sim_pairs=sim_pairs,
            alpha=alpha, top_k=top_k,
        )
        top = out["top_matches"]
        true_article = atom["article_id"]

        top1_article = top.iloc[0]["article_id"] if len(top) else None
        top1_hit = top1_article == true_article
        topk_hit = (true_article in top["article_id"].values) if len(top) else False

        rank: int | None = None
        for r, art in enumerate(top["article_id"].values, start=1):
            if art == true_article:
                rank = r
                break

        results.append({
            "atom_id": atom["atom_id"],
            "true_article": true_article,
            "top1_match": top1_article,
            "predicted_article": out["predicted_article"],
            "top1_hit": top1_hit,
            "topk_hit": topk_hit,
            "rank": rank,
            "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        })

    eval_df = pd.DataFrame(results)
    summary = {
        "n_atoms": len(eval_df),
        "top1_accuracy": round(eval_df["top1_hit"].mean(), 4),
        f"top{top_k}_accuracy": round(eval_df["topk_hit"].mean(), 4),
        "mean_reciprocal_rank": round(eval_df["reciprocal_rank"].mean(), 4),
    }
    return eval_df, summary
