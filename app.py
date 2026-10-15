"""Streamlit UI for the Article Relationship Model.

Run with (from the repo root):

    streamlit run app.py

The app loads the atom corpus, builds the edge tables once, and lets a user
run chained-reasoning queries against the graph via either a preset dropdown
or a custom tag builder. Results are shown in three panels: anchor matches,
the reasoning chain, and a table + plotly subgraph of the articles surfaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project-root setup — lets the app run from anywhere
# ---------------------------------------------------------------------------

def _find_project_root(marker: str = "src/data/atom_table.py") -> Path:
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / marker).exists():
            return p
    # Fall back: assume this file sits at the project root
    here = Path(__file__).resolve().parent
    if (here / marker).exists():
        return here
    raise RuntimeError(f"Could not locate project root (no {marker} found)")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.atom_table import (  # noqa: E402
    build_atom_table, build_edge_table, build_semantic_lookup, classify_atom,
    compute_tag_weights, compute_tag_similarities, drop_synthetic,
    unique_tags_by_field, LIST_FIELDS,
)
from src.data.hierarchy import (  # noqa: E402
    add_taxonomy_columns, build_reference_edges,
)
from src.data.query import (  # noqa: E402
    rank_articles_for_query_atoms, text_to_query_atoms,
)
from src.data.reasoning import chained_reasoning  # noqa: E402


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Article Relationship Model",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Pipeline loading — cached across the whole session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading atoms, edges, and encoder…")
def load_pipeline() -> dict:
    """One-time pipeline load. Rebuilds only if the underlying files change."""
    # Real articles only: the synthetic Lunar corpus is a test fixture.
    df = drop_synthetic(build_atom_table(PROJECT_ROOT / "data" / "annotations"))
    df_tax = add_taxonomy_columns(df)
    weights_idf = compute_tag_weights(df, method="idf")
    edges_weighted = build_edge_table(df, weights=weights_idf)
    ref_edges = build_reference_edges(df_tax)

    # Semantic similarity is the slowest step — first run downloads ~420 MB
    # and takes ~30 seconds. Subsequent runs are instant thanks to caching.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    sim_pairs = compute_tag_similarities(df, threshold=0.55, model=model)

    return {
        "df": df,
        "df_tax": df_tax,
        "weights_idf": weights_idf,
        "edges_weighted": edges_weighted,
        "sim_pairs": sim_pairs,
        "sem_lookup": build_semantic_lookup(sim_pairs),
        "ref_edges": ref_edges,
        "tags_by_field": unique_tags_by_field(df),
    }


# ---------------------------------------------------------------------------
# Preset queries — the three validated in §10 of the notebook
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "Can a patient request their file be destroyed?": {
        "actors": ["patient", "care provider"],
        "acts": ["destroys"],
        "residual": ["file", "request"],
        "temporal": ["after request"],
    },
    "Care provider failed to keep proper records, causing harm": {
        "actors": ["care provider", "patient"],
        "acts": ["compensates"],
        "legal_relations": ["obligation"],
        "residual": ["damage", "failure to perform", "file"],
    },
    "Hospital being sued for a doctor's mistake": {
        "actors": ["employer", "subordinate", "third party"],
        "legal_relations": ["service relationship"],
        "residual": ["damage", "fault", "liability"],
    },
}


def _query_template() -> dict:
    """Empty atom-shaped query with all fields present."""
    q: dict = {"atom_id": "UI_QUERY", "article_id": "UNKNOWN"}
    for field in LIST_FIELDS:
        q[field] = []
    return q


def _merge_query_atoms(atoms: list[dict]) -> dict:
    """Union the tags of several query atoms into one query for the walker."""
    q = _query_template()
    for atom in atoms:
        for field in LIST_FIELDS:
            for tag in atom.get(field, []):
                if tag not in q[field]:
                    q[field].append(tag)
    return q


@st.cache_data(show_spinner=False)
def decompose_question(question: str, vocabulary: dict[str, list[str]]) -> list[dict]:
    """LLM call, cached so re-running the same question costs nothing."""
    return text_to_query_atoms(question, vocabulary=vocabulary)


QUESTION_EXAMPLE = (
    "Het ziekenhuis heeft mijn medisch dossier kwijtgeraakt en nu kan ik "
    "niet bewijzen dat de behandeling verkeerd is gegaan."
)


# ---------------------------------------------------------------------------
# Sidebar: query builder
# ---------------------------------------------------------------------------

def build_sidebar_query(tags_by_field: dict[str, list[str]]) -> tuple[dict, str, dict, str | None]:
    """Render the sidebar UI.

    Returns the query dict, a label for display, the parameters, and the
    question text when the plain-language mode is used (else None).
    """
    st.sidebar.header("Query")
    mode = st.sidebar.radio(
        "Input mode",
        ["Plain-language question", "Preset query", "Custom tag builder"],
        index=0,
    )

    query = _query_template()
    label = ""
    question: str | None = None

    if mode == "Plain-language question":
        question = st.sidebar.text_area(
            "Describe the legal problem (Dutch or English)",
            QUESTION_EXAMPLE, height=140,
        ).strip()
        label = question
        st.sidebar.caption(
            "An LLM splits the question into atoms and labels them with the "
            "same 9-label schema as the corpus. Needs an Anthropic API key; "
            "each new question costs a few cents."
        )

    elif mode == "Preset query":
        preset_name = st.sidebar.selectbox(
            "Choose a query",
            list(PRESETS.keys()),
            index=0,
        )
        label = preset_name
        # Populate query from preset (leave unspecified fields empty)
        for field, values in PRESETS[preset_name].items():
            query[field] = list(values)

        # Show what tags the preset uses
        with st.sidebar.expander("Preset tags", expanded=False):
            for field in LIST_FIELDS:
                if query.get(field):
                    st.write(f"**{field}**: {', '.join(query[field])}")

    else:  # Custom tag builder
        label = "Custom query"
        st.sidebar.caption(
            "Pick tags from the corpus vocabulary. Empty fields are ignored."
        )
        primary_fields = ["actors", "acts", "legal_relations", "residual", "temporal"]
        for field in primary_fields:
            options = sorted(tags_by_field.get(field, []))
            query[field] = st.sidebar.multiselect(
                field.replace("_", " ").title(),
                options,
                key=f"tag_{field}",
            )
        with st.sidebar.expander("Advanced fields", expanded=False):
            for field in ["geographical_domain", "explicit_references", "hierarchies"]:
                options = sorted(tags_by_field.get(field, []))
                query[field] = st.sidebar.multiselect(
                    field.replace("_", " ").title(),
                    options,
                    key=f"tag_{field}",
                )

    st.sidebar.divider()
    st.sidebar.subheader("Parameters")
    alpha = st.sidebar.slider(
        "alpha (exact vs semantic)", 0.0, 1.0, 0.5, 0.05,
        help="Higher values weight exact tag matches more; lower values weight "
             "semantic similarity more. 0.5 values an exact match and a perfect "
             "semantic match equally.",
    )
    top_k = st.sidebar.slider("Top-k anchors", 1, 5, 3)
    max_hops = st.sidebar.slider("Max hops", 1, 2, 1)
    min_edge = st.sidebar.slider(
        "Minimum edge strength (for tag expansion)", 0.5, 5.0, 2.0, 0.5,
    )

    return query, label, {
        "alpha": alpha,
        "top_k": top_k,
        "max_hops": max_hops,
        "min_edge_strength": min_edge,
    }, question


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def render_query_atoms(atoms: list[dict]) -> None:
    with st.expander(f"How the question was decomposed ({len(atoms)} atoms)", expanded=True):
        for atom in atoms:
            tags = "  ·  ".join(
                f"**{f}** = {', '.join(atom[f])}" for f in LIST_FIELDS if atom[f]
            )
            st.markdown(f"`{atom['atom_id']}` *{atom['condition_type']}* — {atom['text']}")
            st.caption(tags or "no tags")


def render_article_ranking(ranking: pd.DataFrame, df: pd.DataFrame,
                           multi_atom: bool, top_n: int = 10) -> None:
    st.subheader("Article ranking")
    if multi_atom:
        st.caption(
            "Each query atom scores every article by its best-matching atom; an "
            "article's score is the sum over query atoms, so articles that "
            "answer several parts of the question rank higher."
        )
    else:
        st.caption(
            "Each article scores as its best-matching atom. Scores run from 0 "
            "(nothing in common) to 1 (identical tags)."
        )
    texts = df.set_index("atom_id")["text"]
    rows = []
    for r in ranking.head(top_n).itertuples(index=False):
        if multi_atom:
            best = ", ".join(sorted(set(r.best_atoms.values()))) or "—"
            rows.append({"article": r.article_id, "score": r.score,
                         "query atoms matched": r.n_query_atoms_matched,
                         "best-matching atoms": best})
        else:
            rows.append({"article": r.article_id, "score": r.article_score,
                         "best-matching atom": r.best_atom,
                         "text": texts.get(r.best_atom, "")})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_anchors(result: dict, df: pd.DataFrame) -> None:
    st.subheader("Anchor matches")
    st.caption(
        "Top-k atoms returned by the classifier before graph expansion. "
        "Scores run from 0 (nothing in common) to 1 (identical tags)."
    )
    primary = result["primary_matches"]
    if primary.empty:
        st.info("No anchor matches — try loosening the query.")
        return
    for rank, (_, row) in enumerate(primary.iterrows(), start=1):
        atom_row = df[df["atom_id"] == row["atom_id"]].iloc[0]
        with st.container(border=True):
            st.markdown(
                f"**{row['atom_id']}**  ·  article `{row['article_id']}`  "
                f"·  score `{row['combined_score']:.2f}`  ·  rank {rank}"
            )
            st.write(atom_row["text"])
            st.caption(
                f"exact = {row['exact_score']:.2f}  ·  "
                f"semantic = {row['semantic_score']:.2f}"
            )


def render_chain(result: dict) -> None:
    st.subheader("Reasoning chain")
    st.caption(
        "Graph expansion from each anchor. Every step names its edge, so the "
        "trace is auditable."
    )
    chain = result["chain"]
    if not chain:
        st.info("No chain produced.")
        return
    max_hop = max(c["hop"] for c in chain)
    for hop in range(max_hop + 1):
        hop_items = [c for c in chain if c["hop"] == hop]
        if not hop_items:
            continue
        label = "Hop 0 — anchor" if hop == 0 else f"Hop {hop} — graph expansion"
        st.markdown(f"**{label}**")
        for c in hop_items:
            with st.container(border=True):
                header = f"`{c['atom_id']}`  ·  article `{c['article_id']}`  ·  edge: `{c['edge_kind']}`"
                if c.get("score") is not None:
                    header += f"  ·  score `{c['score']:.2f}`"
                st.markdown(header)
                st.write(c["text"])
                st.caption(f"via: {c['via']}")


def render_subgraph_table(result: dict, ref_edges: pd.DataFrame,
                          edges_weighted: pd.DataFrame) -> None:
    st.subheader("Articles surfaced")
    articles = result["articles_touched"]
    st.write(f"{len(articles)} articles: " + ", ".join(f"`{a}`" for a in articles))

    # Build the subgraph edge list restricted to the surfaced articles
    atom_ids_surfaced = {c["atom_id"] for c in result["chain"]}

    ref_sub = ref_edges[
        ref_edges["source"].isin(atom_ids_surfaced) &
        ref_edges["target"].isin(articles)
    ] if len(ref_edges) else pd.DataFrame()

    tag_sub = edges_weighted[
        edges_weighted["atom_a"].isin(atom_ids_surfaced) &
        edges_weighted["atom_b"].isin(atom_ids_surfaced)
    ] if len(edges_weighted) else pd.DataFrame()

    tab_ref, tab_tag = st.tabs(["Reference edges", "Shared-tag edges"])
    with tab_ref:
        if len(ref_sub):
            st.dataframe(
                ref_sub[["source", "target", "ref_text"]].reset_index(drop=True),
                width="stretch",
            )
        else:
            st.caption("No cross-article references among the surfaced atoms.")
    with tab_tag:
        if len(tag_sub):
            cols = ["atom_a", "atom_b", "field", "shared", "weighted_strength"]
            st.dataframe(
                tag_sub[cols].sort_values("weighted_strength", ascending=False)
                             .reset_index(drop=True),
                width="stretch",
            )
        else:
            st.caption("No shared-tag edges among the surfaced atoms.")


def render_subgraph_plot(result: dict, ref_edges: pd.DataFrame,
                         edges_weighted: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    import networkx as nx

    articles = result["articles_touched"]
    if len(articles) < 2:
        st.caption("Need at least two surfaced articles to draw a graph.")
        return

    anchor_articles = set(result["primary_matches"]["article_id"].tolist())

    # Build article-level graph
    G = nx.DiGraph()
    for a in articles:
        G.add_node(a, is_anchor=(a in anchor_articles))

    # Reference edges: aggregate atom→article citations into article→article
    ref_pairs: dict[tuple[str, str], list[str]] = {}
    if len(ref_edges):
        atom_ids_surfaced = {c["atom_id"] for c in result["chain"]}
        rs = ref_edges[
            ref_edges["source"].isin(atom_ids_surfaced) &
            ref_edges["target"].isin(articles)
        ]
        for _, r in rs.iterrows():
            src_article = r["source"].split("(")[0]
            key = (src_article, r["target"])
            ref_pairs.setdefault(key, []).append(r["ref_text"])

    # Shared-tag edges: aggregate atom-pairs into article-pairs, sum strength
    tag_pairs: dict[tuple[str, str], float] = {}
    if len(edges_weighted):
        atom_ids_surfaced = {c["atom_id"] for c in result["chain"]}
        ts = edges_weighted[
            edges_weighted["atom_a"].isin(atom_ids_surfaced) &
            edges_weighted["atom_b"].isin(atom_ids_surfaced)
        ]
        for _, r in ts.iterrows():
            key = tuple(sorted([r["article_a"], r["article_b"]]))
            tag_pairs[key] = tag_pairs.get(key, 0.0) + float(r["weighted_strength"])

    # Layout
    for (u, v) in ref_pairs:
        if u in G and v in G:
            G.add_edge(u, v, kind="reference")
    for (u, v) in tag_pairs:
        if u == v or (u, v) in ref_pairs or (v, u) in ref_pairs:
            continue
        if u in G and v in G:
            G.add_edge(u, v, kind="shared")

    if G.number_of_edges() == 0:
        st.caption("No edges among surfaced articles at the current thresholds.")
        return

    pos = nx.spring_layout(G, seed=42, k=1.4 / max(1, len(articles) ** 0.5))

    # Edge traces (two, one per kind, so we can color them)
    ref_x, ref_y = [], []
    shared_x, shared_y = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if data["kind"] == "reference":
            ref_x.extend([x0, x1, None])
            ref_y.extend([y0, y1, None])
        else:
            shared_x.extend([x0, x1, None])
            shared_y.extend([y0, y1, None])

    ref_trace = go.Scatter(
        x=ref_x, y=ref_y, mode="lines",
        line=dict(color="#c0392b", width=2.5),
        hoverinfo="skip", name="reference",
    )
    shared_trace = go.Scatter(
        x=shared_x, y=shared_y, mode="lines",
        line=dict(color="#7f8c8d", width=1.5, dash="dot"),
        hoverinfo="skip", name="shared tags",
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        is_anchor = G.nodes[n]["is_anchor"]
        node_color.append("#2980b9" if is_anchor else "#95a5a6")
        node_size.append(38 if is_anchor else 28)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="middle center",
        textfont=dict(color="white", size=11),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(color="white", width=1.5),
        ),
        hoverinfo="text",
        name="articles",
    )

    fig = go.Figure(
        data=[shared_trace, ref_trace, node_trace],
        layout=go.Layout(
            showlegend=True,
            hovermode="closest",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
            legend=dict(orientation="h", y=-0.05),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Blue = anchor articles (from the classifier). "
        "Grey = surfaced via graph expansion. "
        "Red edges = explicit references. Dotted edges = shared tags."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Article Relationship Model")
    pipe = load_pipeline()
    df = pipe["df"]
    st.caption(
        "Atom-level knowledge graph over Dutch statutory law "
        f"({df['article_id'].nunique()} articles, {len(df)} atoms). "
        "Query in the sidebar, results below."
    )

    query, label, params, question = build_sidebar_query(pipe["tags_by_field"])
    run = st.sidebar.button("Run", type="primary", width="stretch")

    st.markdown(f"### Query: *{label}*")

    query_atoms: list[dict] | None = None
    if question is not None:
        if not question:
            st.warning("Type a question in the sidebar.")
            return
        if not run:
            st.info("Click Run to decompose the question and search.")
            return
        try:
            with st.spinner("Decomposing the question into atoms…"):
                query_atoms = decompose_question(question, pipe["tags_by_field"])
        except Exception as e:  # missing key, network, refusal: show, don't crash
            st.error(
                f"Could not decompose the question ({type(e).__name__}: {e}). "
                "Check that ANTHROPIC_API_KEY is set, or use a preset or the "
                "tag builder instead."
            )
            return
        if not query_atoms:
            st.warning("The question produced no atoms. Try rephrasing it.")
            return
        render_query_atoms(query_atoms)
        query = _merge_query_atoms(query_atoms)

    # Show the tags the query will actually use
    active_tags = {
        f: query[f] for f in LIST_FIELDS
        if query.get(f)
    }
    if active_tags:
        st.markdown(
            "Tags: " + "  ·  ".join(
                f"**{f}** = {v}" for f, v in active_tags.items()
            )
        )
    else:
        st.warning("No tags selected. Pick a preset or add tags in the sidebar.")
        return

    if not run:
        st.info("Configure the query in the sidebar and click Run.")
        return

    # Article ranking: the quantity the evaluation measures
    if query_atoms is not None:
        ranking = rank_articles_for_query_atoms(
            query_atoms, df, weights=pipe["weights_idf"],
            sim_pairs=pipe["sim_pairs"], alpha=params["alpha"],
        )
    else:
        ranking = classify_atom(
            query, df, weights=pipe["weights_idf"], sem_lookup=pipe["sem_lookup"],
            alpha=params["alpha"],
        )["article_ranking"]
    render_article_ranking(ranking, df, multi_atom=query_atoms is not None)

    st.divider()

    with st.spinner("Walking the graph…"):
        result = chained_reasoning(
            query, df,
            pipe["ref_edges"], pipe["edges_weighted"],
            weights=pipe["weights_idf"],
            sim_pairs=pipe["sim_pairs"],
            alpha=params["alpha"],
            top_k=params["top_k"],
            max_hops=params["max_hops"],
            min_edge_strength=params["min_edge_strength"],
            max_expansions_per_atom=4,
        )

    # Top row: anchors and chain side by side
    left, right = st.columns(2)
    with left:
        render_anchors(result, df)
    with right:
        render_chain(result)

    st.divider()

    # Bottom: subgraph — plot first, tables in an expander
    render_subgraph_plot(result, pipe["ref_edges"], pipe["edges_weighted"])
    with st.expander("Subgraph tables (reference edges, shared-tag edges)", expanded=False):
        render_subgraph_table(result, pipe["ref_edges"], pipe["edges_weighted"])


if __name__ == "__main__":
    main()
