"""Statute Navigator: the Streamlit UI for the Article Relationship Model.

Run with (from the repo root):

    streamlit run app.py

A search page: describe a legal problem in plain Dutch or English, and the
app decomposes it into atoms (LLM), ranks the statutory articles, walks the
atom graph from the best matches, and shows the evidence. Preset and
tag-based queries and the scoring parameters live in the sidebar.
"""

from __future__ import annotations

import hmac
import html
import os
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
    add_taxonomy_columns, build_reference_edges, parse_article_id,
)
from src.data.query import (  # noqa: E402
    embed_vocabulary, extend_semantic_lookup, rank_articles_for_query_atoms,
    text_to_query_atoms,
)
from src.data.reasoning import chained_reasoning  # noqa: E402


# ---------------------------------------------------------------------------
# Page config and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Statute Navigator",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Colours from the validated reference palette: slot 1 (blue) marks the
# query's direct matches, slot 2 (orange) marks explicit references.
ACCENT = "#2a78d6"
REFERENCE = "#eb6834"
NEUTRAL = "#a8a6a0"

st.markdown(
    f"""
    <style>
    .block-container {{ max-width: 1150px; padding-top: 2.5rem; }}
    footer {{ visibility: hidden; }}

    .hero-title {{ text-align: center; font-size: 2.6rem; font-weight: 700;
                   letter-spacing: -0.02em; margin: 4vh 0 0.2rem; }}
    .hero-sub   {{ text-align: center; opacity: 0.65; margin-bottom: 1.6rem; }}

    .st-key-q_input textarea {{ font-size: 1.08rem; line-height: 1.5; }}
    .st-key-q_input [data-testid="stChatInput"] {{ border-radius: 1.4rem; }}
    .try-label {{ text-align: center; opacity: 0.6; font-size: 0.8rem; margin: 1.1rem 0 0.3rem; }}
    .st-key-examples button, .st-key-presets button {{
        border-radius: 999px; font-size: 0.85rem; padding: 0.15rem 0.9rem;
    }}
    .query-echo {{ font-size: 1.05rem; margin-top: 0.4rem; }}

    .section {{ font-size: 1.15rem; font-weight: 600; }}
    .subsection {{ font-size: 0.98rem; font-weight: 600; }}
    .why-part {{ margin: 0.3rem 0 0.15rem; }}
    .why-line {{ font-size: 0.85rem; margin-left: 0.6rem; }}
    div[class*="st-key-sec_"] {{ margin-top: 1.6rem; }}
    .section-sub {{ opacity: 0.6; font-size: 0.85rem; margin-bottom: 0.8rem; }}

    .result {{ padding: 0.35rem 0 1.15rem; }}
    .crumb  {{ font-size: 0.8rem; opacity: 0.6; }}
    .result-title {{ font-size: 1.2rem; font-weight: 600; color: {ACCENT}; }}
    .result-meta  {{ font-size: 0.8rem; opacity: 0.6; margin-left: 0.4rem; }}
    .snippet {{ font-size: 0.95rem; line-height: 1.45; margin-top: 0.15rem; }}
    .scorebar {{ height: 4px; border-radius: 2px; background: rgba(128,128,128,0.18);
                 margin-top: 0.45rem; width: 220px; }}
    .scorebar > div {{ height: 4px; border-radius: 2px; background: {ACCENT}; }}

    .atom-row {{ margin: 0.25rem 0 0.7rem; }}
    .atom-text {{ font-size: 0.95rem; }}
    .chip {{ display: inline-block; padding: 1px 10px; margin: 3px 4px 0 0;
             border-radius: 999px; border: 1px solid rgba(128,128,128,0.35);
             font-size: 0.78rem; }}
    .chip b {{ font-weight: 500; opacity: 0.6; margin-right: 4px; }}

    .step {{ border-left: 2px solid rgba(128,128,128,0.25); padding: 0.1rem 0 0.6rem 0.9rem; }}
    .step.anchor {{ border-left-color: {ACCENT}; }}
    .step.reference {{ border-left-color: {REFERENCE}; }}
    .step-head {{ font-size: 0.85rem; }}
    .step-via  {{ font-size: 0.78rem; opacity: 0.6; }}
    </style>
    """,
    unsafe_allow_html=True,
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
    # EMBEDDING_MODEL lets a memory-constrained host swap in a smaller model
    # (e.g. sentence-transformers/all-MiniLM-L6-v2) without code changes.
    model = SentenceTransformer(
        _secret("EMBEDDING_MODEL") or "sentence-transformers/all-mpnet-base-v2"
    )
    sim_pairs = compute_tag_similarities(df, threshold=0.55, model=model)

    tags_by_field = unique_tags_by_field(df)
    return {
        "df": df,
        "df_tax": df_tax,
        "model": model,
        "vocab_embeddings": embed_vocabulary(tags_by_field, model),
        "weights_idf": weights_idf,
        "edges_weighted": edges_weighted,
        "sim_pairs": sim_pairs,
        "sem_lookup": build_semantic_lookup(sim_pairs),
        "ref_edges": ref_edges,
        "tags_by_field": tags_by_field,
        "taxonomy": {a: parse_article_id(a) for a in df["article_id"].unique()},
    }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "Patient asks to destroy their file": {
        "actors": ["patient", "care provider"],
        "acts": ["destroys"],
        "residual": ["file", "request"],
        "temporal": ["after request"],
    },
    "Poor record-keeping caused harm": {
        "actors": ["care provider", "patient"],
        "acts": ["compensates"],
        "legal_relations": ["obligation"],
        "residual": ["damage", "failure to perform", "file"],
    },
    "Hospital sued for a doctor's error": {
        "actors": ["employer", "subordinate", "third party"],
        "legal_relations": ["service relationship"],
        "residual": ["damage", "fault", "liability"],
    },
}

EXAMPLES: dict[str, str] = {
    "Risks never explained": (
        "De chirurg heeft mij voor de operatie nooit verteld wat de risico's waren "
        "of welke andere behandelingen er waren. Nu heb ik blijvende zenuwschade. "
        "Kan ik een schadevergoeding eisen?"
    ),
    "Doctor told my employer": (
        "Mijn huisarts heeft zonder mijn toestemming informatie over mijn diagnose "
        "aan mijn werkgever gegeven. Mag dat?"
    ),
    "Father died after an error": (
        "Mijn vader is overleden door een fout van de anesthesist. Kunnen wij als "
        "nabestaanden de begrafeniskosten en het verlies van zijn inkomen vergoed krijgen?"
    ),
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


def _secret(name: str) -> str | None:
    """Read a setting from the environment, else from .streamlit/secrets.toml.

    Hosting platforms (e.g. Hugging Face Spaces) pass secrets as environment
    variables; locally they live in the git-ignored secrets file.
    """
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file
        return None


def _load_api_key() -> None:
    """Make the API key available to the Anthropic client."""
    key = _secret("ANTHROPIC_API_KEY")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key


def _check_password() -> bool:
    """Ask for APP_PASSWORD before showing the app. No password set = open app."""
    expected = _secret("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return True

    st.markdown('<div class="hero-title">⚖ Statute Navigator</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">This demo is password protected.</div>',
                unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 3, 2])
    with mid:
        with st.form("login", border=False):
            entered = st.text_input("Password", type="password")
            if st.form_submit_button("Enter", type="primary", width="stretch"):
                if hmac.compare_digest(entered.encode(), expected.encode()):
                    st.session_state["authenticated"] = True
                    st.rerun()
                st.error("Wrong password.")
    return False


@st.cache_data(show_spinner=False)
def decompose_question(question: str, vocabulary: dict[str, list[str]]) -> list[dict]:
    """LLM call, cached so re-running the same question costs nothing."""
    return text_to_query_atoms(question, vocabulary=vocabulary)


def _submit_question(text: str) -> None:
    text = (text or "").strip()
    if text:
        st.session_state["submitted"] = {"kind": "question", "text": text}


def _on_search() -> None:
    """Chat-input callback: the submitted text is in session state under its key."""
    _submit_question(st.session_state.get("q_input"))


def _submit_tags(label: str, query: dict) -> None:
    st.session_state["submitted"] = {"kind": "tags", "label": label, "query": query}


def _use_preset(label: str) -> None:
    query = _query_template()
    for field, values in PRESETS[label].items():
        query[field] = list(values)
    _submit_tags(label, query)


# ---------------------------------------------------------------------------
# Sidebar: settings and tag-based queries
# ---------------------------------------------------------------------------

def render_sidebar(tags_by_field: dict[str, list[str]]) -> dict:
    st.sidebar.header("Search settings")
    params = {
        "alpha": st.sidebar.slider(
            "Exact vs semantic (alpha)", 0.0, 1.0, 0.5, 0.05,
            help="Higher values weight exact tag matches more; lower values weight "
                 "semantic similarity more. 0.5 values an exact match and a perfect "
                 "semantic match equally.",
        ),
        "top_k": st.sidebar.slider("Starting atoms for the graph walk", 1, 5, 3),
        "max_hops": st.sidebar.slider("Graph hops", 1, 2, 1),
        "min_edge_strength": st.sidebar.slider(
            "Minimum shared-tag strength", 0.5, 5.0, 2.0, 0.5,
            help="How strongly two atoms must share tags before the walker follows the link.",
        ),
        "n_results": st.sidebar.slider("Results shown", 3, 10, 5),
    }

    st.sidebar.divider()
    st.sidebar.header("Tag builder (advanced)")
    st.sidebar.caption("Skip the LLM and query with tags from the corpus vocabulary.")

    query = _query_template()
    label = "Custom tag query"
    for field in ["actors", "acts", "legal_relations", "residual", "temporal",
                  "geographical_domain", "explicit_references", "hierarchies"]:
        query[field] = st.sidebar.multiselect(
            field.replace("_", " ").capitalize(),
            sorted(tags_by_field.get(field, [])),
            key=f"tag_{field}",
        )
    st.sidebar.button(
        "Run tag query", width="stretch",
        on_click=_submit_tags, args=(label, query),
        disabled=not any(query[f] for f in LIST_FIELDS),
    )
    return params


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return html.escape(str(text))


def _crumb(taxonomy: dict) -> str:
    parts = [taxonomy.get("wetboek"), taxonomy.get("boek"),
             taxonomy.get("afdeling_name") or taxonomy.get("titel_name")]
    return " › ".join(_esc(p) for p in parts if p)


def _short(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _chips(atom: dict) -> str:
    return "".join(
        f'<span class="chip"><b>{_esc(f.replace("_", " "))}</b>{_esc(t)}</span>'
        for f in LIST_FIELDS for t in atom.get(f, [])
    )


INFO = {
    "decomposition": (
        "**How your question was split up**\n\n"
        "An AI model broke your question into *atoms*: the smallest pieces that each "
        "say one legal thing. A part marked *fact / requirement* describes what "
        "happened; a part marked *consequence* describes what you want or what "
        "should follow.\n\n"
        "Each part gets labels (who acts, what they do, which terms matter) using the "
        "same scheme as the annotated statutes. That is what lets the system compare "
        "your question to the law label by label."
    ),
    "results_question": (
        "**What the match score means**\n\n"
        "Every statement in the law carries labels: who acts, what they do, and the "
        "key terms. Your question was split into {n} part(s), and each part got labels "
        "the same way. The match score measures how much those labels overlap:\n\n"
        "- A label both sides share adds to the score.\n"
        "- Rare labels count more than common ones. *Confidentiality* appears in few "
        "statements, so sharing it says a lot; *patient* appears almost everywhere, so "
        "it says little.\n"
        "- A label that is only *similar* (say *doctor* and *care provider*) counts "
        "partly, depending on how similar the two are.\n"
        "- Labels on either side that find no match pull the score down.\n\n"
        "For each part, an article is scored by its single best-matching statement, "
        "from **0** (nothing in common) to **1** (exactly the same labels). The number "
        "shown is the **average over all {n} part(s)**, so you can compare it between "
        "questions. *Covers 2 of 3 parts* counts the parts the article matches at "
        "all.\n\n"
        "There is no pass mark. A score tells you how much the labels overlap, not "
        "whether the article legally applies, so compare results with each other and "
        "read the article itself. Open **Why this score?** under a result to see "
        "exactly which labels matched."
    ),
    "results_tags": (
        "**What the match score means**\n\n"
        "Every statement in the law carries labels: who acts, what they do, and the "
        "key terms. The match score measures how much your tags overlap with the "
        "labels of an article's best-matching statement:\n\n"
        "- A tag the statement also has adds to the score.\n"
        "- Rare tags count more than common ones. *Patient* appears almost "
        "everywhere, so sharing it says little; a specific act or term says a lot.\n"
        "- A tag that is only *similar* to one of the statement's labels counts "
        "partly, depending on how similar the two are.\n"
        "- Tags or labels that find no match pull the score down.\n\n"
        "The score runs from **0** (nothing in common) to **1** (exactly the same "
        "labels). There is no pass mark: it measures overlap, not whether the article "
        "legally applies. Open **Why this score?** under a result to see exactly "
        "which labels matched."
    ),
    "chart": (
        "**Reading the chart**\n\n"
        "Each bar is one article's match score, the same number as in the result "
        "list. The blue bar is the best match.\n\n"
        "The axis always runs from 0 to 1, so you can compare charts from different "
        "searches. Hover over a bar to see the text that matched. Articles that "
        "matched nothing are left out."
    ),
    "graph": (
        "**Two views of the same articles**\n\n"
        "Both graphs show the articles the search reached and how they are linked. "
        "The *focused view* puts the direct matches in the centre and sizes each "
        "article by its match score. The *network view* is the original layout, "
        "where linked articles pull each other together.\n\n"
        "The graphs show *why* articles are related, not whether they apply to your "
        "case. Hover over a circle for details."
    ),
    "graph_focused": (
        "**Reading the focused view**\n\n"
        "**Blue** circles matched your question directly and sit in the centre; "
        "**grey** circles were reached by following links from them. A bigger circle "
        "means a higher match score.\n\n"
        "An **orange arrow** means one article explicitly refers to the other in its "
        "text, for example *without prejudice to article 448*. The arrow points to "
        "the article being cited.\n\n"
        "A **dotted line** means the two articles share concepts: the same actors, "
        "acts or terms. A thicker line means more, and more specific, shared concepts."
    ),
    "graph_network": (
        "**Reading the network view**\n\n"
        "This is the original graph. Articles are placed so that linked articles pull "
        "towards each other, so tightly connected groups end up close together.\n\n"
        "**Blue** circles are the articles that matched your question directly "
        "(the starting points); **grey** ones were reached through the graph. A "
        "**red line** is an explicit reference between two articles; a **dotted "
        "line** means they share concepts. Circle size only marks starting points "
        "versus the rest; it does not show the score."
    ),
    "trail": (
        "**Reading the trail**\n\n"
        "**Starting points** are the single legal statements (atoms) that matched "
        "your question best. Their number is the match score, from 0 to 1.\n\n"
        "**Linked atoms** were reached in one step from a starting point. The grey "
        "line under each says how:\n\n"
        "- *references* means one statement explicitly cites the other article;\n"
        "- *shares [...]* lists the concepts both statements contain.\n\n"
        "The number next to a shared-concept link is its **strength**, not a 0-1 "
        "score. Each shared concept adds points, and rarer concepts add more, so "
        "higher means more and more specific overlap. Links weaker than the "
        "*Minimum shared-tag strength* setting in the sidebar are not followed."
    ),
}


def section(title: str, subtitle: str = "", info: str | None = None,
            level: str = "section") -> None:
    """Section heading, with an (i) button that explains it when ``info`` is given.

    ``level="subsection"`` gives a smaller heading for a part of a section.
    """
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key=f"sec_{abs(hash(title))}"):
        st.markdown(f'<div class="{level}">{_esc(title)}</div>', unsafe_allow_html=True)
        if info:
            with st.popover("", icon=":material/info:", type="tertiary",
                            help="What do these numbers mean?"):
                st.markdown(info)
    if subtitle:
        st.markdown(f'<div class="section-sub">{_esc(subtitle)}</div>', unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown('<div class="hero-title">⚖ Statute Navigator</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Describe a legal problem in English or Dutch and find the '
        "Dutch statutory articles that apply. WGBO and Book 6 BW.</div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        # Grows with the text; Enter searches, Shift+Enter adds a line.
        st.chat_input(
            "Describe your legal problem…  (Enter to search, Shift+Enter for a new line)",
            key="q_input", on_submit=_on_search,
        )
        st.markdown('<div class="try-label">Example questions</div>', unsafe_allow_html=True)
        with st.container(key="examples", horizontal=True, horizontal_alignment="center"):
            for label, example in EXAMPLES.items():
                st.button(label, key=f"ex_{label}", width="content",
                          on_click=_submit_question, args=(example,))
        st.markdown('<div class="try-label">Preset tag queries · no LLM needed</div>',
                    unsafe_allow_html=True)
        with st.container(key="presets", horizontal=True, horizontal_alignment="center"):
            for label in PRESETS:
                st.button(label, key=f"pre_{label}", width="content",
                          on_click=_use_preset, args=(label,))


def render_decomposition(atoms: list[dict]) -> None:
    section("How your question was read",
            f"Split into {len(atoms)} atoms and labelled with the same schema as the statutes.",
            info=INFO["decomposition"])
    for atom in atoms:
        kind = "fact / requirement" if atom["condition_type"] == "pre_condition" else "consequence"
        st.markdown(
            f'<div class="atom-row"><span class="atom-text">{_esc(atom["text"])}</span> '
            f'<span class="result-meta">{kind}</span><br>{_chips(atom)}</div>',
            unsafe_allow_html=True,
        )


def _rarity(weight: float) -> str:
    if weight >= 2.5:
        return "rare label, counts heavily"
    if weight >= 1.2:
        return "fairly common label"
    return "very common label, counts little"


def _explain_match(part: dict, atom_id: str, pipe: dict, alpha: float) -> pd.Series:
    """Score one query part against one statement, keeping the match details."""
    candidate = pipe["df"][pipe["df"]["atom_id"] == atom_id]
    return classify_atom(
        part, candidate, weights=pipe["weights_idf"], sem_lookup=pipe["sem_lookup"],
        alpha=alpha,
    )["scores_by_atom"].iloc[0]


def render_why(parts: list[dict], best_atoms: dict[str, str], pipe: dict, alpha: float,
               multi_atom: bool) -> None:
    """Explain a result's score: which labels matched, how, and which did not."""
    df = pipe["df"].set_index("atom_id")
    lines = []
    for i, part in enumerate(parts, start=1):
        title = (f"Part {i}: “{_esc(_short(part.get('text', ''), 90))}”" if multi_atom
                 else "Your tags")
        atom_id = best_atoms.get(part["atom_id"])
        if not atom_id:
            lines.append(f'<div class="why-part"><b>{title}</b></div>'
                         '<div class="why-line">Not matched by any statement in this article.</div>')
            continue
        row = _explain_match(part, atom_id, pipe, alpha)
        lines.append(
            f'<div class="why-part"><b>{title}</b> → best statement '
            f'<b>{_esc(atom_id)}</b>, match {row["combined_score"]:.2f}</div>'
            f'<div class="why-line" style="opacity:0.7">“{_esc(_short(df.at[atom_id, "text"], 140))}”</div>'
        )
        matched_q = set()
        for field, tag, weight in row["exact_matches"]:
            matched_q.add((field, tag))
            lines.append(f'<div class="why-line">✓ Same label · {_esc(field.replace("_", " "))}: '
                         f'<i>{_esc(tag)}</i> ({_rarity(weight)})</div>')
        for field, q_tag, c_tag, sim in row["semantic_matches"]:
            matched_q.add((field, q_tag))
            lines.append(f'<div class="why-line">≈ Similar label · {_esc(field.replace("_", " "))}: '
                         f'your <i>{_esc(q_tag)}</i> ~ the article\'s <i>{_esc(c_tag)}</i> '
                         f'({sim:.0%} similar, counts partly)</div>')
        missing = [(f, t) for f in LIST_FIELDS for t in part.get(f, []) if (f, t) not in matched_q]
        for field, tag in missing:
            lines.append(f'<div class="why-line">✗ No match · {_esc(field.replace("_", " "))}: '
                         f'<i>{_esc(tag)}</i> (lowers the score)</div>')
        own = sum(len(df.at[atom_id, f]) for f in LIST_FIELDS)
        extra = own - len(row["exact_matches"])
        if extra > 0:
            lines.append(f'<div class="why-line" style="opacity:0.7">The statement has {extra} '
                         "other label(s) your question did not mention, which also lowers "
                         "the score a little.</div>")
    if multi_atom:
        lines.append('<div class="why-line" style="margin-top:0.5rem;opacity:0.7">The article\'s '
                     f"match is the average of these {len(parts)} part score(s); unmatched "
                     "parts count as 0.</div>")
    st.markdown("".join(lines), unsafe_allow_html=True)


def render_results(ranking: pd.DataFrame, pipe: dict, multi_atom: bool, n: int,
                   parts: list[dict], alpha: float) -> None:
    n_parts = len(parts)
    texts = pipe["df"].set_index("atom_id")["text"]
    score_col = "coverage" if multi_atom else "article_score"
    top = ranking.head(n)
    max_score = max(float(ranking[score_col].max()), 1e-9)

    left, right = st.columns([3, 2], gap="large")
    with left:
        info = (INFO["results_question"].format(n=n_parts) if multi_atom
                else INFO["results_tags"])
        section("Best matching articles",
                f"{len(ranking)} articles ranked. Showing the top {len(top)}.", info=info)
        for r in top.itertuples(index=False):
            score = getattr(r, score_col)
            if multi_atom:
                best = sorted(set(r.best_atoms.values()))
                snippet = " … ".join(texts.get(a, "") for a in best[:2])
                meta = f"covers {r.n_query_atoms_matched} of {n_parts} part(s)"
            else:
                best = [r.best_atom]
                snippet = texts.get(r.best_atom, "")
                meta = f"best match {r.best_atom}"
            st.markdown(
                f'<div class="result">'
                f'<div class="crumb">{_crumb(pipe["taxonomy"].get(r.article_id, {}))}</div>'
                f'<span class="result-title">Artikel {_esc(r.article_id)} BW</span>'
                f'<span class="result-meta">match {score:.2f} · {_esc(meta)}</span>'
                f'<div class="snippet">{_esc(_short(snippet, 230))}</div>'
                f'<div class="scorebar"><div style="width:{100 * score / max_score:.0f}%"></div></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            best_atoms = r.best_atoms if multi_atom else {parts[0]["atom_id"]: r.best_atom}
            with st.expander("Why this score?"):
                render_why(parts, best_atoms, pipe, alpha, multi_atom)
    with right:
        section("Score by article", "Top 10 articles with any match. The best match is highlighted.",
                info=INFO["chart"])
        render_score_chart(ranking[ranking[score_col] > 0].head(10), score_col, texts, multi_atom)


def render_score_chart(top: pd.DataFrame, score_col: str, texts: pd.Series,
                       multi_atom: bool) -> None:
    import plotly.graph_objects as go

    if top.empty:
        st.caption("No article matched the query.")
        return
    top = top.iloc[::-1]  # plotly draws bottom-up; best match on top
    hover = []
    for r in top.itertuples(index=False):
        atom = (sorted(r.best_atoms.values())[0] if r.best_atoms else "") if multi_atom else r.best_atom
        hover.append(_esc(texts.get(atom, ""))[:140])
    colors = [NEUTRAL] * (len(top) - 1) + [ACCENT]

    fig = go.Figure(go.Bar(
        x=top[score_col], y=top["article_id"], orientation="h",
        marker=dict(color=colors, cornerradius=4),
        customdata=hover,
        hovertemplate="<b>%{y}</b>  match %{x:.2f}<br>%{customdata}<extra></extra>",
    ))
    best = top.iloc[-1]
    fig.add_annotation(x=best[score_col], y=best["article_id"], text=f"{best[score_col]:.2f}",
                       showarrow=False, xanchor="left", xshift=6)
    fig.update_layout(
        height=380, bargap=0.45, margin=dict(l=0, r=30, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=None, showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                   zeroline=False, range=[0, max(1.0, float(top[score_col].max()) * 1.08)],
                   tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
        yaxis=dict(title=None, showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_graph(result: dict, ranking: pd.DataFrame, score_col: str, pipe: dict) -> None:
    import networkx as nx
    import plotly.graph_objects as go

    section("How the articles connect",
            "Two views of the articles the search reached and the links between them.",
            info=INFO["graph"])
    section("Focused view",
            "Blue: matched your question directly. Grey: reached by walking the graph. "
            "Orange arrows: one article explicitly cites the other. Dotted: shared concepts.",
            info=INFO["graph_focused"], level="subsection")

    articles = result["articles_touched"]
    if len(articles) < 2:
        st.caption("Only one article was reached, so there is no graph to draw.")
        return

    anchors = set(result["primary_matches"]["article_id"])
    surfaced = {c["atom_id"] for c in result["chain"]}
    scores = dict(zip(ranking["article_id"], ranking[score_col]))
    texts = pipe["df"].set_index("atom_id")["text"]
    best_atom = {c["article_id"]: c["atom_id"] for c in reversed(result["chain"])}

    ref_pairs: dict[tuple[str, str], list[str]] = {}
    ref = pipe["ref_edges"]
    if len(ref):
        for r in ref[ref["source"].isin(surfaced) & ref["target"].isin(articles)].itertuples():
            ref_pairs.setdefault((r.source.split("(")[0], r.target), []).append(r.ref_text)

    tag_pairs: dict[tuple[str, str], float] = {}
    tags = pipe["edges_weighted"]
    if len(tags):
        sub = tags[tags["atom_a"].isin(surfaced) & tags["atom_b"].isin(surfaced)]
        for r in sub.itertuples():
            key = tuple(sorted([r.article_a, r.article_b]))
            tag_pairs[key] = tag_pairs.get(key, 0.0) + float(r.weighted_strength)

    G = nx.Graph()
    G.add_nodes_from(articles)
    G.add_edges_from(k for k in ref_pairs if k[0] != k[1])
    G.add_edges_from(k for k in tag_pairs if k[0] != k[1])

    shells = [sorted(anchors & set(articles)), sorted(set(articles) - anchors)]
    shells = [s for s in shells if s]
    pos = nx.shell_layout(G, shells) if len(shells) > 1 else nx.spring_layout(G, seed=7)

    traces = []
    # Shared-tag edges: dotted, width by strength, drawn first so they sit underneath
    if tag_pairs:
        top_strength = max(tag_pairs.values())
        for (u, v), s in tag_pairs.items():
            if u == v or (u, v) in ref_pairs or (v, u) in ref_pairs:
                continue
            (x0, y0), (x1, y1) = pos[u], pos[v]
            traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color="rgba(128,128,128,0.45)", dash="dot",
                          width=1 + 2.5 * s / top_strength),
            ))

    max_score = max([scores.get(a, 0.0) for a in articles] + [1e-9])
    node_x, node_y, sizes, colors, hovers = [], [], [], [], []
    for a in articles:
        x, y = pos[a]
        node_x.append(x)
        node_y.append(y)
        sizes.append(22 + 26 * scores.get(a, 0.0) / max_score)
        colors.append(ACCENT if a in anchors else NEUTRAL)
        tax = pipe["taxonomy"].get(a, {})
        hovers.append(
            f"<b>Artikel {a}</b> · match {scores.get(a, 0.0):.2f}<br>"
            f"{_esc(tax.get('afdeling_name') or tax.get('titel_name') or '')}<br>"
            f"{_esc(texts.get(best_atom.get(a, ''), ''))[:120]}"
        )
    traces.append(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=articles,
        textposition="bottom center", textfont=dict(size=12),
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=2)),
        customdata=hovers, hovertemplate="%{customdata}<extra></extra>", showlegend=False,
    ))

    # Legend entries (dummy traces) so identity is never colour alone
    for name, kw in [
        ("Matched directly", dict(mode="markers", marker=dict(size=12, color=ACCENT))),
        ("Reached via the graph", dict(mode="markers", marker=dict(size=12, color=NEUTRAL))),
        ("Explicit reference", dict(mode="lines", line=dict(color=REFERENCE, width=2.5))),
        ("Shared concepts", dict(mode="lines", line=dict(color="rgba(128,128,128,0.6)", dash="dot"))),
    ]:
        traces.append(go.Scatter(x=[None], y=[None], name=name, **kw))

    fig = go.Figure(traces)
    # Reference edges as arrows (citing article -> cited article)
    for (u, v), refs in ref_pairs.items():
        if u == v:
            continue
        fig.add_annotation(
            x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5,
            arrowcolor=REFERENCE, standoff=16, startstandoff=14,
            hovertext=f"{u} cites {v}: " + "; ".join(sorted(set(refs))), text="",
        )
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


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


def render_chain(result: dict) -> None:
    section("Reasoning trail",
            "Where the walk through the graph started, and which link brought in each atom.",
            info=INFO["trail"])
    chain = result["chain"]
    if not chain:
        st.caption("No trail produced.")
        return
    def step_html(c: dict) -> str:
        kind = ("anchor" if c["edge_kind"] == "classifier"
                else "reference" if c["edge_kind"].startswith("reference") else "")
        score = f" · {c['score']:.2f}" if c.get("score") is not None else ""
        return (f'<div class="step {kind}">'
                f'<div class="step-head"><b>{_esc(c["atom_id"])}</b>{score}</div>'
                f'<div class="snippet">{_esc(_short(c["text"], 170))}</div>'
                f'<div class="step-via">{_esc(c["via"])}</div></div>')

    max_hop = max(c["hop"] for c in chain)
    cols = st.columns(max_hop + 1, gap="large")
    for hop, col in enumerate(cols):
        with col:
            st.markdown("**Starting points**" if hop == 0 else f"**Step {hop}: linked atoms**")
            steps = [c for c in chain if c["hop"] == hop]
            for c in steps[:6]:
                st.markdown(step_html(c), unsafe_allow_html=True)
            if len(steps) > 6:
                with st.expander(f"Show {len(steps) - 6} more"):
                    for c in steps[6:]:
                        st.markdown(step_html(c), unsafe_allow_html=True)


def render_tables(result: dict, pipe: dict) -> None:
    articles = result["articles_touched"]
    surfaced = {c["atom_id"] for c in result["chain"]}
    ref, tags = pipe["ref_edges"], pipe["edges_weighted"]
    with st.expander("Raw evidence tables"):
        tab_ref, tab_tag = st.tabs(["Reference edges", "Shared-tag edges"])
        with tab_ref:
            sub = ref[ref["source"].isin(surfaced) & ref["target"].isin(articles)] if len(ref) else ref
            if len(sub):
                st.dataframe(sub[["source", "target", "ref_text"]], width="stretch", hide_index=True)
            else:
                st.caption("No explicit references among the surfaced atoms.")
        with tab_tag:
            sub = tags[tags["atom_a"].isin(surfaced) & tags["atom_b"].isin(surfaced)] if len(tags) else tags
            if len(sub):
                st.dataframe(
                    sub[["atom_a", "atom_b", "field", "shared", "weighted_strength"]]
                    .sort_values("weighted_strength", ascending=False),
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("No shared-tag edges among the surfaced atoms.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_password():
        return
    pipe = load_pipeline()
    df = pipe["df"]
    params = render_sidebar(pipe["tags_by_field"])
    render_hero()

    submitted = st.session_state.get("submitted")
    if not submitted:
        st.markdown(
            f'<div class="try-label" style="margin-top:3rem">Searching {df["article_id"].nunique()} '
            f"articles · {len(df)} annotated atoms · settings and the tag builder in the sidebar</div>",
            unsafe_allow_html=True,
        )
        return

    query_atoms: list[dict] | None = None
    if submitted["kind"] == "question":
        _load_api_key()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "No Anthropic API key found. Put it in `.streamlit/secrets.toml` as "
                '`ANTHROPIC_API_KEY = "sk-ant-..."` and restart the app, or set the '
                "ANTHROPIC_API_KEY environment variable in the terminal that runs "
                "Streamlit. Tag queries in the sidebar work without a key."
            )
            return
        try:
            with st.spinner("Reading your question…"):
                query_atoms = decompose_question(submitted["text"], pipe["tags_by_field"])
        except Exception as e:  # missing key, network, refusal: show, don't crash
            st.error(
                f"Could not read the question ({type(e).__name__}: {e}). Check that "
                "ANTHROPIC_API_KEY is set, or use a tag query from the sidebar."
            )
            return
        if not query_atoms:
            st.warning("The question produced no atoms. Try rephrasing it.")
            return
        query = _merge_query_atoms(query_atoms)
        # Let tags the corpus has never seen (e.g. "doctor") match similar
        # corpus tags (e.g. "care provider"); used by every step below.
        pipe = {**pipe, "sem_lookup": extend_semantic_lookup(
            pipe["sem_lookup"], query_atoms, pipe["vocab_embeddings"], pipe["model"],
        )}
        ranking = rank_articles_for_query_atoms(
            query_atoms, df, weights=pipe["weights_idf"],
            sem_lookup=pipe["sem_lookup"], alpha=params["alpha"],
        )
        score_col = "coverage"
    else:
        query = submitted["query"]
        ranking = classify_atom(
            query, df, weights=pipe["weights_idf"], sem_lookup=pipe["sem_lookup"],
            alpha=params["alpha"],
        )["article_ranking"]
        score_col = "article_score"

    st.divider()
    if query_atoms is not None:
        st.markdown(f'<div class="query-echo">Results for “{_esc(submitted["text"])}”</div>',
                    unsafe_allow_html=True)
        render_decomposition(query_atoms)
    else:
        section(f"Tag query: {submitted['label']}")
        st.markdown(_chips(query), unsafe_allow_html=True)

    render_results(ranking, pipe, multi_atom=query_atoms is not None, n=params["n_results"],
                   parts=query_atoms if query_atoms else [query], alpha=params["alpha"])

    with st.spinner("Walking the graph…"):
        result = chained_reasoning(
            query, df, pipe["ref_edges"], pipe["edges_weighted"],
            weights=pipe["weights_idf"], sim_pairs=pipe["sim_pairs"],
            alpha=params["alpha"], top_k=params["top_k"], max_hops=params["max_hops"],
            min_edge_strength=params["min_edge_strength"], max_expansions_per_atom=4,
            sem_lookup=pipe["sem_lookup"],
        )
    render_graph(result, ranking, score_col, pipe)
    section("Network view",
            "The original layout. Blue: starting articles. Grey: reached via the graph. "
            "Red lines: explicit references. Dotted: shared concepts.",
            info=INFO["graph_network"], level="subsection")
    render_subgraph_plot(result, pipe["ref_edges"], pipe["edges_weighted"])
    render_chain(result)
    render_tables(result, pipe)


if __name__ == "__main__":
    main()
