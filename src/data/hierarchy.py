"""Model of the Dutch legal tree, plus article-ID parsing.

The Dutch civil code (Burgerlijk Wetboek, BW) is organised as:

    Rechtsgebied  (private_law / public_law)
        Wetboek       (BW, Sr, Sv, ...)
            Boek          (1-10 for BW)
                Titel         (e.g. 7.7 - Opdracht)
                    Afdeling      (e.g. 7.7.5 - WGBO)
                        Artikel       (e.g. 7:454)
                            Lid           (e.g. 7:454(1))
                                Atom          (e.g. 7:454(1).a)

This module exposes:

- ``LEGAL_TREE``: nested dict describing the hierarchy
- ``parse_article_id(article_id)``: single lookup returning the taxonomy path
- ``add_taxonomy_columns(df)``: enrich an atom DataFrame with taxonomy columns
- ``tree_summary(df)``: count atoms at each level, for reporting

Book 7 is fully populated because that is where the WGBO corpus lives. Books
1-10 have names but no title-level breakdown. Extend as new corpora are
annotated.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Tree definition
# ---------------------------------------------------------------------------

LEGAL_TREE: dict[str, Any] = {
    "private_law": {
        "name": "Privaatrecht",
        "wetboeken": {
            "BW": {
                "name": "Burgerlijk Wetboek",
                "boeken": {
                    1: {"name": "Personen- en familierecht"},
                    2: {"name": "Rechtspersonen"},
                    3: {"name": "Vermogensrecht in het algemeen"},
                    4: {"name": "Erfrecht"},
                    5: {"name": "Zakelijke rechten"},
                    6: {
                        "name": "Algemeen gedeelte van het verbintenissenrecht",
                        "titels": {
                            "6.1": {
                                "name": "Verbintenissen in het algemeen",
                                "articles": (1, 161),
                                "afdelingen": {
                                    "6.1.9": {
                                        "name": "Gevolgen van het niet-nakomen van een verbintenis",
                                        "articles": (74, 94),
                                    },
                                    "6.1.10": {
                                        "name": "Wettelijke verplichtingen tot schadevergoeding",
                                        "articles": (95, 110),
                                    },
                                },
                            },
                            "6.3": {
                                "name": "Onrechtmatige daad",
                                "articles": (162, 197),
                                "afdelingen": {
                                    "6.3.1": {
                                        "name": "Algemene bepalingen",
                                        "articles": (162, 168),
                                    },
                                    "6.3.2": {
                                        "name": "Aansprakelijkheid voor personen en zaken",
                                        "articles": (169, 181),
                                    },
                                },
                            },
                            "6.5": {"name": "Overeenkomsten in het algemeen", "articles": (213, 260)},
                        },
                    },
                    7: {
                        "name": "Bijzondere overeenkomsten",
                        "titels": {
                            "7.1": {"name": "Koop en ruil", "articles": (1, 50)},
                            "7.3": {"name": "Schenking", "articles": (175, 188)},
                            "7.4": {"name": "Huur", "articles": (201, 310)},
                            "7.5": {"name": "Pacht", "articles": (311, 399)},
                            "7.7": {
                                "name": "Opdracht",
                                "articles": (400, 468),
                                "afdelingen": {
                                    "7.7.1": {
                                        "name": "Opdracht in het algemeen",
                                        "articles": (400, 413),
                                    },
                                    "7.7.2": {
                                        "name": "Lastgeving",
                                        "articles": (414, 424),
                                    },
                                    "7.7.3": {
                                        "name": "Bemiddelingsovereenkomst",
                                        "articles": (425, 427),
                                    },
                                    "7.7.4": {
                                        "name": "Agentuurovereenkomst",
                                        "articles": (428, 445),
                                    },
                                    "7.7.5": {
                                        "name": "Overeenkomst inzake geneeskundige behandeling (WGBO)",
                                        "articles": (446, 468),
                                    },
                                },
                            },
                            "7.7A": {"name": "Reisovereenkomst", "articles": (500, 513)},
                            "7.8":  {"name": "Bewaarneming", "articles": (600, 609)},
                            "7.10": {"name": "Arbeidsovereenkomst", "articles": (610, 690)},
                            "7.12": {"name": "Aanneming van werk", "articles": (750, 769)},
                            "7.14": {"name": "Vaststellingsovereenkomst", "articles": (900, 910)},
                            "7.15": {"name": "Borgtocht", "articles": (850, 870)},
                        },
                    },
                    8:  {"name": "Verkeersmiddelen en vervoer"},
                    10: {"name": "Internationaal privaatrecht"},
                },
            },
        },
    },
    "public_law": {
        "name": "Publiekrecht",
        "wetboeken": {
            "Sr": {"name": "Wetboek van Strafrecht"},
            "Sv": {"name": "Wetboek van Strafvordering"},
        },
    },
    "synthetic": {
        "name": "Synthetische corpora",
        "wetboeken": {
            "LMC": {"name": "Lunar Medical Code"},
        },
    },
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches "7:454", "7:454(1)", "7:454(1).a", etc. Captures book number and
# article number; ignores anything after.
_BW_ARTICLE = re.compile(r"^(\d+):(\d+[a-zA-Z]?)")
_SYNTH_ARTICLE = re.compile(r"^synth:(\d+)")


def parse_article_id(article_id: str) -> dict[str, Any]:
    """Return the taxonomy path for one article ID.

    Handles BW notation (``'7:454'``, ``'7:454(1).a'``) and synthetic corpora
    (``'synth:15'``, ``'synth:15(1).a'``). Returns a dict with keys:
    ``rechtsgebied``, ``wetboek``, ``boek``, ``titel``, ``afdeling``, ``artikel``,
    plus their ``_name`` variants where a display name exists, plus ``path``
    (a human-readable ``>``-joined string).

    On failure, returns a dict with ``error`` set and the other fields as
    ``None``.
    """
    if not isinstance(article_id, str):
        return _error(f"non-string article_id: {article_id!r}")

    # Synthetic corpus
    m = _SYNTH_ARTICLE.match(article_id)
    if m:
        return {
            "rechtsgebied": "synthetic",
            "rechtsgebied_name": LEGAL_TREE["synthetic"]["name"],
            "wetboek": "LMC",
            "wetboek_name": LEGAL_TREE["synthetic"]["wetboeken"]["LMC"]["name"],
            "boek": None,
            "boek_name": None,
            "titel": None,
            "titel_name": None,
            "afdeling": None,
            "afdeling_name": None,
            "artikel": article_id.split("(")[0],
            "path": f"synthetic > LMC > {article_id.split('(')[0]}",
            "error": None,
        }

    # BW article
    m = _BW_ARTICLE.match(article_id)
    if not m:
        return _error(f"unparseable article_id: {article_id!r}")

    book_num = int(m.group(1))
    art_str = m.group(2)  # keep letter suffix like '50z' as-is
    art_num_match = re.match(r"(\d+)", art_str)
    art_num = int(art_num_match.group(1)) if art_num_match else 0

    bw = LEGAL_TREE["private_law"]["wetboeken"]["BW"]
    book = bw["boeken"].get(book_num)
    if book is None:
        return _error(f"unknown Book: {book_num}")

    titel_key = None
    titel_name = None
    afdeling_key = None
    afdeling_name = None

    for tk, tv in book.get("titels", {}).items():
        lo, hi = tv.get("articles", (0, 0))
        if lo <= art_num <= hi:
            titel_key = tk
            titel_name = tv.get("name")
            for ak, av in tv.get("afdelingen", {}).items():
                alo, ahi = av.get("articles", (0, 0))
                if alo <= art_num <= ahi:
                    afdeling_key = ak
                    afdeling_name = av.get("name")
                    break
            break

    path_parts = [
        "private_law",
        "BW",
        f"Boek {book_num}",
    ]
    if titel_key:
        path_parts.append(f"Titel {titel_key}")
    if afdeling_key:
        path_parts.append(f"Afd {afdeling_key}")
    path_parts.append(f"Art {book_num}:{art_num}")

    return {
        "rechtsgebied": "private_law",
        "rechtsgebied_name": LEGAL_TREE["private_law"]["name"],
        "wetboek": "BW",
        "wetboek_name": bw["name"],
        "boek": f"Boek {book_num}",
        "boek_name": book["name"],
        "titel": titel_key,
        "titel_name": titel_name,
        "afdeling": afdeling_key,
        "afdeling_name": afdeling_name,
        "artikel": f"{book_num}:{art_num}",
        "path": " > ".join(path_parts),
        "error": None,
    }


def _error(msg: str) -> dict[str, Any]:
    return {
        "rechtsgebied": None,
        "rechtsgebied_name": None,
        "wetboek": None,
        "wetboek_name": None,
        "boek": None,
        "boek_name": None,
        "titel": None,
        "titel_name": None,
        "afdeling": None,
        "afdeling_name": None,
        "artikel": None,
        "path": None,
        "error": msg,
    }


# ---------------------------------------------------------------------------
# DataFrame enrichment and reporting
# ---------------------------------------------------------------------------

TAXONOMY_COLUMNS = [
    "rechtsgebied",
    "wetboek",
    "boek",
    "titel",
    "afdeling",
    "taxonomy_path",
]


def add_taxonomy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with taxonomy columns added.

    Adds: ``rechtsgebied``, ``wetboek``, ``boek``, ``titel``, ``afdeling``,
    ``taxonomy_path``. Values are looked up from LEGAL_TREE via
    parse_article_id.
    """
    if "article_id" not in df.columns:
        raise ValueError("df must have an 'article_id' column")

    parsed = df["article_id"].apply(parse_article_id)
    out = df.copy()
    out["rechtsgebied"] = parsed.apply(lambda x: x["rechtsgebied"])
    out["wetboek"]      = parsed.apply(lambda x: x["wetboek"])
    out["boek"]         = parsed.apply(lambda x: x["boek"])
    out["titel"]        = parsed.apply(lambda x: x["titel"])
    out["afdeling"]     = parsed.apply(lambda x: x["afdeling"])
    out["taxonomy_path"] = parsed.apply(lambda x: x["path"])
    return out


def tree_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count atoms at each level of the tree, for reporting.

    Returns a DataFrame with columns ``level``, ``value``, ``n_atoms``.
    """
    df = add_taxonomy_columns(df) if "boek" not in df.columns else df
    rows = []
    for level in ["rechtsgebied", "wetboek", "boek", "titel", "afdeling"]:
        counts = df[level].fillna("(none)").value_counts()
        for value, n in counts.items():
            rows.append({"level": level, "value": value, "n_atoms": int(n)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hierarchy containment edges
# ---------------------------------------------------------------------------

def _missing(v: Any) -> bool:
    """True if v is None or a pandas/NumPy missing value.

    Needed because ``bool(float('nan')) is True`` in Python, which breaks
    naive truthiness checks on DataFrame cells.
    """
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def build_hierarchy_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Build the containment edges of the legal tree.

    Returns a DataFrame with columns ``source``, ``target``, ``edge_kind``,
    ``level``. Edge kind is always ``'contains'`` (a directed parent-child
    relationship). Level names the transition (e.g. ``atom_to_article``,
    ``article_to_afdeling``).

    Nodes are labelled with their taxonomy keys (``'Boek 7'``, ``'7.7.5'``,
    ``'7:454'``, ``'7:454(1).a'``) — string IDs suitable for a graph library.
    """
    df = add_taxonomy_columns(df) if "boek" not in df.columns else df
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(src: Any, tgt: Any, level: str) -> None:
        if _missing(src) or _missing(tgt):
            return
        key = (str(src), str(tgt))
        if key in seen:
            return
        seen.add(key)
        edges.append({"source": key[0], "target": key[1], "edge_kind": "contains", "level": level})

    # Walk each atom up the tree, only adding edges between levels where
    # both nodes exist (skips missing intermediates for the synthetic corpus).
    for _, atom in df.iterrows():
        atom_id = atom["atom_id"]
        article = atom["article_id"]
        afdeling = atom.get("afdeling")
        titel = atom.get("titel")
        boek = atom.get("boek")
        wetboek = atom.get("wetboek")
        rechtsgebied = atom.get("rechtsgebied")

        # atom -> article: always
        add(atom_id, article, "atom_to_article")

        # article -> next non-missing level up
        if not _missing(afdeling):
            add(article, afdeling, "article_to_afdeling")
        elif not _missing(titel):
            add(article, titel, "article_to_titel")
        elif not _missing(boek):
            add(article, boek, "article_to_boek")
        elif not _missing(wetboek):
            add(article, wetboek, "article_to_wetboek")

        # inter-level edges: skip any pair with a missing endpoint
        if not _missing(afdeling) and not _missing(titel):
            add(afdeling, titel, "afdeling_to_titel")
        if not _missing(titel) and not _missing(boek):
            add(titel, boek, "titel_to_boek")
        if not _missing(boek) and not _missing(wetboek):
            add(boek, wetboek, "boek_to_wetboek")
        if not _missing(wetboek) and not _missing(rechtsgebied):
            add(wetboek, rechtsgebied, "wetboek_to_rechtsgebied")

    return pd.DataFrame(edges)


# ---------------------------------------------------------------------------
# Hierarchical LOO evaluation
# ---------------------------------------------------------------------------

def evaluate_at_levels(
    df: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Leave-one-out classifier accuracy at each hierarchy level.

    For each atom, remove it, classify against the rest, and look up the
    taxonomy of the top-1 predicted atom. Compare the predicted taxonomy to
    the true one at each level (rechtsgebied → wetboek → boek → titel →
    afdeling → artikel).

    Returns a per-atom DataFrame and a summary dict of accuracies at each
    level.

    Note: for a corpus concentrated in one afdeling, the higher-level
    accuracies (boek, titel, afdeling) will be trivially high or trivially
    100%. Per-level accuracy becomes meaningful when the corpus spans
    multiple sections.
    """
    # Delayed import to avoid circular reference with atom_table
    from src.data.atom_table import classify_atom

    df = add_taxonomy_columns(df) if "boek" not in df.columns else df
    levels = ["rechtsgebied", "wetboek", "boek", "titel", "afdeling", "article_id"]
    rows = []

    for idx, atom in df.iterrows():
        rest = df.drop(idx)
        out = classify_atom(
            atom.to_dict(), rest,
            weights=weights, sim_pairs=sim_pairs,
            alpha=alpha, top_k=top_k,
        )
        top = out["top_matches"]
        if len(top) == 0:
            continue
        top_atom_id = top.iloc[0]["atom_id"]
        top_row = rest[rest["atom_id"] == top_atom_id].iloc[0]

        row = {"atom_id": atom["atom_id"]}
        for level in levels:
            true_val = atom.get(level)
            pred_val = top_row.get(level)
            hit = (
                not (_missing(true_val) or _missing(pred_val))
                and true_val == pred_val
            )
            row[f"{level}_true"] = true_val
            row[f"{level}_pred"] = pred_val
            row[f"{level}_hit"] = hit
        rows.append(row)

    eval_df = pd.DataFrame(rows)
    summary = {}
    for level in levels:
        summary[f"{level}_accuracy"] = round(eval_df[f"{level}_hit"].mean(), 3)
    return eval_df, summary


# ---------------------------------------------------------------------------
# Cross-article reference edges
# ---------------------------------------------------------------------------

# Reference-text patterns for _resolve_reference
_REF_INTERNAL = re.compile(
    r'^\s*(paragraph|lid|first paragraph|second paragraph|third paragraph|'
    r'fourth paragraph|next paragraph|preceding paragraph)\b',
    re.IGNORECASE,
)
_REF_BOOK_ARTICLE = re.compile(r'(\d+):(\d+[a-zA-Z]?)')
_REF_ARTICLE_ONLY = re.compile(r'article\s+(\d+[a-zA-Z]?)', re.IGNORECASE)


def _resolve_reference(ref_text: str, source_book: str | None) -> str | None:
    """Resolve a reference string to a target article ID, or None if internal.

    Handles three patterns, in order of specificity:

    1. Full BW notation, e.g. ``"article 6:74"`` or ``"6:162"``  → ``"6:74"``
    2. Same-book shorthand, e.g. ``"article 455"``               → uses source_book
    3. Internal references (``"paragraph 1"``, ``"lid 2"``)      → None
    """
    if not isinstance(ref_text, str):
        return None
    text = ref_text.strip()

    # Skip internal references (same-article lid/paragraph)
    if _REF_INTERNAL.match(text):
        return None

    # Full book:article notation
    m = _REF_BOOK_ARTICLE.search(text)
    if m:
        return f"{m.group(1)}:{m.group(2)}"

    # "article N" — resolve using source book
    m = _REF_ARTICLE_ONLY.search(text)
    if m and source_book:
        return f"{source_book}:{m.group(1)}"

    return None


def build_reference_edges(
    df: pd.DataFrame,
    return_unresolved: bool = False,
) -> pd.DataFrame:
    """Build directed edges from atoms to articles they explicitly reference.

    Reads the ``explicit_references`` field on each atom, parses each reference
    string, and resolves it to a target article ID. Same-article references
    (``paragraph 1``, ``second paragraph``, etc.) are skipped.

    Returns a DataFrame with columns ``source``, ``target``, ``edge_kind``,
    ``ref_text``. ``edge_kind`` is always ``'references'``.

    If ``return_unresolved=True``, returns a second DataFrame listing reference
    strings that could not be parsed, for inspection.
    """
    if "explicit_references" not in df.columns:
        raise ValueError("df must have an 'explicit_references' column")

    edges: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []

    for _, atom in df.iterrows():
        refs = atom.get("explicit_references")
        if refs is None or (isinstance(refs, list) and len(refs) == 0):
            continue
        if not isinstance(refs, list):
            continue

        atom_id = atom["atom_id"]
        source_article = atom["article_id"]

        # Source book number for same-book resolution
        source_book: str | None = None
        m = re.match(r"^(\d+):", str(source_article))
        if m:
            source_book = m.group(1)

        for ref_text in refs:
            target = _resolve_reference(ref_text, source_book)
            if target is None:
                # Distinguish "intentionally internal" from "unparseable"
                if not _REF_INTERNAL.match(str(ref_text).strip()):
                    unresolved.append({
                        "atom_id": atom_id,
                        "ref_text": ref_text,
                        "reason": "could not parse",
                    })
                continue
            if target == source_article:
                # Self-reference (article citing itself); skip
                continue
            edges.append({
                "source": atom_id,
                "target": target,
                "edge_kind": "references",
                "ref_text": ref_text,
            })

    edges_df = pd.DataFrame(edges)
    if return_unresolved:
        return edges_df, pd.DataFrame(unresolved)
    return edges_df
