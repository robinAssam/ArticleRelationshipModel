"""Official statute text from the Basiswettenbestand (BWB).

The BWB publishes every Dutch statute as consolidated XML: one file per
version, with the full structure (boek > titeldeel > afdeling > paragraaf >
artikel > lid) and machine-readable cross-references (``intref``). This
module turns those files into one record per article.

Usage:
    from src.data.statutes import parse_bwb, load_statutes

    articles = parse_bwb(xml_bytes, prefixes=["/Boek7/Titeldeel7/Afdeling5"])
    df = load_statutes()          # data/processed/statutes.jsonl as a DataFrame

Downloading lives in scripts/fetch_statutes.py.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Structural levels above an article, outermost first
LEVELS = ["boek", "titeldeel", "afdeling", "paragraaf", "sub-paragraaf"]

# Never part of the legal text
_SKIP_TAGS = {"meta-data", "kop", "lidnr"}

DEFAULT_PROCESSED = Path("data/processed/statutes.jsonl")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text(elem: ET.Element) -> str:
    """Legal text of an element, without headings, lid numbers or metadata."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.tag in _SKIP_TAGS:
            if e.tail:
                parts.append(e.tail)
            return
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
        if e is not elem and e.tail:
            parts.append(e.tail)

    walk(elem)
    return _clean(" ".join(parts))


def _heading(elem: ET.Element) -> dict:
    """``{"label": "Afdeling 5", "title": "De overeenkomst ..."}`` from a kop."""
    kop = elem.find("kop")
    if kop is None:
        return {"label": elem.get("label", ""), "title": ""}
    title = kop.find("titel")
    return {
        "label": elem.get("label") or _clean(" ".join(
            (kop.findtext(t) or "") for t in ("label", "nr"))),
        "title": _clean("".join(title.itertext())) if title is not None else "",
    }


_PATH_ARTICLE = re.compile(r"/Boek(\d+)/.*?Artikel([0-9]+[a-z]*)(?:/Lid([0-9]+[a-z]*))?", re.I)


def path_to_article(path: str) -> tuple[str | None, str | None]:
    """'/Boek7/Titeldeel7/Afdeling5/Artikel455/Lid2' -> ('7:455', '2')."""
    m = _PATH_ARTICLE.search(path or "")
    if not m:
        return None, None
    return f"{m.group(1)}:{m.group(2)}", m.group(3)


def parse_bwb(
    xml_bytes: bytes,
    prefixes: list[str] | None = None,
    bwb_id: str = "",
    version: str = "",
) -> tuple[list[dict], list[str]]:
    """Parse a BWB XML file into article records.

    Parameters
    ----------
    xml_bytes : bytes
        The raw XML of one BWB version.
    prefixes : list of str or None
        Keep only articles whose structural path starts with one of these,
        e.g. ``"/Boek7/Titeldeel7/Afdeling5"``. None keeps every article.

    Returns
    -------
    (articles, repealed)
        ``articles``: one dict per article in force, with its position in the
        legal tree, its text per lid (an unnumbered single paragraph has
        ``lid == ""``), and the references it makes.
        ``repealed``: ids of matching articles marked as repealed (vervallen).
    """
    root = ET.fromstring(xml_bytes)
    parent = {child: p for p in root.iter() for child in p}
    articles: list[dict] = []
    repealed: list[str] = []

    for art in root.iter("artikel"):
        path = art.get("bwb-ng-variabel-deel", "")
        if prefixes and not any(path.startswith(p) for p in prefixes):
            continue
        article_id, _ = path_to_article(path)
        if article_id is None:
            continue
        if art.get("status") == "vervallen":
            repealed.append(article_id)
            continue

        # Position in the legal tree, outermost level first
        tree: list[dict] = []
        p = parent.get(art)
        while p is not None:
            if p.tag in LEVELS:
                tree.append({"level": p.tag, **_heading(p)})
            p = parent.get(p)
        tree.reverse()

        # Articles with a single paragraph have no <lid> elements; they get one
        # entry with an empty number, so every article has at least one lid.
        lids = [
            {"lid": (lid.findtext("lidnr") or "").strip(), "text": _text(lid)}
            for lid in art.findall("lid")
        ] or [{"lid": "", "text": _text(art)}]
        text = " ".join(l["text"] for l in lids)

        references = []
        for ref in art.iter("intref"):
            src = ref
            while src is not None and src.tag not in ("lid", "artikel"):
                src = parent.get(src)
            target_path = ref.get("bwb-ng-variabel-deel", "")
            target_article, target_lid = path_to_article(target_path)
            references.append({
                "source_lid": (src.findtext("lidnr") or "").strip() if src is not None and src.tag == "lid" else "",
                "ref_text": _clean("".join(ref.itertext())),
                "target_bwb": ref.get("bwb-id", ""),
                "target_article": target_article,
                "target_lid": target_lid,
                "target_path": target_path,
            })

        by_level = {t["level"]: t for t in tree}
        articles.append({
            "article_id": article_id,
            "bwb_id": bwb_id,
            "version": version,
            "path": path,
            "in_force_since": art.get("inwerking", ""),
            "source": art.get("bron", ""),
            **{f"{lvl.replace('-', '_')}": by_level.get(lvl, {}).get("label", "") for lvl in LEVELS},
            **{f"{lvl.replace('-', '_')}_name": by_level.get(lvl, {}).get("title", "") for lvl in LEVELS},
            "lids": lids,
            "text": text,
            "references": references,
        })
    return articles, repealed


def load_statutes(path: str | Path = DEFAULT_PROCESSED) -> pd.DataFrame:
    """Load the processed statute records (one row per article)."""
    with Path(path).open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])
