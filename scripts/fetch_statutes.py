"""Download the official statute text and turn it into article records.

Source: the Basiswettenbestand (BWB) repository of the Dutch government,
which publishes every statute as consolidated XML, one file per version.

Writes:
    data/raw/statutes/<BWB-id>_<version>.xml   raw XML (git-ignored, re-downloadable)
    data/raw/statutes/sources.json             which version was used + checksum
    data/processed/statutes.jsonl              one article per line
    data/processed/statute_references.csv      article -> article references
    outputs/tables/annotation_text_check.csv   annotated atoms vs official text

Examples (from the repo root):

    python scripts/fetch_statutes.py                       # latest versions
    python scripts/fetch_statutes.py --version BWBR0005290=2026-07-01_1
    python scripts/fetch_statutes.py --offline             # re-parse saved XML
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.statutes import parse_bwb  # noqa: E402

REPO = "https://repository.officiele-overheidspublicaties.nl/bwb"

# Which statutes, and which parts of them, make up the corpus
SOURCES = {
    "BWBR0005290": {  # Burgerlijk Wetboek Boek 7
        "prefixes": ["/Boek7/Titeldeel7/Afdeling5"],        # WGBO, 7:446-468
    },
    "BWBR0005289": {  # Burgerlijk Wetboek Boek 6
        "prefixes": [
            "/Boek6/Titeldeel1/Afdeling9",    # non-performance, 6:74-94
            "/Boek6/Titeldeel1/Afdeling10",   # statutory damages, 6:95-110
            "/Boek6/Titeldeel3/Afdeling1",    # tort, general, 6:162-168
            "/Boek6/Titeldeel3/Afdeling2",    # liability for persons/things, 6:169-184
        ],
    },
}

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statutes"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "outputs" / "tables"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "statute-research-script"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def latest_version(bwb_id: str) -> str:
    manifest = fetch(f"{REPO}/{bwb_id}/manifest.xml").decode("utf-8")
    m = re.search(r'_latestItem="([^"/]+)/', manifest)
    if not m:
        raise RuntimeError(f"No latest version found in the manifest of {bwb_id}")
    return m.group(1)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9à-ÿ]+", " ", text.lower()).strip()


def _coverage(atom: str, official: str) -> float:
    """Share of the atom's characters found, in order, in the official text."""
    from difflib import SequenceMatcher

    if not atom:
        return 0.0
    blocks = SequenceMatcher(None, atom, official, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(atom)


def check_annotations(articles: list[dict]) -> list[dict]:
    """Compare each annotated atom's Dutch text with the official article text.

    ``coverage`` is the share of the atom's text found, in order, in the
    official article (1.0 = verbatim). Statuses:
      matches             verbatim in the current text
      minor differences   coverage >= 0.85: small wording changes, a different
                          cut, or words repeated across atoms
      differs             coverage < 0.85: the wording is substantially not
                          the current law (older version, or not copied verbatim)
    ``article_in_force_since`` shows when the current version took effect,
    which helps tell an amended article from an inaccurate annotation.
    """
    from src.data.atom_table import build_atom_table, drop_synthetic

    official = {a["article_id"]: a for a in articles}
    atoms = drop_synthetic(build_atom_table(PROJECT_ROOT / "data" / "annotations"))
    rows = []
    for atom in atoms.itertuples(index=False):
        text = _norm(atom.text_nl or "")
        art = official.get(atom.article_id)
        coverage = None
        if art is None:
            status = "article not downloaded"
        elif not text:
            status = "no Dutch text in annotation"
        else:
            full = _norm(art["text"])
            coverage = 1.0 if text in full else round(_coverage(text, full), 3)
            status = ("matches" if coverage == 1.0
                      else "minor differences" if coverage >= 0.85 else "differs")
        rows.append({
            "atom_id": atom.atom_id, "article_id": atom.article_id, "status": status,
            "coverage": coverage,
            "article_in_force_since": art["in_force_since"] if art else "",
            "annotated_text_nl": atom.text_nl,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="append", default=[], metavar="ID=VERSION",
                        help="pin a version, e.g. BWBR0005290=2026-07-01_1")
    parser.add_argument("--offline", action="store_true",
                        help="re-parse the XML recorded in sources.json, download nothing")
    args = parser.parse_args()
    pinned = dict(v.split("=", 1) for v in args.version)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sources_file = RAW_DIR / "sources.json"
    previous = json.loads(sources_file.read_text("utf-8")) if sources_file.exists() else {}

    sources, articles = {}, []
    for bwb_id, cfg in SOURCES.items():
        if args.offline:
            version = previous[bwb_id]["version"]
            raw = (RAW_DIR / f"{bwb_id}_{version}.xml").read_bytes()
        else:
            version = pinned.get(bwb_id) or latest_version(bwb_id)
            url = f"{REPO}/{bwb_id}/{version}/xml/{bwb_id}_{version}.xml"
            print(f"Downloading {bwb_id} version {version} ...")
            raw = fetch(url)
            (RAW_DIR / f"{bwb_id}_{version}.xml").write_bytes(raw)
            sources[bwb_id] = {
                "version": version, "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            }
        parsed, repealed = parse_bwb(raw, cfg["prefixes"], bwb_id=bwb_id, version=version)
        print(f"  {bwb_id}: {len(parsed)} articles in force"
              + (f", skipped repealed: {', '.join(repealed)}" if repealed else ""))
        articles.extend(parsed)

    if not args.offline:
        sources_file.write_text(json.dumps(sources, indent=2), "utf-8")

    with (PROCESSED / "statutes.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    refs = [{"source_article": a["article_id"], **r} for a in articles for r in a["references"]]
    with (PROCESSED / "statute_references.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_article", "source_lid", "ref_text",
                                               "target_article", "target_lid", "target_bwb",
                                               "target_path"])
        writer.writeheader()
        writer.writerows(refs)

    check = check_annotations(articles)
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "annotation_text_check.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(check[0]))
        writer.writeheader()
        writer.writerows(check)

    in_corpus = {a["article_id"] for a in articles}
    internal = sum(1 for r in refs if r["target_article"] in in_corpus)
    counts: dict[str, int] = {}
    for row in check:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"\n{len(articles)} articles -> data/processed/statutes.jsonl")
    print(f"{len(refs)} references ({internal} between articles in this corpus) "
          "-> data/processed/statute_references.csv")
    print("Annotated atoms vs official text: "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
          + " -> outputs/tables/annotation_text_check.csv")


if __name__ == "__main__":
    main()
