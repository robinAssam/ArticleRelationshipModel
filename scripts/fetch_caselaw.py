
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.caselaw import bw_citations  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw" / "case_law"
XML_DIR = RAW / "xml"
INDEX = RAW / "index.csv"
SAMPLE_LIST = RAW / "sample.csv"
CITATIONS = PROJECT_ROOT / "data" / "processed" / "caselaw_citations.csv"

API = "https://data.rechtspraak.nl/uitspraken"
SUBJECT = "http://psi.rechtspraak.nl/rechtsgebied#civielRecht"
PAUSE = 0.25  # seconds between requests: be polite to the public API

WGBO = {f"7:{n}" for n in range(446, 469)}

# Medical vocabulary for the candidate filter (matched in title + summary)


MEDICAL = re.compile(
    r"\b(wgbo|geneeskundige behandeling|behandelingsovereenkomst|hulpverlener|zorgverlener|"
    r"medisch\w*|arts|artsen|huisarts\w*|chirurg\w*|ziekenhuis\w*|kliniek|patiënt\w*|patient\w*|"
    r"tandarts\w*|orthodontist|fysiotherapeut\w*|verloskundig\w*|gynaecolo\w*|anesthesi\w*|"
    r"orthope\w*|psychiater|psycholoog|ggz|verpleeg\w*|operatie\w*|beroepsfout|"
    r"medisch dossier|informed consent|7:4[4-6]\d)\b",
    re.IGNORECASE,
)


def get(url: str, params: list[tuple[str, str]] | None = None, tries: int = 4) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thesis-research-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # network hiccup: back off and retry
            if attempt == tries - 1:
                raise
            print(f"  retry after error: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------- index

def cmd_index(args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows, offset, total = [], 0, None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    while total is None or offset < total:
        body = get(f"{API}/zoeken", [
            ("type", "Uitspraak"), ("return", "DOC"), ("subject", SUBJECT),
            ("date", args.date_from), ("date", args.date_to),
            ("max", "1000"), ("from", str(offset)),
        ])
        if total is None:
            total = int(re.search(rb"Aantal gevonden ECLI's: (\d+)", body).group(1))
            print(f"{total} civil judgments with text, {args.date_from} to {args.date_to}")
        feed = ET.fromstring(body.lstrip(b"\xef\xbb\xbf"))
        entries = feed.findall("a:entry", ns)
        if not entries:
            break
        for e in entries:
            title = e.findtext("a:title", "", ns)
            summary = e.findtext("a:summary", "", ns)
            # Title: "ECLI, court name (may contain commas), dd-mm-yyyy, case number"
            date = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", title)
            court = title.split(",", 1)[1][: date.start() - len(title.split(",", 1)[0]) - 1] if date else ""
            rows.append({
                "ecli": e.findtext("a:id", "", ns),
                "court": court.strip(" ,"),
                "date": date.group(1) if date else "",
                "candidate": int(bool(MEDICAL.search(title + " " + summary))),
                "summary": " ".join(summary.split()),
            })
        offset += len(entries)
        print(f"  {offset}/{total}", end="\r")
        time.sleep(PAUSE)
    with INDEX.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_cand = sum(r["candidate"] for r in rows)
    print(f"\n{len(rows)} summaries -> {INDEX.relative_to(PROJECT_ROOT)}; "
          f"{n_cand} medical candidates ({100 * n_cand / len(rows):.1f}%)")


# ------------------------------------------------------------- fetch / sample

def _read_index() -> list[dict]:
    with INDEX.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _download(eclis: list[str], label: str) -> None:
    XML_DIR.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eclis if not (XML_DIR / f"{e.replace(':', '_')}.xml").exists()]
    print(f"{label}: {len(eclis)} judgments, {len(eclis) - len(todo)} already downloaded")
    for i, ecli in enumerate(todo, 1):
        body = get(f"{API}/content", [("id", ecli)])
        (XML_DIR / f"{ecli.replace(':', '_')}.xml").write_bytes(body)
        print(f"  {i}/{len(todo)}", end="\r")
        time.sleep(PAUSE)
    print()


def _judgment_text(path: Path) -> tuple[str, str]:
    """(procedure, full text) of a downloaded judgment."""
    raw = path.read_bytes()
    root = ET.fromstring(raw.lstrip(b"\xef\xbb\xbf"))
    procedure = ""
    for e in root.iter():
        if e.tag.endswith("}procedure") and e.text:
            procedure = e.text.strip()
            break
    body = next((e for e in root.iter() if e.tag.endswith("}uitspraak")), None)
    text = " ".join(body.itertext()) if body is not None else ""
    return procedure, " ".join(text.split())


def _write_citations() -> None:
    index = {r["ecli"]: r for r in _read_index()}
    sampled = set()
    if SAMPLE_LIST.exists():
        with SAMPLE_LIST.open(encoding="utf-8") as f:
            sampled = {r["ecli"] for r in csv.DictReader(f)}
    rows = []
    for path in sorted(XML_DIR.glob("*.xml")):
        ecli = path.stem.replace("_", ":")
        meta = index.get(ecli, {})
        procedure, text = _judgment_text(path)
        cited = bw_citations(text)
        wgbo = sorted(cited & WGBO, key=lambda a: int(a.split(":")[1].rstrip("abcdefgh")))
        rows.append({
            "ecli": ecli, "year": ecli.split(":")[3], "date": meta.get("date", ""),
            "court": meta.get("court", ""),
            "procedure": procedure,
            "set": "sample" if ecli in sampled else "candidate",
            "mentions_wgbo": int(bool(re.search(r"\bWGBO\b", text))),
            "n_wgbo_articles": len(wgbo),
            "wgbo_articles": " ".join(wgbo),
            "book6_7_articles": " ".join(sorted(cited)),
        })
    CITATIONS.parent.mkdir(parents=True, exist_ok=True)
    with CITATIONS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_fetch(args) -> None:
    cands = [r["ecli"] for r in _read_index() if r["candidate"] == "1"]
    if args.limit:
        cands = cands[: args.limit]
    _download(cands, "Candidates")
    _write_citations()
    cmd_report(args)


def cmd_sample(args) -> None:
    rest = [r["ecli"] for r in _read_index() if r["candidate"] == "0"]
    random.Random(args.seed).shuffle(rest)
    sample = rest[: args.n]
    with SAMPLE_LIST.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecli"])
        w.writerows([e] for e in sample)
    _download(sample, f"Random non-candidates (seed {args.seed})")
    _write_citations()
    cmd_report(args)


# -------------------------------------------------------------------- report

def cmd_report(args) -> None:
    import pandas as pd

    if XML_DIR.exists() and any(XML_DIR.glob("*.xml")):
        _write_citations()  # rebuild from the downloaded judgments

    idx = pd.read_csv(INDEX)
    n_all, n_cand = len(idx), int(idx["candidate"].sum())
    print(f"\nIndex: {n_all} civil judgments with text, {n_cand} medical candidates")
    if not CITATIONS.exists():
        return
    c = pd.read_csv(CITATIONS, dtype=str).fillna("")
    c["n_wgbo_articles"] = c["n_wgbo_articles"].astype(int)
    cand = c[c["set"] == "candidate"]
    hits = cand[cand["n_wgbo_articles"] > 0]
    print(f"Candidates downloaded: {len(cand)}/{n_cand}")
    print(f"  cite at least one WGBO article: {len(hits)}  (mention 'WGBO': {int(cand['mentions_wgbo'].astype(int).sum())})")
    if len(hits):
        counts = pd.Series(" ".join(hits["wgbo_articles"]).split()).value_counts()
        print("  most cited WGBO articles: " + ", ".join(f"{a} ({n})" for a, n in counts.head(10).items()))
        years = hits["year"].value_counts().sort_index()
        print("  per year: " + ", ".join(f"{y}: {n}" for y, n in years.items()))
        print("  cite 2+ WGBO articles: " + str(int((hits["n_wgbo_articles"] >= 2).sum())))
    samp = c[c["set"] == "sample"]
    if len(samp):
        s_hits = int((samp["n_wgbo_articles"] > 0).sum())
        est = s_hits / len(samp) * (n_all - n_cand)
        print(f"Random non-candidates checked: {len(samp)}; cite WGBO: {s_hits} "
              f"-> roughly {est:.0f} WGBO judgments missed by the keyword filter")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--date-from", default="2000-01-01")
    pi.add_argument("--date-to", default="2026-12-31")
    pf = sub.add_parser("fetch")
    pf.add_argument("--limit", type=int, default=0, help="only the first N candidates")
    ps = sub.add_parser("sample")
    ps.add_argument("--n", type=int, default=1000)
    ps.add_argument("--seed", type=int, default=42)
    sub.add_parser("report", help="rebuild the citation table from downloaded judgments and print counts")
    args = p.parse_args()
    {"index": cmd_index, "fetch": cmd_fetch, "sample": cmd_sample, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
