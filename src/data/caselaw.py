"""Judgments from rechtspraak.nl: parsing, facts extraction and citation masking.

Study A uses real judgments as retrieval questions: the facts section of a
judgment is the question, and the BW articles the judgment cites are the
answer. This module holds the parts that work on a single judgment:

    bw_citations(text)        Book 6/7 BW articles cited in a text
    parse_judgment(xml)       metadata, related judgments, full text, facts
    mask_citations(text)      replace article references by [ARTIKEL]

Masking matters because judges often name the article inside the facts
("ingevolge art. 7:453 BW"). Left in, the answer would be part of the
question and every retrieval system would look better than it is.

Downloading lives in scripts/fetch_caselaw.py, building the dataset in
scripts/build_study_a.py.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------ citations

# "artikel 7:454 BW", "art. 7:448 lid 2 BW", "artikelen 7:446 en 7:453 BW", "(6:162 BW)"
_CITE = re.compile(r"(?<![\d:])([67]):(\d{1,3}[a-z]?)(?![\d:])")
_NEXT_LAW = re.compile(r"\b(BW|Burgerlijk Wetboek|Awb|Rv|Wft|Sr|Sv|Wvw|AOW|WIA|ZW)\b")


def bw_citations(text: str) -> set[str]:
    """Book 6/7 BW article citations in a judgment, e.g. {'7:454', '6:162'}.

    A citation counts unless the next law named within 60 characters is a
    different act (e.g. "artikel 7:11 Awb"); in civil judgments a bare
    "7:454" refers to the BW.
    """
    found = set()
    for m in _CITE.finditer(text):
        law = _NEXT_LAW.search(text, m.end(), m.end() + 60)
        if law is None or law.group(1) in ("BW", "Burgerlijk Wetboek"):
            found.add(f"{m.group(1)}:{m.group(2)}")
    return found


# -------------------------------------------------------------------- masking

ARTICLE_MASK = "[ARTIKEL]"
LAW_MASK = "[WET]"

_NUM = r"\d{1,3}[a-z]?(?::\d{1,4}[a-z]?)?"
# "lid 2", "leden 1 en 3", "sub b", "onder a", "aanhef en onder c", "tweede volzin"
_PART = (r"(?:\s*,?\s*(?:lid|leden)\s+\d+(?:\s*(?:,|en|t/m|-)\s*\d+)*"
         r"|\s*,?\s*(?:aanhef\s+en\s+)?(?:sub|onder)\s+[a-z\d]{1,3}\b\.?"
         r"|\s*,?\s*(?:eerste|tweede|derde|vierde|vijfde)\s+(?:lid|volzin|zin)"
         r"|\s*\(?\s*oud\s*\)?)*")
_SEP = r"\s*(?:,|/|-|–|\ben\b|\bof\b|\bjo\.?|\bjuncto\b|\bt/m\b|\btot\s+en\s+met\b)\s*"
_LAW = (r"(?:BW|Burgerlijk\s+Wetboek|WGBO"
        r"|Wet\s+op\s+de\s+geneeskundige\s+behandelingsovereenkomst)")
# "van Boek 7 BW", "Boek 6 van het Burgerlijk Wetboek", "BW"
_LAW_SUFFIX = (rf"(?:\s+(?:van\s+)?(?:het\s+)?(?:Boek\s+\d+\s+)?(?:van\s+(?:het\s+)?)?{_LAW}"
               r"|\s+(?:van\s+)?Boek\s+\d+)?")

# References introduced by "artikel(en)" / "art." / "artt.": any law, any form
_WORDED_REF = re.compile(
    rf"\b(?:artikelen|artikel|artt?\.?)\s*{_NUM}{_PART}(?:{_SEP}{_NUM}{_PART})*{_LAW_SUFFIX}",
    re.IGNORECASE,
)
# Bare "7:453 BW", "6:162 lid 1 BW", "(7:446 e.v. BW)"
_BARE_REF = re.compile(
    rf"(?<![\w:.])(\d{{1,2}}):(\d{{1,4}}[a-z]?)(?![\d:]){_PART}"
    rf"(?:{_SEP}\d{{1,2}}:\d{{1,4}}[a-z]?{_PART})*(?:\s+e\.v\.)?{_LAW_SUFFIX}",
)
_LAW_NAME = re.compile(r"\b(?:WGBO|Wet\s+op\s+de\s+geneeskundige\s+behandelingsovereenkomst)\b",
                       re.IGNORECASE)


def _is_bare_reference(m: re.Match) -> bool:
    """A bare book:article reference, as opposed to a clock time ("om 7:45 uur").

    True when a law follows it, or when it is Book 6/7 with an article
    number of 60 or more (every corpus article; never a time).
    """
    book, art = int(m.group(1)), int(re.match(r"\d+", m.group(2)).group())
    names_law = re.search(_LAW + r"\s*$", m.group(0)) is not None
    return names_law or (book in (6, 7) and art >= 60)


def mask_citations(text: str) -> tuple[str, int]:
    """Replace statutory article references by [ARTIKEL] and WGBO by [WET].

    Returns the masked text and the number of replacements. References to
    any law are masked, not only the BW: it keeps the rule simple, and an
    article number is never a fact of the case.
    """
    n = 0

    def article(m: re.Match) -> str:
        nonlocal n
        if m.re is _BARE_REF and not _is_bare_reference(m):
            return m.group(0)
        n += 1
        return ARTICLE_MASK

    masked = _WORDED_REF.sub(article, text)
    masked = _BARE_REF.sub(article, masked)
    masked, k = _LAW_NAME.subn(LAW_MASK, masked)
    return masked, n + k


# ------------------------------------------------------------------- parsing

# Section titles that introduce the facts: "2 De feiten", "Feiten",
# "De vaststaande feiten", "Feitelijke achtergrond", "Uitgangspunten en feiten"
_FACTS_TITLE = re.compile(
    r"^(?:de\s+)?(?:vaststaande\s+|relevante\s+|uitgangspunten\s+en\s+)?feiten\b"
    r"|^feitelijke\s+achtergrond\b",
    re.IGNORECASE,
)
# The same as a numbered heading in running text, for judgments without
# section markup: "2. De feiten 2.1 ..."
_FACTS_HEADING = re.compile(
    r"(?:^|\s)(\d{1,2})\.?\s+(?:De\s+|Feitelijke\s+)?(?:vaststaande\s+|relevante\s+)?"
    r"(?:feiten|achtergrond)\b(?:\s+en\s+omstandigheden)?",
    re.IGNORECASE,
)
_MAX_FALLBACK_CHARS = 40_000


def _clean(text: str) -> str:
    return " ".join(text.split())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _facts_from_sections(body: ET.Element) -> str:
    for section in body.iter():
        if _local(section.tag) != "section":
            continue
        title = next((c for c in section if _local(c.tag) == "title"), None)
        if title is None:
            continue
        heading = re.sub(r"^[\d.\s]+", "", _clean("".join(title.itertext())))
        if _FACTS_TITLE.match(heading):
            return _clean(" ".join(" ".join(c.itertext()) for c in section if c is not title))
    return ""


def _facts_from_text(text: str) -> str:
    m = _FACTS_HEADING.search(text)
    if not m:
        return ""
    nxt = int(m.group(1)) + 1
    # The next top-level heading: "3. De beoordeling", "3 Het geschil"
    end = re.compile(rf"\s{nxt}\.?\s+(?=[A-Z])").search(text, m.end())
    stop = end.start() if end else min(len(text), m.end() + _MAX_FALLBACK_CHARS)
    return _clean(text[m.end():stop])


def parse_judgment(raw: bytes) -> dict:
    """One judgment's metadata, related judgments, full text and facts section.

    ``facts_method`` says how the facts were found: "section" (a section
    titled "feiten"), "heading" (a numbered "feiten" heading in running text,
    for judgments without section markup) or "" (none found).
    """
    root = ET.fromstring(raw.lstrip(b"\xef\xbb\xbf"))
    meta: dict = {"ecli": "", "date": "", "court": "", "procedure": "", "related": []}
    for e in root.iter():
        tag = _local(e.tag)
        if tag == "identifier" and not meta["ecli"] and (e.text or "").startswith("ECLI:"):
            meta["ecli"] = e.text.strip()
        elif tag == "date" and not meta["date"]:
            meta["date"] = (e.text or "").strip()
        elif tag == "creator" and not meta["court"]:
            meta["court"] = (e.text or "").strip()
        elif tag == "procedure" and not meta["procedure"]:
            meta["procedure"] = (e.text or "").strip()
        elif tag == "relation":
            ref = next((v for k, v in e.attrib.items() if k.endswith("resourceIdentifier")), "")
            if ref.startswith("ECLI:"):
                meta["related"].append(ref)

    summary = next((e for e in root.iter() if _local(e.tag) == "inhoudsindicatie"), None)
    body = next((e for e in root.iter() if _local(e.tag) == "uitspraak"), None)
    text = _clean(" ".join(body.itertext())) if body is not None else ""

    facts, method = "", ""
    if body is not None:
        facts = _facts_from_sections(body)
        method = "section" if facts else ""
        if not facts:
            facts = _facts_from_text(text)
            method = "heading" if facts else ""

    return {
        **meta,
        "summary": _clean(" ".join(summary.itertext())) if summary is not None else "",
        "text": text,
        "facts": facts,
        "facts_method": method,
    }
