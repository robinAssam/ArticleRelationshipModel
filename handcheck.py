"""Hand check of the study A dataset: are the questions and answers right?

Run with (from the repo root):

    streamlit run handcheck.py

For each judgment in data/validation/handcheck_sample.csv (drawn by
scripts/sample_handcheck.py) the annotator answers:

    1. Facts: did the script find the facts section, and is it the right text?
    2. Answers: for each corpus article the judgment cites, is it applied in
       substance, cited only in passing, cited for procedure, or misread?
    3. Missed articles, and free notes.

The page shows no system output, so the check stays blind to the results.
Each annotator's labels go to data/validation/handcheck_<name>.csv (one row
per judgment, saved on every submit); annotators never see each other's
file. Judgments not yet downloaded are fetched from rechtspraak.nl the first
time they are opened. Local tool only: not part of the deployed app.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.caselaw import bw_citation_spans, bw_citations, mask_citations, parse_judgment  # noqa: E402

SAMPLE = PROJECT_ROOT / "data" / "validation" / "handcheck_sample.csv"
OUT_DIR = PROJECT_ROOT / "data" / "validation"
XML_DIR = PROJECT_ROOT / "data" / "raw" / "case_law" / "xml"
STATUTES = PROJECT_ROOT / "data" / "processed" / "statutes.jsonl"
SPLIT = PROJECT_ROOT / "data" / "processed" / "study_a" / "split.csv"
API = "https://data.rechtspraak.nl/uitspraken/content"
MIN_WORDS = 80  # as in scripts/build_study_a.py
CONTEXT = 300   # characters shown either side of a citation

FACTS_FOUND = {
    "clean": "Correct: this is the facts section, complete",
    "incomplete": "Partly: facts are cut short",
    "too_much": "Partly: runs on into the parties' arguments or the court's reasoning",
    "wrong": "Wrong: this is not the facts section",
}
FACTS_NONE = {
    "none_correct": "Right: the judgment has no usable facts section",
    "none_missed": "Missed: there is a facts section the script did not find",
}
ARTICLE_LABELS = {
    "substantive": "Substantive",
    "passing": "In passing",
    "procedural": "Procedural",
    "misread": "Misread",
}
INSTRUCTIONS = """
**Facts.** The question the systems get is the *facts section*: what happened,
as the court establishes it, without the parties' arguments or the court's
assessment. Check that the text shown under *Facts as the systems see them*
is that, and all of it. For judgments the build dropped, check whether the
judgment really has no facts section, or the script missed it.

**Leak.** Article numbers are masked as [ARTIKEL] and "WGBO" as [WET]. Tick
the box if the masked facts still give the answer away (e.g. "the duty to
inform under the treatment contract was breached").

**Answers.** Every corpus article the judgment cites counts as an answer.
For each, read where it is cited and choose:

- **Substantive**: the court applies or interprets it to decide this case,
  or it is the legal basis of a claim the court actually assesses.
- **In passing**: mentioned but not decisive, e.g. only as background
  ("there was a treatment contract (7:446 BW)"), only in a party's argument
  the court does not go into, or only in a quoted earlier judgment.
- **Procedural**: cited for a procedural point (costs, limitation as a
  formality, admissibility) rather than the substance of the dispute.
- **Misread**: not a BW article citation at all (a date, a time, a
  paragraph number), or a different article than the script thinks.

**Missed articles.** Corpus articles (WGBO 7:446–468, Book 6 non-performance,
damages, tort) the court clearly applies but the script did not list,
e.g. because the judgment only names it in words.
"""


# ------------------------------------------------------------------- data

@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


@st.cache_data
def load_corpus() -> dict[str, str]:
    """Article id -> article text, shown next to each answer as a reminder."""
    out = {}
    with STATUTES.open(encoding="utf-8") as f:
        for line in f:
            art = json.loads(line)
            out[art["article_id"]] = art["text"]
    return out


@st.cache_data
def load_split() -> dict[str, str]:
    if not SPLIT.exists():
        return {}
    return dict(pd.read_csv(SPLIT, usecols=["ecli", "split"]).itertuples(index=False))


def xml_path(ecli: str) -> Path:
    return XML_DIR / f"{ecli.replace(':', '_')}.xml"


def download(ecli: str) -> None:
    import urllib.parse
    import urllib.request

    url = API + "?" + urllib.parse.urlencode({"id": ecli})
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-research-script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    XML_DIR.mkdir(parents=True, exist_ok=True)
    xml_path(ecli).write_bytes(body)
    time.sleep(0.25)  # be polite to the public API


@st.cache_data(show_spinner=False)
def load_judgment(ecli: str) -> dict:
    """The judgment parsed as build_study_a.py sees it, plus why it was kept or dropped."""
    if not xml_path(ecli).exists():
        download(ecli)
    j = parse_judgment(xml_path(ecli).read_bytes())
    corpus = load_corpus()
    j["answers"] = sorted(bw_citations(j["text"]) & set(corpus),
                          key=lambda a: (a[0], int(re.match(r"\d+", a[2:]).group())))
    j["question"], j["n_masked"] = mask_citations(j["facts"]) if j["facts"] else ("", 0)
    if not j["facts"]:
        j["status"] = "dropped: no facts section found"
    elif len(j["question"].split()) < MIN_WORDS:
        j["status"] = f"dropped: facts shorter than {MIN_WORDS} words"
    elif bw_citations(j["question"]) & set(corpus):
        j["status"] = "dropped: article reference left after masking"
    else:
        j["status"] = "in the dataset"
    return j


def labels_path(annotator: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", annotator.lower()).strip("_")
    return OUT_DIR / f"handcheck_{slug}.csv"


def load_labels(annotator: str) -> dict[str, dict]:
    path = labels_path(annotator)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r["ecli"]: r for r in df.to_dict("records")}


def save_labels(annotator: str, labels: dict[str, dict]) -> None:
    order = list(load_sample()["ecli"])
    rows = sorted(labels.values(), key=lambda r: order.index(r["ecli"]))
    labels_path(annotator).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(labels_path(annotator), index=False)


def parse_articles(cell: str) -> dict[str, str]:
    """"7:446=passing 7:453=substantive" -> {"7:446": "passing", "7:453": "substantive"}"""
    return dict(pair.split("=", 1) for pair in cell.split() if "=" in pair)


# ---------------------------------------------------------------- display

def highlight(text: str, articles: set[str]) -> str:
    """HTML-escaped text with every citation of the given articles marked."""
    out, pos = [], 0
    for start, end, article in bw_citation_spans(text):
        if article in articles:
            out += [html.escape(text[pos:start]),
                    f"<mark>{html.escape(text[start:end])}</mark>"]
            pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def contexts(text: str, article: str) -> list[str]:
    """Each place the article is cited, with CONTEXT characters either side."""
    out = []
    for start, end, a in bw_citation_spans(text):
        if a == article:
            lo, hi = max(0, start - CONTEXT), min(len(text), end + CONTEXT)
            out.append(("…" if lo else "") + html.escape(text[lo:start])
                       + f"<mark>{html.escape(text[start:end])}</mark>"
                       + html.escape(text[end:hi]) + ("…" if hi < len(text) else ""))
    return out


def scroll_box(body: str, height: int = 420) -> None:
    st.markdown(f'<div style="max-height:{height}px;overflow-y:auto;padding:0.75rem 1rem;'
                f'border:1px solid rgba(128,128,128,0.35);border-radius:6px;'
                f'line-height:1.55;font-size:0.92rem">{body}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ pages

def check_page(annotator: str, sample: pd.DataFrame, labels: dict[str, dict]) -> None:
    eclis = list(sample["ecli"])
    idx = st.session_state["idx"]
    ecli = eclis[idx]

    try:
        with st.spinner(f"Downloading {ecli} from rechtspraak.nl…"):
            j = load_judgment(ecli)
    except Exception as e:
        st.error(f"Could not load {ecli}: {e}")
        return
    done = labels.get(ecli, {})

    st.markdown(f"### {idx + 1} / {len(eclis)} · {ecli}")
    split = load_split().get(ecli)
    st.caption(" · ".join(x for x in [
        j["court"], j["date"], j["procedure"],
        f"[open on rechtspraak.nl](https://uitspraken.rechtspraak.nl/details?id={ecli})",
    ] if x))
    status = j["status"] + (f" ({split})" if split else "")
    (st.success if j["status"] == "in the dataset" else st.warning)(f"Build: {status}")

    answers = set(j["answers"])
    tab_q, tab_f, tab_all = st.tabs(["Facts as the systems see them", "Facts, unmasked",
                                     "Full judgment"])
    with tab_q:
        if j["question"]:
            st.caption(f"{len(j['question'].split())} words · found by {j['facts_method']} · "
                       f"{j['n_masked']} references masked")
            scroll_box(html.escape(j["question"]).replace("[ARTIKEL]", "<b>[ARTIKEL]</b>")
                       .replace("[WET]", "<b>[WET]</b>"))
        else:
            st.info("The script found no facts section. Look in the full judgment.")
    with tab_f:
        scroll_box(highlight(j["facts"], answers) if j["facts"] else "<i>none found</i>")
    with tab_all:
        if j["summary"]:
            st.caption("Summary (inhoudsindicatie): " + j["summary"])
        scroll_box(highlight(j["text"], answers), height=640)

    with st.form(key=f"form_{ecli}"):
        st.markdown("#### 1. Facts")
        options = FACTS_FOUND if j["facts"] else FACTS_NONE
        keys = list(options)
        facts = st.radio("Facts section", keys, format_func=options.get, label_visibility="collapsed",
                         index=keys.index(done["facts"]) if done.get("facts") in keys else None)
        leak = st.checkbox("The masked facts still give an answer away",
                           value=done.get("leak") == "1", disabled=not j["question"])

        st.markdown(f"#### 2. Answers ({len(j['answers'])} cited corpus articles)")
        previous = parse_articles(done.get("articles", ""))
        chosen = {}
        corpus = load_corpus()
        label_keys = list(ARTICLE_LABELS)
        for article in j["answers"]:
            places = contexts(j["text"], article)
            st.markdown(f"**{article}** · cited {len(places)}×")
            with st.expander(f"Where {article} is cited · article text"):
                st.caption(corpus.get(article, "")[:600])
                for c in places:
                    scroll_box(c, height=200)
            chosen[article] = st.radio(
                article, label_keys, format_func=ARTICLE_LABELS.get, horizontal=True,
                key=f"{ecli}_{article}", label_visibility="collapsed",
                index=label_keys.index(previous[article]) if previous.get(article) in label_keys
                else None)

        st.markdown("#### 3. Anything else")
        missed = st.text_input("Corpus articles applied but not listed (space-separated, e.g. 7:450)",
                               value=done.get("missed", ""))
        notes = st.text_area("Notes", value=done.get("notes", ""), height=80)
        submitted = st.form_submit_button("Save and next", type="primary")

    if submitted:
        missing = [a for a, v in chosen.items() if v is None]
        if facts is None or missing:
            st.error("Answer every question first: "
                     + ", ".join((["facts"] if facts is None else []) + missing))
            return
        labels[ecli] = {
            "ecli": ecli, "annotator": annotator, "build_status": j["status"],
            "facts": facts, "leak": int(leak) if j["question"] else "",
            "articles": " ".join(f"{a}={v}" for a, v in chosen.items()),
            "missed": " ".join(missed.split()), "notes": notes.strip(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_labels(annotator, labels)
        st.session_state["idx"] = next((i for i, e in enumerate(eclis) if e not in labels),
                                       min(idx + 1, len(eclis) - 1))
        st.rerun()


def results_page(annotator: str, labels: dict[str, dict]) -> None:
    st.markdown(f"### Results so far · {annotator}")
    if not labels:
        st.info("Nothing checked yet.")
        return
    df = pd.DataFrame(labels.values())
    in_data = df["build_status"] == "in the dataset"
    st.caption(f"{len(df)} judgments checked: {int(in_data.sum())} in the dataset, "
               f"{int((~in_data).sum())} dropped by the build")

    left, right = st.columns(2)
    with left:
        st.markdown("**Facts section**")
        names = {**FACTS_FOUND, **FACTS_NONE}
        st.dataframe(df["facts"].map(names).value_counts().rename("judgments"))
        leaks = df.loc[in_data, "leak"].astype(str)
        st.caption(f"Masked facts that still give an answer away: "
                   f"{(leaks == '1').sum()} of {len(leaks)}")
    with right:
        st.markdown("**Answers (judgments in the dataset)**")
        arts = pd.DataFrame([{"article": a, "label": v}
                             for cell in df.loc[in_data, "articles"]
                             for a, v in parse_articles(cell).items()])
        if len(arts):
            counts = arts["label"].map(ARTICLE_LABELS).value_counts()
            st.dataframe(pd.DataFrame({"answers": counts,
                                       "share": (counts / counts.sum()).round(2)}))
            st.caption(f"{(arts['label'] == 'substantive').mean():.0%} of answers substantive")
    if len(arts):
        st.markdown("**By article**")
        st.dataframe(pd.crosstab(arts["article"], arts["label"].map(ARTICLE_LABELS),
                                 margins=True, margins_name="total"))
    st.markdown("**All labels**")
    st.dataframe(df, hide_index=True)


# ------------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(page_title="Hand check", layout="wide")
    if not SAMPLE.exists():
        st.error("No sample yet: run `python scripts/sample_handcheck.py` first.")
        return
    sample = load_sample()

    with st.sidebar:
        st.markdown("## Hand check")
        annotator = st.text_input("Your name", key="annotator",
                                  help="Your labels are saved to data/validation/handcheck_<name>.csv")
        if not annotator.strip():
            st.info("Enter your name to start.")
            st.markdown(INSTRUCTIONS)
            return
        labels = load_labels(annotator)
        st.progress(len(labels) / len(sample), text=f"{len(labels)} / {len(sample)} checked")
        page = st.radio("Page", ["Check", "Results"], horizontal=True)
        eclis = list(sample["ecli"])
        # Start at the first judgment not yet checked
        st.session_state.setdefault("idx", next(
            (i for i, e in enumerate(eclis) if e not in labels), 0))
        pick = st.selectbox("Go to", range(len(eclis)), index=st.session_state["idx"],
                            format_func=lambda i: f"{'✓' if eclis[i] in labels else '·'} "
                                                  f"{i + 1}. {eclis[i]}")
        if pick != st.session_state.get("idx"):
            st.session_state["idx"] = pick
        st.caption(f"Saving to `{labels_path(annotator).relative_to(PROJECT_ROOT)}`")
        with st.expander("Instructions"):
            st.markdown(INSTRUCTIONS)

    if page == "Check":
        check_page(annotator, sample, labels)
    else:
        results_page(annotator, labels)


main()
