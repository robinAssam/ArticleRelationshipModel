"""Build the study A dataset: real judgments as retrieval questions.

Each item is one judgment that cites a WGBO article:

    question   the facts section, with every article reference masked
    answer     the corpus articles (data/processed/statutes.jsonl) the
               judgment cites anywhere in its text

Judgments without a recognisable facts section, or with a very short one
(often just "the facts are as established by the court below"), are left out.

The split into dev (for tuning) and test (touched once, at the end) is made
per group: judgments linked on rechtspraak.nl (first instance and appeal,
interim and final judgment) or with identical facts stay on the same side,
so the same facts never appear in both. The split is frozen: assignments
already in split.csv are kept on re-runs; new groups are assigned by a hash
of their ECLIs, so the outcome does not depend on processing order.

Writes:
    data/processed/study_a/queries.jsonl       full items incl. text (git-ignored)
    data/processed/study_a/split.csv           ids, split, answers (committed)
    outputs/tables/study_a_dataset_summary.json

Run after scripts/fetch_caselaw.py fetch and scripts/fetch_statutes.py:

    python scripts/build_study_a.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.caselaw import bw_citations, mask_citations, parse_judgment  # noqa: E402

XML_DIR = PROJECT_ROOT / "data" / "raw" / "case_law" / "xml"
CITATIONS = PROJECT_ROOT / "data" / "processed" / "caselaw_citations.csv"
STATUTES = PROJECT_ROOT / "data" / "processed" / "statutes.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "study_a"
SUMMARY = PROJECT_ROOT / "outputs" / "tables" / "study_a_dataset_summary.json"

MIN_WORDS = 80
DEV_FRACTION = 0.3


def _article_key(article: str) -> tuple[int, int, str]:
    book, num = article.split(":")
    digits = "".join(c for c in num if c.isdigit())
    return int(book), int(digits), num


def _groups(items: list[dict]) -> dict[str, str]:
    """ECLI -> group id (the smallest ECLI in the group), by union-find."""
    parent = {it["ecli"]: it["ecli"] for it in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_facts: dict[str, str] = {}
    for it in items:
        for rel in it["related"]:
            if rel in parent:
                union(it["ecli"], rel)
        first = by_facts.setdefault(it["question"], it["ecli"])
        union(it["ecli"], first)
    return {e: find(e) for e in parent}


def _hash_split(group: str) -> str:
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_FRACTION else "test"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = p.parse_args()

    corpus = {json.loads(line)["article_id"] for line in STATUTES.open(encoding="utf-8")}
    with CITATIONS.open(encoding="utf-8") as f:
        eclis = [r["ecli"] for r in csv.DictReader(f)
                 if r["set"] == "candidate" and int(r["n_wgbo_articles"]) > 0]

    items, dropped = [], Counter()
    for ecli in eclis:
        j = parse_judgment((XML_DIR / f"{ecli.replace(':', '_')}.xml").read_bytes())
        answers = sorted(bw_citations(j["text"]) & corpus, key=_article_key)
        if not j["facts"]:
            dropped["no facts section found"] += 1
            continue
        question, n_masked = mask_citations(j["facts"])
        if len(question.split()) < args.min_words:
            dropped[f"facts shorter than {args.min_words} words"] += 1
            continue
        if bw_citations(question) & corpus:
            dropped["article reference left after masking"] += 1
            continue
        summary, _ = mask_citations(j["summary"])
        items.append({
            "ecli": ecli, "date": j["date"], "court": j["court"], "procedure": j["procedure"],
            "related": j["related"], "facts_method": j["facts_method"],
            "n_words": len(question.split()), "n_masked": n_masked,
            "answers": answers, "question": question, "summary": summary,
        })

    group_of = _groups(items)
    split_file = OUT_DIR / "split.csv"
    frozen: dict[str, str] = {}
    if split_file.exists():
        with split_file.open(encoding="utf-8") as f:
            frozen = {r["ecli"]: r["split"] for r in csv.DictReader(f)}
    group_split: dict[str, str] = {}
    for it in items:  # a group keeps the split any of its members already had
        if it["ecli"] in frozen:
            group_split.setdefault(group_of[it["ecli"]], frozen[it["ecli"]])
    for it in items:
        g = group_of[it["ecli"]]
        it["group"] = g
        it["split"] = group_split.setdefault(g, _hash_split(g))
        if it["ecli"] in frozen and frozen[it["ecli"]] != it["split"]:
            print(f"warning: {it['ecli']} moved from {frozen[it['ecli']]} to {it['split']} "
                  "(it joined a group on the other side)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "queries.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    fields = ["ecli", "split", "group", "date", "court", "procedure", "facts_method",
              "n_words", "n_masked", "answers"]
    with split_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({**it, "answers": " ".join(it["answers"])})

    # Aggregate statistics only: the test items themselves are not printed
    per_split = Counter(it["split"] for it in items)
    answer_counts = Counter(a for it in items for a in it["answers"])
    words = sorted(it["n_words"] for it in items)
    summary = {
        "judgments_citing_wgbo": len(eclis),
        "items": len(items),
        "dropped": dict(dropped),
        "facts_found_by": dict(Counter(it["facts_method"] for it in items)),
        "split": dict(per_split),
        "groups": len({it["group"] for it in items}),
        "question_words": {"min": words[0], "median": words[len(words) // 2], "max": words[-1]},
        "questions_with_masked_references": sum(it["n_masked"] > 0 for it in items),
        "answers_per_item": dict(sorted(Counter(len(it["answers"]) for it in items).items())),
        "articles_used_as_answer": len(answer_counts),
        "corpus_articles": len(corpus),
        "most_frequent_answers": dict(answer_counts.most_common(15)),
        "min_words": args.min_words,
        "dev_fraction": DEV_FRACTION,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(eclis)} judgments citing the WGBO -> {len(items)} items "
          f"({per_split['dev']} dev, {per_split['test']} test, {summary['groups']} groups)")
    for reason, n in dropped.most_common():
        print(f"  dropped: {n} {reason}")
    print(f"  facts found by: {summary['facts_found_by']}")
    print(f"  question length (words): {summary['question_words']}")
    print(f"  questions where references were masked: "
          f"{summary['questions_with_masked_references']}")
    print(f"  answers per item: {summary['answers_per_item']}")
    print(f"  {len(answer_counts)} of {len(corpus)} corpus articles occur as an answer")
    print(f"-> {(OUT_DIR / 'queries.jsonl').relative_to(PROJECT_ROOT)}, "
          f"{split_file.relative_to(PROJECT_ROOT)}, {SUMMARY.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
