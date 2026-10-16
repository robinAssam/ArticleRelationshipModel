# Article Relationship Model

An atom-level knowledge graph over Dutch statutory law. Each article is
decomposed into the smallest units that carry a single legal meaning ("atoms"),
annotated against a 9-label schema, and connected into a graph that can be
navigated and queried in natural language.

nchored on the WGBO (Book 7, Title 7.7.5 BW)
and extended into the Book 6 obligations articles that are typically referenced
from medical malpractice cases.


## Motivation

Statutes are written for lawyers, not machines. A single article stacks
duties, qualifying conditions, and cross-references in dense prose. Treating
the article as one opaque unit  the usual approach in statutory retrieval 
throws away most of that structure. This project decomposes articles into
atoms, so the internal structure of each article becomes explicit and
addressable.


## Method

Each article is annotated by hand against the lab's 9-label extraction schema
(actors, legal relations, acts, geographical domain, temporal, explicit
references, hierarchies, residual, plus pre- vs post-condition). Annotated
atoms are connected through three edge channels:

1. Shared annotation tags (exact set intersection, weighted by IDF)
2. Semantic similarity between tag values via sentence embeddings
3. Hierarchical containment through the Dutch legal tree
   (rechtsgebied - wetboek - boek - titel - afdeling - artikel)

A classifier takes an atom-shaped query, scores every corpus atom with a
normalised (weighted cosine) tag similarity, and ranks articles by their
best-matching atom. A natural-language question can be turned into query
atoms with an LLM first (`src/data/query.py`). Evaluation is leave-one-out at
article level (MRR, hit@k, against a random baseline), and also per level of
the legal tree.


## Corpus

- WGBO articles 7:454 and 7:455 (11 atoms)
- Book 6 fallbacks: 6:74, 6:75, 6:76, 6:162, 6:170 (12 atoms)
- Synthetic corpus (Lunar Medical Code): 15 articles, 38 atoms

The synthetic corpus is a test fixture: `tests/test_pipeline.py` runs every
pipeline stage on it. It is excluded from evaluation, which uses the real
articles only (see the 15/10/2026 entry in `docs/03_decisions_log.md`).


## Repository layout

    docs/
      annotation_schema.md         The 9-label schema and its rules.
      decisions_log.md             Dated notes on design decisions.
      supervisor_notes/            Meeting notes, one file per meeting.

    data/
      raw/statutes/                Official BWB XML (git-ignored) + sources.json. Read-only.
      annotations/                 One JSON per article; flat list of atoms.
      processed/                   statutes.jsonl (87 articles), statute_references.csv,
                                   caselaw_citations.csv (judgments -> cited articles).

    src/
      annotation/schema.py         Atom dataclass and field definitions.
      data/atom_table.py           Loaders, edges, weighting, scoring, evaluation.
      data/hierarchy.py            Legal tree and taxonomy lookup.
      data/reasoning.py            Chained-reasoning graph walker.
      data/query.py                Question text -> query atoms (Claude) -> articles.
      data/statutes.py             Parse official BWB statute XML into article records.

    scripts/
      run_eval.py                  Article-level evaluation; writes outputs/tables/.
      fetch_statutes.py            Download WGBO + Book 6 text from the BWB.
      fetch_caselaw.py             Find judgments citing WGBO articles (rechtspraak.nl).

    tests/
      test_pipeline.py             Pipeline tests on the synthetic corpus.

    notebooks/
      01_explore_atoms.ipynb       End-to-end exploration and evaluation.
      02_experiments.ipynb         Encoder comparison, alpha/normalisation sweep.

    outputs/                       Figures, tables, exported artefacts.
      tables/                      Evaluation results with their config (JSON).


## Setup

    python -m venv .venv
    source .venv/bin/activate           # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

The notebook is the entry point. It loads the annotations, builds the edge
tables, runs the classifier, and produces the figures used in the write-up.

Run the tests from the repo root:

    python -m unittest discover tests

Reproduce the evaluation tables (results and config land in outputs/tables/):

    python scripts/run_eval.py
    python scripts/run_eval.py --name scoring_sweep --sweep

Natural-language queries call the Anthropic API and need credentials
(`ANTHROPIC_API_KEY`, or `ant auth login`).


## Conventions

Files under data/raw/ are the source of truth and are not edited in place.
Derived data lives under data/processed/. Notebooks are for exploration;
stable code moves into src/.

## Status

Implemented:
- Atom-level annotation for 7 real articles and a 15-article synthetic corpus
- Three-channel edge construction (exact, IDF-weighted, semantic)
- Legal tree parsing and per-level LOO evaluation
- Encoder comparison across three sentence transformers, including two
  legal-domain models
- Obsidian vault export of the annotated graph

In progress:
- Extending the corpus to a wider cluster of medical-malpractice articles
- Filter-then-semantic retrieval, following the pattern in G-DSR
- BM25 as an additional retrieval channel for baseline comparison
- A Streamlit interface for interactive queries
- A short technical write-up


