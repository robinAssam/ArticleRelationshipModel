# Article Relationship Model
A network model of relationships across statutory articles in Dutch medical
malpractice and consent law, anchored on Article 7:454 BW (*dossierplicht*).

**Internship project — Maastricht Law and Tech Lab, summer 2026.**

## What this project does

Given a statutory article whose requirements a claimant cannot fully satisfy,
the model identifies alternative articles whose requirements (a) protect a
similar interest and (b) share lexical, semantic, or structural features —
such that they constitute plausible alternative legal grounds.

Anchor article: **7:454 BW (dossierplicht)**.
Neighbourhood: WGBO articles 7:451–7:458 + Book 6 fallbacks (6:74, 6:162,
6:170) + optional GDPR overlap (Arts. 5, 9, 15, 17).

## Method

Each article is decomposed into atomic boolean statements (one per
requirement or duty), each annotated against the lab's 9-label schema
(actors, legal relations, acts, geographical domain, temporal, explicit
references, hierarchies, residual). The annotated atoms become structured
tuples. Two atoms — and by aggregation two articles — are *linked* when they
share field values (same actor, same act, etc.) or when one explicitly
references the other. The resulting multi-relational graph is validated
against rechtspraak.nl judgments that plead grounds *primair / subsidiair*.

## Folder structure

```
.
├── docs/                      # all written documentation
│   ├── 00_project_overview.md
│   ├── 01_data_model_schema.md
│   ├── 02_annotation_schema.md
│   ├── 03_decisions_log.md
│   └── supervisor_notes/      # raw notes from each meeting
├── data/
│   ├── raw/                   # source texts — NEVER edit in place
│   │   ├── statutes/
│   │   └── case_law/
│   ├── annotations/           # tuple files (one per article)
│   ├── processed/             # flat atom table, derived datasets
│   └── validation/            # primair/subsidiair ground truth
├── src/                       # reusable Python modules
│   ├── data/                  # fetchers, parsers
│   ├── annotation/            # tuple schema, validators
│   ├── similarity/            # matching across atoms
│   └── graph/                 # graph construction, queries
├── notebooks/                 # exploration, throwaway analysis
├── outputs/                   # figures, tables, written reports
│   ├── figures/
│   ├── tables/
│   └── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate            # Windows

# install dependencies
pip install -r requirements.txt

# verify
python -c "import pandas, spacy, networkx; print('ok')"
```

## Workflow conventions

- **Never edit files in `data/raw/`** — they are the source of truth. All
  derived data lives in `data/processed/` or `data/annotations/`.
- **One notebook per exploration**, prefixed with a number
  (`01_explore_articles.ipynb`, `02_build_atom_table.ipynb`). Notebooks are
  for thinking; promote stable code into `src/`.
- **Document every decision** in `docs/03_decisions_log.md` with a date.
  Future-you and the supervisor need this.
- **Supervisor notes go in `docs/supervisor_notes/`** — one file per meeting,
  filename `YYYY-MM-DD_topic.md`. Verbatim is best.
- **Commit often, push at end of day** to GitHub. Even messy commits beat
  losing a day of work.

## Status

- [x] Anchor article selected: 7:454 BW
- [x] Boolean decomposition of 7:454
- [ ] Lid 3 fully annotated against 9-label schema (next)
- [ ] Tuple schema validated with supervisor
- [ ] Neighbourhood articles fetched
- [ ] Atom table populated
- [ ] Cross-tuple links computed
- [ ] Graph constructed
- [ ] Validation against case law
- [ ] Final report and visualisation
