# Article Relationship Model

An atom-level knowledge graph over Dutch statutory law. Each article is
decomposed into the smallest units that carry a single legal meaning ("atoms"),
annotated against a 9-label schema, and connected into a graph that can be
navigated and queried in natural language.

Anchored on the WGBO (Book 7, Title 7.7.5 BW) and extended into the Book 6
obligations articles that are typically referenced from medical malpractice
cases.


## Motivation

Statutes are written for lawyers, not machines. A single article stacks
duties, qualifying conditions, and cross-references in dense prose. Treating
the article as one opaque unit — the usual approach in statutory retrieval —
throws away most of that structure. This project decomposes articles into
atoms, so the internal structure of each article becomes explicit and
addressable.


## Method

Each article is annotated by hand against the lab's 9-label extraction schema
(actors, legal relations, acts, geographical domain, temporal, explicit
references, hierarchies, residual, plus pre- vs post-condition). Annotated
atoms are connected through four edge channels:

1. Shared annotation tags (exact set intersection, weighted by IDF)
2. Semantic similarity between tag values via sentence embeddings
3. Hierarchical containment through the Dutch legal tree
   (rechtsgebied → wetboek → boek → titel → afdeling → artikel)
4. Explicit cross-article references extracted from atom text
   ("Without prejudice to article 455" becomes a directed edge)

A classifier scores an atom-shaped query against every corpus atom and
returns a predicted position in the tree. On top of that, a chained-reasoning
walker starts at the classifier's top matches and expands outward through the
graph, surfacing related atoms via reference and shared-tag edges — so a query
about record destruction anchors on 7:455 and then follows the reference edge
back to 7:454's retention duty, without the user asking for it.

Evaluation is leave-one-out, reported at each hierarchy level rather than only
at the article level.


## Corpus

- WGBO articles 7:454 and 7:455 (11 atoms)
- Book 6 fallbacks: 6:74, 6:75, 6:76, 6:162, 6:170 (12 atoms)
- Synthetic corpus (Lunar Medical Code): 15 articles, 38 atoms

The synthetic corpus is used to exercise the pipeline without leaking real
statutory text into training and to keep automated evaluation honest.


## Repository layout

    docs/
      annotation_schema.md         The 9-label schema and its rules.
      decisions_log.md             Dated notes on design decisions.
      supervisor_notes/            Meeting notes, one file per meeting.

    data/
      raw/statutes/                Source article texts. Read-only.
      annotations/                 One JSON per article; flat list of atoms.
      processed/                   Derived tables (edges, embeddings).

    src/
      annotation/schema.py         Atom dataclass and field definitions.
      data/atom_table.py           Loaders, edges, weighting, classifier.
      data/hierarchy.py            Legal tree, containment and reference edges.
      data/reasoning.py            Chained-reasoning walker and trace formatting.

    notebooks/
      01_explore_atoms.ipynb       End-to-end exploration and evaluation.

    outputs/                       Figures, tables, exported artefacts.


## Setup

    python -m venv .venv
    source .venv/bin/activate           # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

The notebook is the entry point. It loads the annotations, builds the edge
tables, runs the classifier, walks the chained-reasoning demos, and produces
the figures used in the write-up.


## Conventions

Files under data/raw/ are the source of truth and are not edited in place.
Derived data lives under data/processed/. Notebooks are for exploration;
stable code moves into src/.


## Status

Implemented:
- Atom-level annotation for 7 real articles (WGBO + Book 6) and a 15-article
  synthetic corpus
- Four-channel edge construction: shared tags (IDF-weighted), semantic
  similarity, hierarchical containment, and explicit cross-article references
- Legal tree parsing (rechtsgebied → article) with per-level LOO evaluation
- Chained-reasoning walker that traverses reference and tag edges from
  anchor atoms, with auditable per-step provenance
- Encoder comparison across three sentence transformers, including two
  legal-domain models
- Obsidian vault export of the annotated graph

In progress:
- Streamlit interface for interactive queries and subgraph visualisation
- Extending the corpus to a wider cluster of medical-malpractice articles
- Filter-then-semantic retrieval, following the pattern in G-DSR
- A short technical write-up