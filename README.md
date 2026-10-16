# Article Relationship Model

An atom-level knowledge graph over Dutch statutory law. Each article is
decomposed into the smallest units that carry a single legal meaning
("atoms"), annotated against a 9-label schema, and connected into a graph
that can be searched with a plain-language question in Dutch or English.

The project focuses on medical liability: the WGBO (the medical treatment
agreement, Book 7 Title 7.7.5 of the Dutch Civil Code, BW) and the Book 6
rules on liability and damages that medical cases fall back on.

**Demo app:** [Statute Navigator](https://statute-navigator.streamlit.app)
(password protected).


## Motivation

Statutes are written for lawyers, not machines. A single article stacks
duties, qualifying conditions and cross-references in dense prose. Treating
the article as one opaque block, the usual approach in statutory retrieval,
throws most of that structure away. This project makes the structure inside
each article explicit, so a question can be matched to the specific
requirement or consequence it concerns, with an auditable trail of why.


## Scope

| Layer | Contents | Size |
|---|---|---|
| **Annotated** (what the app and evaluation search) | 6 WGBO articles (7:446, 448, 453, 454, 455, 457) + 24 Book 6 articles | 30 articles, 80 atoms |
| **Official text** (downloaded, not yet annotated) | All of the WGBO (7:446–468) + Book 6 afd. 6.1.9, 6.1.10, 6.3.1, 6.3.2 | 87 articles |
| **Case law** (citing these articles) | Civil judgments on rechtspraak.nl that cite a WGBO article | 703 judgments |

The code is not specific to medical law: any part of the BW can be
downloaded and parsed. What limits coverage is annotation: an article can
only be searched once it has been split into atoms and labelled.

A synthetic corpus (the "Lunar Medical Code", 15 articles, 38 atoms) is kept
as a test fixture only and is excluded from all evaluation.


## Method

Every atom is annotated with the lab's 9-label schema: pre- vs
post-condition, actors, legal relations, acts, geographical domain,
temporal, explicit references, hierarchies, and residual terms. Atoms are
connected through four channels:

1. **Shared labels**, weighted by IDF (rare labels count more)
2. **Similar labels**, via sentence embeddings (e.g. *doctor* ~ *care provider*)
3. **The legal tree** (boek > titel > afdeling > artikel)
4. **Explicit references** between articles

A query is a set of labelled atoms. A plain-language question is first
decomposed into atoms by an LLM (Anthropic's Claude), reusing the corpus vocabulary.
Every corpus atom is scored with a normalised, IDF-weighted cosine
similarity (0 = nothing in common, 1 = identical labels), articles are
ranked by their best-matching atom, and a graph walk from the top matches
surfaces related articles, recording which link brought each one in.

The annotations in `data/annotations/` were drafted with an LLM and have not
yet been checked by a human (see *Known limitations*).


## Data sources

**Statute text (BWB).** `scripts/fetch_statutes.py` downloads the official
consolidated XML of BW Book 6 and Book 7 from the government's
Basiswettenbestand and turns it into one record per article: its place in
the legal tree, the date the current version took effect, the text per
paragraph (lid), and the cross-references it makes. Result: 87 articles in
`data/processed/statutes.jsonl` and 76 article-to-article references in
`data/processed/statute_references.csv`. The exact versions used, with
checksums, are recorded in `data/raw/statutes/sources.json`.

The same script checks the annotations against the official text
(`outputs/tables/annotation_text_check.csv`): of the 80 annotated atoms, 61
match word for word, 13 differ slightly, and 6 differ substantially, either
because the article was amended after annotation (7:454 in 2020, 7:457 in
2022, 6:106 in 2019) or because the annotated Dutch is not the statutory
wording.

**Case law (rechtspraak.nl).** `scripts/fetch_caselaw.py` searches the open
data of the Dutch judiciary for judgments that cite WGBO articles, which
provide the test set for retrieval: the facts of a case as the question,
the articles the judge cited as the answer.

- 291,252 published civil judgments (2000–2026) were indexed; 5,631 with
  medical terms in their title or summary were downloaded and searched for
  citations.
- **703 cite at least one WGBO article**, 265 cite two or more, and 186 also
  cite a Book 6 article in the corpus.
- 21 of the 23 WGBO articles are cited. Most cited: 7:453 (duty of care,
  202), 7:446 (treatment agreement, 168), 7:464 (medical examinations, 116),
  7:448 (informing the patient, 111), 7:462 (hospital liability, 108),
  7:457 (confidentiality, 99).
- The number is growing: about 20 a year around 2010, 40–60 a year since 2021.
- None of 1,000 randomly checked other judgments cites the WGBO, so the
  keyword filter misses little.

Only the derived citation table (`data/processed/caselaw_citations.csv`:
ECLI, date, court, cited articles) is in the repository; downloaded
judgments stay local.


## Results so far

Leave-one-out on the 80 real atoms: each atom in turn is the query, and the
system ranks the 30 articles.

| | MRR | hit@1 | hit@3 | hit@10 |
|---|---|---|---|---|
| Random ranking | 0.11 | 0.03 | 0.09 | 0.29 |
| Unnormalised scoring | 0.59 | 0.45 | 0.71 | 0.83 |
| **Normalised scoring (default)** | **0.61** | **0.49** | **0.73** | **0.82** |

11 of the 30 articles have a single atom and cannot be found once it is held
out, so the best possible score is 0.86. With 80 queries, differences of a
few points are within noise. Exact label matches carry most of the signal;
most errors are neighbouring articles in the same section. Details and
history: `docs/03_decisions_log.md`.


## The app: Statute Navigator

`app.py` is a Streamlit search page:

- a search box for a legal problem in Dutch or English, with example
  questions and preset tag queries that need no LLM;
- how the question was split into atoms;
- ranked articles with a breadcrumb, a snippet and a 0–1 match score, plus a
  "Why this score?" breakdown (matched, similar and unmatched labels);
- a score chart, two graph views (focused and network) and the reasoning trail;
- (i) buttons explaining every number in plain language.

Typed questions use the Anthropic API (a few cents each, cached); everything
else runs locally. The hosted version runs on Streamlit Community Cloud from
`deploy/streamlit_cloud/streamlit_app.py`, behind a password.


## Repository layout

    app.py                         Statute Navigator (Streamlit).

    docs/
      00_project_overview.md       Problem statement and planned relations.
      01_data_model_schema.md      Entities and relations.
      02_annotation_schema.md      The 9-label schema and its rules.
      03_decisions_log.md          Dated notes on design decisions and results.

    data/
      annotations/                 One JSON per article; flat list of atoms.
      raw/statutes/                Official BWB XML (git-ignored) + sources.json.
      raw/case_law/                Downloaded judgments (git-ignored).
      processed/                   statutes.jsonl, statute_references.csv,
                                   caselaw_citations.csv.

    src/
      annotation/schema.py         Atom dataclass and field definitions.
      data/atom_table.py           Loading, edges, weighting, scoring, evaluation.
      data/hierarchy.py            Legal tree and reference extraction.
      data/reasoning.py            Graph walker with an auditable trail.
      data/query.py                Question -> query atoms (Claude) -> ranked articles.
      data/statutes.py             Parse official BWB XML into article records.
      data/generate_obsidian.py    Optional Obsidian vault export.

    scripts/
      run_eval.py                  Article-level evaluation -> outputs/tables/.
      fetch_statutes.py            Download and parse the statute text.
      fetch_caselaw.py             Find judgments citing WGBO articles.
      deploy_hf_space.py           Optional Hugging Face Spaces deployment.

    deploy/                        Streamlit Cloud and Hugging Face configs.
    tests/test_pipeline.py         27 tests (synthetic corpus, parser, queries).
    notebooks/                     Exploration (01) and experiments (02).
    outputs/tables/                Evaluation results with their config.


## Setup

    python -m venv .venv
    .venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt

For plain-language questions, put your Anthropic API key in
`.streamlit/secrets.toml` (git-ignored):

    ANTHROPIC_API_KEY = "sk-ant-..."

Common commands, from the repo root:

    streamlit run app.py                                   # the app
    python -m unittest discover tests                      # tests
    python scripts/run_eval.py                             # evaluation tables
    python scripts/fetch_statutes.py                       # refresh statute text
    python scripts/fetch_caselaw.py index|fetch|sample|report   # case law


## Known limitations

- **Annotations are unchecked LLM drafts**, and some are based on older
  versions of the law. They must be redone from the official text and
  checked by a human before serving as ground truth.
- **Coverage is 30 annotated articles.** The other 57 downloaded articles,
  including often-cited ones such as 7:450 (consent) and 7:462 (hospital
  liability), are not yet searchable.
- **Annotations are in English, the law and case law in Dutch.** Questions
  are bridged by the LLM; the annotation language is still to be decided.
- The current evaluation queries are corpus atoms, not real questions.
  Real judgments as queries are the next step.


## Next steps

- Check a sample of the 703 judgments by hand: how many give a clear facts
  section and a substantive citation?
- Annotate the full 87-article corpus from the official text (LLM draft plus
  a human-checked sample to measure agreement).
- Evaluate on real judgments: compare BM25, dense retrieval, and dense
  retrieval combined with the atom scores and the graph.
- A qualitative study of alternative legal grounds on a few real cases.


