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

The corpus is deliberately small. It covers one area of law, medical
liability, in three layers:

| Layer | Contents | Size |
|---|---|---|
| **Annotated** (what the app and evaluation search) | 6 WGBO articles (7:446, 448, 453, 454, 455, 457) + 24 Book 6 articles | 30 articles, 80 atoms |
| **Official text** (downloaded, not yet annotated) | All of the WGBO (7:446–468) + Book 6 afd. 6.1.9, 6.1.10, 6.3.1, 6.3.2 | 87 articles |
| **Case law** (citing these articles) | Civil judgments on rechtspraak.nl that cite a WGBO article | 703 judgments |

In other words: the app can currently search 30 articles. The rest of the
Civil Code (BW), such as the other ~870 articles of Book 7 (sale, rent,
employment and so on), is not included. The code is not specific to medical
law, though. What limits coverage is annotation: an article can only be
searched once it has been split into atoms and labelled. See *Scaling up*
below.

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

**Study A test set.** `scripts/build_study_a.py` turns these judgments into
question–answer pairs for measuring retrieval:

- **Question:** the facts section of the judgment ("De feiten"). Article
  numbers in it are replaced by `[ARTIKEL]` (and "WGBO" by `[WET]`), because
  judges often name the article in the facts, which would give the answer
  away.
- **Answer:** the articles of the 87-article corpus that the judgment cites.
- **Result:** 391 usable judgments. 254 were left out because no facts
  section could be found, and 58 because the facts section was very short
  (often only "the facts are as established by the court below").
  Questions are a median of 663 words; 155 have one correct article, the
  rest two to nine. 49 of the 87 articles occur as an answer.
- **Split:** 125 dev (for tuning) and 266 test (used once, at the end).
  Judgments about the same case (first instance and appeal) stay on the
  same side, and the split stays fixed when the script is re-run.

The split and answers are in `data/processed/study_a/split.csv`; the
questions themselves (`queries.jsonl`) are rebuilt locally from the
downloaded judgments. Counts: `outputs/tables/study_a_dataset_summary.json`.


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

**Baselines on real judgments (dev set, 125 questions, 87 articles).** These
systems use no atoms; they are the bar the atom system has to beat.

| | MRR | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---|---|---|---|---|
| Random ranking | 0.10 | 0.06 | 0.11 | 0.23 | 0.06 |
| BM25 (keyword search) | 0.33 | 0.27 | 0.46 | 0.61 | 0.29 |
| Dense (BGE-M3 embeddings) | 0.31 | 0.31 | 0.42 | 0.61 | 0.28 |
| Hybrid: BM25 + dense, z-score fusion | 0.37 | 0.29 | 0.46 | 0.66 | 0.31 |
| Hybrid: BM25 + dense, rank fusion (RRF) | 0.38 | 0.30 | 0.46 | 0.66 | 0.32 |
| RRF + spreading along citations | 0.37 | 0.34 | 0.50 | 0.63 | 0.34 |
| RRF + spreading within legal sections | 0.39 | 0.32 | 0.49 | 0.68 | 0.33 |
| **RRF + spreading along both** | **0.40** | **0.34** | **0.49** | **0.69** | **0.35** |

- **MRR:** 1 / position of the first correct article, averaged.
- **recall@k:** share of the cited articles in the top k.
- **nDCG@10:** how well all correct articles are placed in the top 10
  (1 = all at the top).

BM25 and dense cannot be told apart on the dev set: the difference is not
significant (paired permutation test, p > 0.2), but with 125 questions the
uncertainty is large (recall@10: dense between 13.5 points worse and 4 points
better). This is in line with earlier work: dense models that are not trained
on legal questions often do no better than BM25. The two are not
copies of each other: they share only 3 of their top-10 articles on average,
and fail on different articles. Dense finds
hospital liability (7:462) and consent (7:450) far more often, BM25 the duty
of care (7:453). Neither ever finds 6:74 (failure to perform) or 6:162 (tort)
in the top 10, and dense never finds 7:453, the most cited article. These
are short, abstract articles that share neither words nor obvious meaning
with the story of a case: the gap the atoms and the graph are meant to close.

The **hybrid** merges the BM25 and dense rankings. Rank fusion (RRF) adds
1/(60 + rank) from each system and needs no tuning. It puts the first
correct article higher (MRR 0.38 vs 0.33 for BM25, p = 0.06; vs 0.31 for
dense, p = 0.01) but finds no more of the cited articles in the top 10
(recall@10 0.46, same as BM25). It does not close the gap either: 6:74 and
6:162 are still never in the top 10, and 7:453 only 4 times in 44. RRF is the
baseline the atom system has to beat, and the first stage that picks the
shortlist for atom re-ranking.

**Spreading over links between articles** (no atoms yet). After the hybrid,
each article passes part of its score to the articles it is linked to, so
an article linked to a strong match moves up. Two kinds of link: explicit
citations in the law text (48 links; 47 of the 87 articles have none), and
sections of the legal tree (articles in the same afdeling, the structure
Gijs's G-DSR uses). How strongly to spread was chosen on the dev set, so
these dev gains are optimistic until confirmed on the test set.

- Both links together give the best result (nDCG@10 +0.03 over the hybrid,
  p = 0.005). Sections alone help a little (+0.015, p = 0.02); citations alone
  are not significant.
- Judges do cite linked articles together, about four times more often than
  chance, but it concerns few pairs (17 of 320 in dev).
- The blind spots stay: 6:74 and 6:162 are still never in the top 10, and
  7:453 drops to 3 in 44. 6:162 and 7:453 have no citation links at all, and
  a section boost lifts all their neighbours equally. Reaching them needs
  links the law text does not make explicit, which is what the atom graph
  (shared actors, acts, conditions) adds.


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
                                   caselaw_citations.csv, study_a/split.csv.

    src/
      annotation/schema.py         Atom dataclass and field definitions.
      data/atom_table.py           Loading, edges, weighting, scoring, evaluation.
      data/hierarchy.py            Legal tree and reference extraction.
      data/reasoning.py            Graph walker with an auditable trail.
      data/query.py                Question -> query atoms (Claude) -> ranked articles.
      data/statutes.py             Parse official BWB XML into article records.
      data/caselaw.py              Parse judgments, find the facts, mask citations.
      similarity/retrieval.py      Baseline systems (BM25, dense, hybrid) and metrics.
      graph/propagation.py         Score spreading over citation and section links.
      data/generate_obsidian.py    Optional Obsidian vault export.

    scripts/
      run_eval.py                  Article-level evaluation -> outputs/tables/.
      fetch_statutes.py            Download and parse the statute text.
      fetch_caselaw.py             Find judgments citing WGBO articles.
      build_study_a.py             Judgments -> study A questions, answers, split.
      run_baselines.py             Baselines and graph systems on study A -> outputs/tables/.
      deploy_hf_space.py           Optional Hugging Face Spaces deployment.

    deploy/                        Streamlit Cloud and Hugging Face configs.
    tests/test_pipeline.py         39 tests (synthetic corpus, parsers, queries, masking, metrics, graph).
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
    python scripts/build_study_a.py                        # study A test set
    python scripts/run_baselines.py                        # baselines (dev)


## Scaling up

**More statute text** The download script reads the official
government XML, which covers all Dutch law. Adding another book of the BW is
a one-line change.

**More test data** To measure whether the system finds
the right article, we need questions with known answers. Real court
judgments provide these: the facts of the case are the question, and the
articles the judge cited are the answer.  291,252 published civil judgments (2000–2026) have already 
been indexed and can extract which BW articles each one cites. For now, only search the medical ones are searched(703
judgments). Dropping that filter would give test cases for any part of the
BW, likely thousands, without anyone labelling them by hand. As far as I
know, no such test set exists yet for Dutch statutory law.

**More annotations** An LLM can split and
label thousands of articles quickly and at low cost. A human cannot check
them all, so quality is measured on a hand-checked sample (see *Next
steps*).

**Search speed** Comparing a question with
every atom is fast up to roughly 100,000 atoms. For larger corpora the
search works in two steps: an ordinary text search first picks the most
likely few dozen articles, and the atom comparison then re-orders only
those. This two-step setup is one of the systems we will test anyway (see
*Next steps*), so it is planned rather than an afterthought.

**labels do not scale yet** The 80 atoms use 277 different
labels, and 199 of those occur only once. Every new article adds new labels,
so in a large corpus two atoms that mean the same thing would often carry
different labels and fail to match. Two fixes are needed: a standard list of
labels (for example one fixed term for *doctor* / *care provider* /
*physician*), and giving the LLM only the labels relevant to a question
instead of the whole list, which would not fit at scale.

**How the next steps prepare for this.** The baselines (ordinary text search
and embedding search) tell us how much the atoms add before we invest in
annotating more law. The two-step system is the design that scales. And the
automatic annotation pipeline, checked against human annotations, is what
makes annotating a whole book feasible.


## Known limitations

- **Annotations are unchecked LLM drafts**, and some are based on older
  versions of the law. They must be redone from the official text and
  checked by a human before serving as ground truth.
- **Coverage is 30 annotated articles.** The other 57 downloaded articles,
  including often-cited ones such as 7:450 (consent) and 7:462 (hospital
  liability), are not yet searchable.
- **Annotations are in English, the law and case law in Dutch.** Questions
  are bridged by the LLM; the annotation language is still to be decided.
- The atom system has so far only been evaluated with corpus atoms as
  questions, not real ones. The real-judgment test set exists (391
  questions), but only the systems without atoms have been run on it, on the
  dev part. The test part has not been touched.
- The spreading strengths of the graph systems were chosen on the dev set,
  so their dev scores are optimistic until the test run.
- The test set's answers are every corpus article a judgment cites, including
  ones cited only in passing. A hand check of a sample is still to do.


## Next steps

1. Build the automatic annotation pipeline and annotate the full 87-article
   corpus from the official text, with a human-checked sample to measure
   agreement. The pipeline is developed on the 30 articles already
   annotated; the other 57 are annotated once the annotation protocol is
   agreed. Step 2 waits for this: only 45 of the 125 dev questions have all
   their answers among the 30 annotated articles.
2. Add the atom scores and the atom graph on top of the baselines and compare.
3. Check a sample of the judgments by hand.
4. Run every system once on the test set, with the settings fixed on dev.
5. A qualitative study of alternative legal grounds on a few real cases.


