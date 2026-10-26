## 05/07/2026 — Encoder: keep `all-mpnet-base-v2` as default

Compared three sentence-transformer models on the current corpus:
`all-mpnet-base-v2` (general English), `nlpaueb/legal-bert-base-uncased`
(English legal), and `joelniklaus/legal-xlm-roberta-base` (multilingual
legal). Legal-BERT produces a degenerate embedding space where most tags
match most others (~4,800 pairs at threshold 0.55, with no downstream
accuracy gain). Legal-XLM-R adds mild coverage but no accuracy improvement.

Decision: keep `all-mpnet-base-v2`. Revisit when the corpus is larger or
when a bilingual (Dutch-English) legal-domain sentence-transformer becomes
available.

See `notebooks/02_experiments.ipynb`, Section 1.


## 05/07/2026 — Classifier: alpha = 0.5, top_k = 3 as defaults

Grid search over alpha ∈ {0, 0.25, 0.5, 0.75, 1.0} and top_k ∈ {3, 5}
on leave-one-out evaluation. Accuracy plateaus from alpha ≥ 0.25 — the
exact-match channel does most of the useful work, and additional
semantic weight beyond a small floor produces no further improvement.

Decision: alpha = 0.5, top_k = 3. Revisit if IDF weighting or the
semantic scoring formula changes materially.

See `notebooks/02_experiments.ipynb`, Section 2.


## 12/09/2026 — Filter-then-semantic verified on current corpus

Ran LOO evaluation with and without `filter_candidates` on 61-atom corpus,
alpha=0.5, top_k=3, filter_fields=['actors', 'acts'].

Baseline (no filter):    top1=0.22, top3=0.33, MRR=0.28
With filter:             top1=0.30, top3=0.41, MRR=0.35

Filter frequently returns zero survivors on this small corpus (the query
has no actor/act overlap with any atom); when that happens the code falls
back on the full corpus per the safety default. Even with those fallbacks
counted in the pooled metrics, the filter improves top-1 by ~8 points.

See notebooks/01_explore_atoms.ipynb, sections 7.5 and 7.6.

## 15/10/2026 — Synthetic corpus removed from evaluation

The Lunar Medical Code (38 atoms) no longer takes part in leave-one-out
evaluation, either as queries or as candidates. Its vocabulary is disjoint
from Dutch law, so it inflated coarse-level accuracy (rechtsgebied/wetboek
mostly measured "real vs synthetic") and added distractors at article level.
It is kept as a pipeline fixture in `tests/test_pipeline.py`. IDF weights for
evaluation are computed on the real corpus only.

`evaluate_classifier_loo` and `evaluate_at_levels` now drop synthetic atoms
by default (`include_synthetic=True` to opt back in), and
`evaluate_classifier_loo` takes `filter_fields`.

Re-run on 80 real atoms / 30 articles, alpha=0.5, top_k=3:

    Before (118 atoms incl. synthetic): top1=0.36, top3=0.50, MRR=0.42
    Real only, no filter:               top1=0.48, top3=0.65, MRR=0.55
    Real only, filter (actors, acts):   top1=0.51, top3=0.64, MRR=0.56

The earlier filter gain (0.22 -> 0.30) does not hold on the current corpus;
the filter is now roughly neutral. Rechtsgebied/wetboek accuracy is trivially
1.0 because every real article is in the BW. Encoder and alpha/top_k
decisions are unchanged on the real corpus (mpnet still best; alpha=0.5).


## 15/10/2026 — Normalised scoring, article-level evaluation, text queries

**Scoring.** `classify_atom` now normalises both channels with
`normalize="cosine"` (default): each tag is weighted by IDF, and the score is
the weighted cosine (Ochiai) similarity of the query's and candidate's tag
sets. The semantic channel uses the same weights and denominator, so `alpha`
mixes like with like. Scores are scaled so 1.0 = identical tags (alpha >= 0.5).
`normalize="none"` reproduces the old raw sums.

**Evaluation.** New headline metric `evaluate_article_retrieval`: LOO where
the *articles* are ranked (an article scores as its best atom), IDF is
recomputed inside each fold, ties count by their expected value, and a random
baseline is reported. 11 of the 30 real articles have a single atom, so 11 of
80 queries cannot be answered (ceiling MRR = hit@k = 0.8625).

Real corpus, 80 queries, mpnet tag similarity at 0.55, article = max atom:

    old scoring (none), alpha=0.5:   MRR=0.593  hit@1=0.448  hit@3=0.713
    new scoring (cosine), alpha=0.5: MRR=0.611  hit@1=0.494  hit@3=0.726
    random ranking:                  MRR=0.115  hit@1=0.029  hit@3=0.086

The gain is about 4 queries out of 80: real but within noise at this size.
Fixing the IDF leak changes MRR by 0.002. Semantic-only (alpha=0) is
MRR 0.23, close to random; exact matches carry the signal. `agg="sum"` is
worse than `max` in every setting. alpha=0.75 gives MRR 0.621, but the default
stays 0.5 rather than tuning on the evaluation set. Results:
`outputs/tables/article_retrieval_*`, `outputs/tables/scoring_sweep_*`.

**Text queries.** `src/data/query.py` decomposes a Dutch or English question
into query atoms with Claude Opus 5.5 (structured JSON output, the corpus tag
vocabulary in the prompt so exact matches can fire) and sums each article's
score over the query atoms. Not yet evaluated — that is study A.


## 16/10/2026 — Official statute text downloaded (BWB)

`scripts/fetch_statutes.py` downloads the consolidated XML of BW Book 6
(BWBR0005289, version 2026-07-16_0) and Book 7 (BWBR0005290, version
2026-07-01_1) from the government BWB repository and parses it with
`src/data/statutes.py`. Corpus scope: WGBO (afd. 7.7.5) and Book 6 afd.
6.1.9, 6.1.10, 6.3.1 and 6.3.2: 87 articles in force (up from 30 annotated),
with headings, in-force dates, text per lid, and 76 machine-readable
cross-references (63 between corpus articles; the BWB `intref` elements).
Versions and checksums are recorded in `data/raw/statutes/sources.json`; the
raw XML is git-ignored and can be re-downloaded by pinning the version.

Checking the annotations against the official text
(`outputs/tables/annotation_text_check.csv`): of 80 real atoms, 61 are
verbatim, 13 differ slightly (different cut, repeated words), and 6 differ
substantially. Two causes:
- the law changed after the text used for annotation: 7:454 (2020), 7:457
  (2022), 6:106 (2019: now one paragraph with a list, old lid 2 gone);
- the annotated Dutch is not the statutory wording (e.g. 6:101 lid 2,
  unchanged since 1992), consistent with LLM-drafted annotations.
Consequence: re-annotate from the official text before using the
annotations as ground truth.


## 16/10/2026 — Case law available for study A (rechtspraak.nl)

`scripts/fetch_caselaw.py` indexed every published civil-law judgment with
text on rechtspraak.nl open data (291,252, 2000–2026), selected 5,631 with
medical terms in title or summary, downloaded them and extracted Book 6/7
BW citations from the full text (`data/processed/caselaw_citations.csv`;
judgment texts stay local, git-ignored).

- 703 judgments cite at least one WGBO article; 265 cite two or more;
  186 also cite a Book 6 article in the corpus. Growing over time
  (~20/year around 2010, 40–60/year since 2021).
- Variety: 21 of the 23 WGBO articles are cited; 15 in >= 10 judgments,
  8 in >= 30. Most cited: 7:453 (202), 7:446 (168), 7:464 (116),
  7:448 (111), 7:462 (108), 7:457 (99), 7:450 (87), 7:454 (68).
- Courts: 548 Rechtbank, 124 Gerechtshof, 16 Hoge Raad.
- Spot checks: citations are genuine and usually substantive.
- Recall of the keyword filter: 0 of 1,000 random non-candidates cite the
  WGBO. That bounds the miss rate below ~0.3% (rule of three), i.e. at
  most a few hundred judgments across the remaining 285k; likely far fewer.

Conclusion: study A is feasible as planned (well above the ~300 threshold),
pending a manual check of how many judgments yield a clean facts section
and a substantive (not merely procedural) citation. Note that several often
cited articles (7:450 consent, 7:462 hospital liability, 7:464 examinations)
are not yet annotated: the full 87-article corpus needs annotating.


## 19/10/2026 — Study A dataset built

`scripts/build_study_a.py` (logic in `src/data/caselaw.py`) turns the 703
judgments citing a WGBO article into retrieval items:

- **Question** = the facts section. Found by section title ("De feiten",
  "Feiten", "De vaststaande feiten", "Feitelijke achtergrond": 293) or, for
  judgments without section markup, by a numbered "N. De feiten" heading in
  the running text up to heading N+1 (98).
- **Masking.** Every statutory article reference in the question becomes
  `[ARTIKEL]` (any law, any form: "art. 7:453 BW", "artikelen 7:446 en 7:453
  BW", "artikel 453 van Boek 7 BW", bare "7:453 BW"), and "WGBO" / its full
  name becomes `[WET]`. Bare `n:nnn` is masked only when a law follows or it
  is Book 6/7 with article >= 60, so clock times ("19:20 uur") stay. 72
  questions contained references. After masking no corpus citation is left
  in any question (checked with the same extractor that produces the answers).
- **Answer** = corpus articles (statutes.jsonl, 87) cited anywhere in the
  judgment. Includes citations in passing; to be checked by hand on a sample.
- **Dropped:** 254 without a recognisable facts section (interim/final
  judgments referring to earlier ones, Hoge Raad, unusual structure), 58 with
  facts under 80 words (mostly "the facts as established below apply").
- **Split** per group (union-find over rechtspraak `dcterms:relation` links and
  identical facts), by a hash of the group id: dev 125, test 266, 388 groups.
  Frozen: existing assignments in split.csv are kept on re-runs.

Result: 391 items, question median 663 words (80–4,788), 155 with one answer,
236 with two to nine; 49 of the 87 corpus articles occur as an answer. Most
frequent: 7:453, 7:446, 7:457, 7:448. Long questions exceed the input limit of
some dense encoders (e.g. 512 tokens for multilingual-e5); truncation or
chunking is a per-system choice, not part of the dataset.

Not done: Hoge Raad judgments mostly state facts inside "beoordeling van het
middel" ("In cassatie kan worden uitgegaan van...") and are not recovered.


## 19/10/2026 — Baselines on the study A dev set

`scripts/run_baselines.py` (systems and metrics in `src/similarity/retrieval.py`).
Every system produces a questions x articles score matrix; `evaluate()` turns
it into recall@k, MRR and nDCG@10 per question, averaging over random orders
when articles tie. Article documents are the statute text only (no number, no
headings). BM25: rank-bm25 Okapi (k1 1.5, b 0.75), lowercased words minus a
small Dutch stopword list, no stemming. Dense: BAAI/bge-m3 via
sentence-transformers, max 8,192 tokens (dev questions: median ~1,100 tokens,
max ~6,750, so none truncated), cosine of normalised embeddings; 14 minutes on
CPU, scores cached. Random: mean over 100 random orders.

Dev, 125 questions, 87 articles:

    random: MRR 0.097  R@5 0.056  R@10 0.112  R@20 0.226  nDCG@10 0.063
    bm25:   MRR 0.325  R@5 0.271  R@10 0.463  R@20 0.608  nDCG@10 0.293
    dense:  MRR 0.313  R@5 0.306  R@10 0.416  R@20 0.614  nDCG@10 0.281

BM25 vs dense, paired sign-flip permutation test (10,000 draws): MRR -0.012
(p 0.72), R@10 -0.048 (p 0.29), nDCG@10 -0.012 (p 0.67): no difference.

Per article (answer in top 10, BM25 / dense / times an answer): 7:453 15/0/44,
7:457 11/15/23, 7:448 20/18/22, 7:464 9/4/22, 7:446 12/9/20, 7:462 12/19/19,
7:450 4/16/18, 6:96 11/5/16, 7:454 9/4/13, 6:74 0/0/11, 6:162 0/0/9.
The systems fail on different articles; neither reaches the short, abstract
general norms (6:74, 6:162; 7:453 for dense). That is where atoms and the
graph should add something.

Exploratory, not yet a system in the script: summing z-normalised BM25 and
dense scores gives MRR 0.367, R@10 0.461, nDCG@10 0.310 on dev. A BM25+dense
hybrid is a standard strong baseline and should be added properly (and is the
natural first stage for the atom re-ranker).

Sanity checks (is BM25 ~ dense a bug?). No: the rankings differ (median
Spearman 0.40 per question, median top-10 overlap 3 of 10). The tie is a
statement of uncertainty, not equality: bootstrap 95% interval for dense -
BM25 is [-0.078, +0.054] for MRR and [-0.135, +0.041] for recall@10. Dense
lags more on the longer half of the questions (R@10 0.47 vs 0.55; shorter
half 0.36 vs 0.38), suggesting one embedding of a long story dilutes it, but
truncating questions to 512 tokens makes dense worse (R@10 0.373, MRR 0.310),
so the whole text helps. A chunked variant (score question pieces separately,
aggregate) is a candidate. Zero-shot dense ~ BM25 matches the literature
(BEIR; BSARD, where only dense models trained on the task beat BM25).


## 19/10/2026 — Hybrid baseline (BM25 + dense)

Two fusion methods in `src/similarity/retrieval.py`, both systems in
`scripts/run_baselines.py`:
- `hybrid`: z-score fusion, sum of per-question standardised scores, equal
  weights (no tuning).
- `hybrid_rrf`: reciprocal rank fusion, sum of 1/(60 + rank) (Cormack et al.
  2009; k = 60 not tuned). Ties share the average rank.

Dev, 125 questions:

    hybrid:     MRR 0.367  R@5 0.292  R@10 0.461  R@20 0.658  nDCG@10 0.310
    hybrid_rrf: MRR 0.376  R@5 0.300  R@10 0.463  R@20 0.658  nDCG@10 0.318

Paired tests, hybrid_rrf minus (bootstrap 95% interval, permutation p):
- vs BM25: MRR +0.051 [-0.001, +0.104] p 0.06; R@10 0.000 p 1.0; nDCG@10 +0.025 p 0.24
- vs dense: MRR +0.063 [+0.013, +0.110] p 0.01; R@10 +0.047 p 0.19; nDCG@10 +0.037 p 0.09
- vs z-score hybrid: no difference (MRR +0.009, p 0.60)

The hybrid improves the order (the first correct article comes earlier) but
not coverage (recall@10 unchanged). Per article in the top 10 (RRF): 7:453
4/44 (BM25 15, dense 0: fusion dilutes BM25's hits), 7:450 13/18, 7:457 17/23,
6:74 0/11, 6:162 0/9. The general-norm articles remain unreachable.

Decision: `hybrid_rrf` is the reference baseline for study A (best on every
metric, no parameter tuned on dev) and the first stage for the atom
re-ranker (shortlist = its top k).


## 19/10/2026 — Score spreading over citation and section links

`src/graph/propagation.py`; systems `graph_cite`, `graph_tree`, `graph_both`
in `scripts/run_baselines.py`. First stage: `hybrid_rrf`, scaled per question
to [0, 1]. One spreading step s' = s + beta * norm(A) s per link type.
- Citation links: BWB references between corpus articles, undirected
  (48 distinct pairs; 47 of 87 articles have none; 6:162 and 7:453 none,
  6:74 one).
- Tree links: same afdeling (WGBO = 23 articles in one section).
- Normalisation: symmetric (D^-1/2 A D^-1/2). The plain neighbour average was
  tried first and hurt citations at every beta (MRR 0.36 at beta 0.25, 0.28
  at 1.0): an article with one link gets its neighbour's full score; e.g.
  7:457 was pushed into the top 10 wrongly 36 times. Max-of-neighbours was
  neutral.

Co-citation check (dev): of 320 answer pairs, 17 are linked by citation
(5.3% vs 1.3% for random pairs) and 145 share a section (45% vs 17%).

Beta chosen on dev by nDCG@10 (`--sweep` -> outputs/tables/graph_sweep_dev.csv):
citations 0.25, sections 1.0, both 0.1 + 1.0. Higher citation betas hurt
(beta 1: nDCG 0.292, 4: 0.259).

Dev, 125 questions:

    hybrid_rrf: MRR 0.376  R@5 0.300  R@10 0.463  R@20 0.658  nDCG@10 0.318
    graph_cite: MRR 0.370  R@5 0.339  R@10 0.495  R@20 0.627  nDCG@10 0.336
    graph_tree: MRR 0.388  R@5 0.319  R@10 0.485  R@20 0.684  nDCG@10 0.333
    graph_both: MRR 0.401  R@5 0.341  R@10 0.493  R@20 0.694  nDCG@10 0.347

Paired vs hybrid_rrf (bootstrap 95% interval, permutation p):
- graph_cite: nDCG +0.019 [-0.013, +0.050] p 0.25; MRR -0.006 p 0.80; R@10 +0.032 p 0.11
- graph_tree: nDCG +0.015 [+0.003, +0.028] p 0.02; MRR +0.012 p 0.23; R@10 +0.022 p 0.07
- graph_both: nDCG +0.030 [+0.009, +0.051] p 0.005; MRR +0.025 p 0.12; R@10 +0.030 p 0.07

Caveat: beta was selected on the same dev questions, so these gains are
optimistic; the test run decides. Per article in the top 10 (hybrid ->
graph_both): 7:457 17 -> 20, 7:454 7 -> 9, 7:462 12 -> 14, 6:96 8 -> 6,
7:453 4 -> 3, 6:74 0 -> 0, 6:162 0 -> 0. The law's explicit structure gives a
small, broad gain but cannot reach the general-norm articles.
