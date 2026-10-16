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
annotations as ground truth (part C).


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
are not yet annotated: the full 87-article corpus needs annotating (part C).
