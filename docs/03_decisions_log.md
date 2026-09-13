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