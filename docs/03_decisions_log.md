# 03 — Decisions Log

A chronological record of every meaningful project decision, what
motivated it, and any alternatives considered. Append at the bottom.
Future-you and the supervisor need this.

Format: one entry per decision, dated, with **D** for the decision,
**W** for why, **A** for alternatives considered (where relevant).

---

## 2026-06-XX — Anchor article

**D:** Anchor the project on Article 7:454 BW (*dossierplicht*).
**W:** Rich dual role — substantive duty + evidential lever via
*omkeringsregel*; clean lid structure (3 leden); strong case-law base on
rechtspraak.nl; explicit cross-reference to 7:455 already gives one
ground-truth edge.
**A:** 7:448 (informatieplicht) — also strong, but more diffusely cited.
6:162 — too general, too many edges, hard to scope.

## 2026-06-XX — Unit of analysis

**D:** Lid-level. Each lid of an article becomes a unit; atoms live
inside lids.
**W:** Different leden of the same article fail and are pleaded
independently in practice. Article-level units lose this resolution.
**A:** Article-level (too coarse), atom-level alone (loses the lid
grouping that matters in pleading).

## 2026-06-XX — Annotation schema

**D:** Adopt the lab's 9-label *Legal Preconditions* schema (Task 1–9)
verbatim as the tuple structure.
**W:** This is the lab's existing standard, runs on the Lawnotation
platform, and supervisor explicitly directed us to use it.
**A:** A custom narrower schema. Rejected — divergence from lab
conventions would cost more than it saves.

## 2026-06-XX — Modelling before extraction

**D:** Complete data model, annotation schema, and validation set
*before* writing extraction code.
**W:** Supervisor directive ("modelling before extraction"). Avoids
retro-fitting interpretation onto whatever embeddings happen to surface.
**A:** Scrape first and refine schema later. Rejected.

## 2026-06-XX — Neighbourhood scope

**D:** Tier 1 (WGBO dossier cluster: 7:454–7:458), Tier 2 (7:451, 7:453),
Tier 3 (6:74, 6:162, 6:170). Tier 4 (GDPR) optional.
**W:** ~10–15 articles is the right size for a 3-week project — large
enough for meaningful graph structure, small enough to hand-annotate.
**A:** Whole WGBO; whole BW Boek 7. Rejected — scope too large.

## 2026-07-XX — Encoder: keep `all-mpnet-base-v2` as default

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

## 2025-07-XX — Classifier: alpha = 0.5, top_k = 3 as defaults

Grid search over alpha ∈ {0, 0.25, 0.5, 0.75, 1.0} and top_k ∈ {3, 5}
on leave-one-out evaluation. Accuracy plateaus from alpha ≥ 0.25 — the
exact-match channel does most of the useful work, and additional
semantic weight beyond a small floor produces no further improvement.

Decision: alpha = 0.5, top_k = 3. Revisit if IDF weighting or the
semantic scoring formula changes materially.

See `notebooks/02_experiments.ipynb`, Section 2.
---
