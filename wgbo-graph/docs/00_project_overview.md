# 00 — Project Overview

## Problem

Take the case where someone wants to sue based on perceived protection by a
certain article for which that person does not meet all the requirements.
That person would then use another article to sue. The other article and
the article used originally do not have the same requirements, but the
requirements share context, similar wording, or sometimes the same exact
wording.

The project's job is to **model the relationships across specific articles**
such that, given one article whose requirements are not met, the model can
surface plausible alternative grounds.

## Anchor: Article 7:454 BW (dossierplicht)

The Dutch civil code obligation on healthcare providers to maintain a
medical record. Three *leden*:

1. Duty to set up and maintain the *dossier*, recording health data and
   treatments necessary for good care.
2. Duty to add the patient's written statement on request.
3. Retention period of twenty years (or longer if a good care provider
   would).

7:454 plays a dual role: it is a substantive duty *and* an evidential lever
that supports claims under 6:74 or 6:162 when records are incomplete (via
the *omkeringsregel*). This dual role gives the project more interesting
relations to model than a simpler article would.

## Neighbourhood

| Tier | Articles | Why |
|---|---|---|
| 1 — WGBO dossier cluster | 7:454, 7:455, 7:456, 7:457, 7:458 | Same domain, share vocabulary and explicit cross-references |
| 2 — WGBO co-pleaded | 7:451, 7:453 | Often invoked alongside 7:454 in cases |
| 3 — Book 6 fallbacks | 6:74, 6:162, 6:170 | Tort / contract grounds when WGBO claims fail |
| 4 — Optional GDPR overlap | Arts. 5, 9, 15, 17 | Cross-regime extension if time permits |

## Relation types modelled

| Relation | Definition | Evidence |
|---|---|---|
| Cross-reference | Article A's text explicitly cites Article B | regex on statutory text |
| Lexical similarity | A and B share non-stopword tokens above threshold | Jaccard / BM25 / TF-IDF |
| Semantic similarity | A and B's embeddings have cosine above threshold | sentence transformer |
| Alternative-grounds (substantive) | A and B pleaded as *primair / subsidiair* for the same claim | rechtspraak.nl judgments |
| Evidential support (special to 7:454) | A's breach reverses burden of proof on a B-based claim | *omkeringsregel* case law |

## Method summary

1. Decompose each article in the neighbourhood into atomic boolean
   statements (one per requirement / duty).
2. Annotate each atom against the lab's 9-label schema — see
   `02_annotation_schema.md`.
3. Produce one structured tuple per atom.
4. Compute pairwise overlap on each labelled field across atoms in
   different articles → edges.
5. Augment with case-law co-citation edges from a primair/subsidiair set.
6. Build a multi-relational graph (networkx).
7. Validate against an independent set of ECLI judgments.
8. Visualise and report.

## Out of scope

- Full automation of the annotation pipeline. Hand annotation is acceptable
  for a corpus this size and more defensible methodologically.
- Articles outside the listed neighbourhood.
- Languages other than Dutch (English/EU extension is "future work").

## Anchor case for validation

**HR 17 december 2010, ECLI:NL:HR:2010:BN6236** — failed sterilisation;
*dossierplicht* (7:454) breach combined with claim under 6:162; the Hoge
Raad applies the *omkeringsregel*. This is the canonical example of the
evidential-support relation the project models.
