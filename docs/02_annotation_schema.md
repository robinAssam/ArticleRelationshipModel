# 02 — Annotation Schema

This is the 9-label structure from the supervisor's *Annotation Guidelines:
Legal Preconditions* document, adapted to this project's atom-level unit.
The raw guidelines are filed under `docs/supervisor_notes/`.

## The 9 labels

| # | Label | Captures |
|---|---|---|
| 1 | Pre-condition / Post-condition | The IF / THEN halves of the provision |
| 2 | Actors | Natural / legal persons with legal capacity |
| 3 | Legal Relations | Contractual, employment, public-law, family relations |
| 4 | Acts | Positive or negative actions required |
| 5 | Geographical / Physical Domain | Where the requirement applies |
| 6 | Temporal | When, before / after, deadlines |
| 7 | Explicit References | Citations to other articles or documents |
| 8 | Hierarchies | "Subject to", "Notwithstanding", "Without prejudice to" |
| 9 | Residual | Pre-conditional attributes not captured by 2–8 |

For full definitions and edge cases, see
`docs/supervisor_notes/annotation_guidelines.md`.

## Atom tuple schema

Each atom (one boolean statement from an article's decomposition) is
annotated with the structure below. All list fields default to `[]`.

```python
{
  # ---- metadata ----
  "atom_id":         str,        # "7:454(1).a"
  "article_id":      str,        # "7:454"
  "lid":             int,        # 1
  "source":          str,        # "BW Boek 7, Titel 7.7.5"
  "text":            str,        # the raw text span

  # ---- Task 1 ----
  "condition_type":  str,        # "pre_condition" or "post_condition"

  # ---- Task 2–9 ----
  "actors":              list[str],
  "legal_relations":     list[str],
  "acts":                list[str],
  "geographical_domain": list[str],
  "temporal":            list[str],
  "explicit_references": list[str],
  "hierarchies":         list[str],
  "residual":            list[str],

  # ---- provenance ----
  "annotator":  str,             # initials of who annotated
  "annotated_at": str,           # ISO date
  "notes":      str,             # free-text annotator commentary
}
```

## Annotation conventions

These follow the supervisor's guidelines. Repeated here for quick reference.

**Do:**
- Exclude articles ("de", "een").
- Label pronouns each time they appear; link them to their antecedent.
- Label adjacent adjectives together with the actor.
- Label nested actors as one unit.

**Do NOT:**
- Include punctuation (commas, colons, quotes) unless necessary for meaning.
- Label "and" / "or".
- Try to enforce chronological order in temporal labels.
- Annotate tasks sequentially across documents — finish Task 1 for all
  documents before starting Task 2.

**On `or` / `and` in Task 1:**
- Pre/post-conditions separated by *or* → annotate as one chunk.
- Pre/post-conditions separated by *and* (or cumulative) → annotate as
  separate chunks, but only if each chunk is independently readable.

## Worked example — Article 7:454 lid 1

Text (Dutch):

> De hulpverlener richt een dossier in met betrekking tot de behandeling van
> de patiënt. Hij houdt in het dossier aantekening van de gegevens omtrent de
> gezondheid van de patiënt en de te diens aanzien uitgevoerde verrichtingen
> en neemt andere stukken, bevattende zodanige gegevens, daarin op, een en
> ander voor zover dit voor een goede hulpverlening aan de patiënt
> noodzakelijk is.

Decomposed atoms:

| atom_id | text (shortened) | type | actors | acts | residual |
|---|---|---|---|---|---|
| 7:454(1).a | hulpverlener richt dossier in | post | [hulpverlener] | [richt … in] | [dossier, behandeling, patiënt] |
| 7:454(1).b | houdt aantekening van gezondheidsgegevens | post | [hulpverlener] | [houdt aantekening van] | [gezondheidsgegevens, patiënt] |
| 7:454(1).c | houdt aantekening van uitgevoerde verrichtingen | post | [hulpverlener] | [houdt aantekening van] | [uitgevoerde verrichtingen] |
| 7:454(1).d | neemt andere stukken op | post | [hulpverlener] | [neemt op] | [stukken, gegevens] |
| 7:454(1).e | voor zover noodzakelijk voor goede hulpverlening | **pre** | [] | [] | [noodzakelijk, goede hulpverlening] |

Atom **.e** is the only pre-condition; the others are post-conditions
(duties). For lid 3 you'll additionally see populated `temporal`,
`explicit_references` ("artikel 455"), and `hierarchies` ("onverminderd
artikel 455").

## File format on disk

Annotations are stored under `data/annotations/{article_id}.json` as a
list of atom tuples. The Python dataclass + JSON serialiser is in
`src/annotation/schema.py`.
