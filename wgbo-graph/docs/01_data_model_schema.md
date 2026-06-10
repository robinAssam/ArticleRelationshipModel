# 01 — Data Model Schema

The conceptual blueprint for what we store and how it relates. This
predates any code: any extraction must populate something defined here.

## Entities

### Article
The whole statutory article (e.g. `7:454 BW`).

| Field | Type | Example |
|---|---|---|
| `article_id` | str | `"7:454"` |
| `source` | str | `"BW Boek 7, Titel 7.7.5"` |
| `title` | str | `"Dossierplicht"` |
| `full_text` | str | (full Dutch text) |

### Lid
A paragraph within an article (e.g. `7:454(1)`).

| Field | Type | Example |
|---|---|---|
| `lid_id` | str | `"7:454(1)"` |
| `article_id` | str (FK) | `"7:454"` |
| `lid_number` | int | `1` |
| `text` | str | (full Dutch text of this lid) |

### Atom
A boolean statement / atomic requirement. The working unit.

| Field | Type | Example |
|---|---|---|
| `atom_id` | str | `"7:454(1).a"` |
| `lid_id` | str (FK) | `"7:454(1)"` |
| `text` | str | `"De hulpverlener richt een dossier in..."` |
| `condition_type` | enum | `"pre_condition"` \| `"post_condition"` |
| `actors` | list[str] | `["hulpverlener"]` |
| `legal_relations` | list[str] | `["behandelingsovereenkomst"]` |
| `acts` | list[str] | `["richt dossier in"]` |
| `geographical_domain` | list[str] | `[]` |
| `temporal` | list[str] | `[]` |
| `explicit_references` | list[str] | `[]` |
| `hierarchies` | list[str] | `[]` |
| `residual` | list[str] | `["dossier", "behandeling"]` |

### Case
A judgment used for validation.

| Field | Type | Example |
|---|---|---|
| `ecli` | str | `"ECLI:NL:HR:2010:BN6236"` |
| `court` | str | `"HR"` |
| `date` | date | `2010-12-17` |
| `articles_cited` | list[str] | `["7:454", "6:162"]` |
| `primair` | list[str] | `["6:162"]` |
| `subsidiair` | list[str] | `["7:454"]` |
| `summary` | str | (short factual summary) |

## Relations

### contains-lid
Article → Lid (directed, 1-to-many).

### contains-atom
Lid → Atom (directed, 1-to-many).

### cross-references
Atom → Article (directed). Source: explicit citation in atom text.
Example: 7:454(3).a → 7:455 via "onverminderd artikel 455".

### shared-field
Atom ↔ Atom (undirected, multi-edge by field type).
Sub-types: `shared-actor`, `shared-act`, `shared-residual`,
`shared-legal-relation`, `shared-temporal`, `shared-hierarchy`.

### lexical-similar
Atom ↔ Atom (undirected, weighted). Cosine/Jaccard score above threshold.

### semantic-similar
Atom ↔ Atom (undirected, weighted). Embedding cosine above threshold.

### co-pleaded
Article ↔ Article (undirected, weighted by frequency). Derived from
case-law evidence.

### alternative-to
Article → Article (directed). Derived from primair/subsidiair patterns.

### supports-evidentially
Article → Article (directed). The *omkeringsregel* relation. 7:454 →
6:162 is the canonical instance.

## Aggregation rule

Article-level relations are derived from atom-level relations by:

- **coverage**: fraction of A's atoms with at least one matched atom in B
- **strength**: mean similarity score of matched atom pairs
- **type profile**: count of edges per sub-type

These aggregated metrics are what the final graph shows; atom-level
edges remain available for drill-down.

## Open questions

- Should `Element` be a separate entity, or always identical to `Atom`?
  Current decision: yes, identical. Revisit if element-level decomposition
  diverges from boolean atoms.
- How to represent the *evidential support* relation when it derives from
  case law rather than statutory text? Current decision: as a directed
  edge with `source: case_law` and a list of supporting ECLIs.
