"""Natural-language queries: turn a legal question into atom-shaped queries.

The retriever (classify_atom) scores *atoms* against atoms, so a question
such as "the hospital lost my medical file and now I cannot prove the
treatment went wrong" must first be decomposed into atoms and labelled with
the same 9-label schema as the corpus. This module asks Claude to do that,
then ranks articles for the whole question.

Tags only match exactly when they are spelled the same way, so the corpus
vocabulary is put in the prompt and the model is told to reuse those strings
whenever the meaning is the same. Tags are in English because the corpus
annotations are; the question itself may be Dutch or English.

Usage:
    from src.data.query import retrieve_for_text

    result = retrieve_for_text("De patiënt wil dat zijn dossier vernietigd wordt", df)
    result["query_atoms"]       # the decomposed query
    result["article_ranking"]   # articles, best first

Requires the ``anthropic`` package and credentials (ANTHROPIC_API_KEY or an
``ant auth login`` profile). Every call costs a few cents of API usage.
"""

from __future__ import annotations

import json

import pandas as pd

from src.data.atom_table import (
    LIST_FIELDS, build_semantic_lookup, classify_atom, compute_tag_weights,
    unique_tags_by_field,
)

DEFAULT_MODEL = "claude-opus-5-5"

# Re-runs a request on another model server-side if a safety classifier
# declines it. Legal questions are very unlikely to trigger this, but it
# costs nothing when unused.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_SYSTEM_PROMPT = """\
You decompose a legal question or case description into atoms for a statutory \
retrieval system over Dutch law. An atom is the smallest piece of text that \
carries a single legal meaning. Facts of the case and things the person wants \
are separate atoms; a sentence with "and" between two independent facts is two atoms.

Label each atom with the lab's annotation schema:
- condition_type: "pre_condition" for facts or requirements (the IF part), \
"post_condition" for a legal consequence, duty or remedy (the THEN part).
- actors: persons or bodies with legal capacity (e.g. care provider, patient, employer).
- legal_relations: the relationship between actors (e.g. treatment agreement, \
employment relationship).
- acts: what an actor does or must do, as a short verb phrase.
- geographical_domain: where the rule applies.
- temporal: when, before or after, deadlines, durations.
- explicit_references: articles or documents named in the text.
- hierarchies: phrases such as "without prejudice to" or "notwithstanding".
- residual: other legally relevant terms not covered above (objects, standards, \
qualifiers such as "file", "damage", "proper care").

Write every tag in English, as a short lowercase phrase, even when the question \
is in Dutch. Leave a field empty when nothing in the atom fits it; do not guess.

The retrieval system matches tags by exact string. When a tag with the same \
meaning already exists in the vocabulary below, copy that exact string instead \
of writing a synonym. Only write a new tag when no existing one fits.

Existing tag vocabulary, by field:
{vocabulary}"""


def _atom_schema() -> dict:
    list_field = {"type": "array", "items": {"type": "string"}}
    atom = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "condition_type": {"type": "string", "enum": ["pre_condition", "post_condition"]},
            **{f: list_field for f in LIST_FIELDS},
        },
        "required": ["text", "condition_type", *LIST_FIELDS],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"atoms": {"type": "array", "items": atom}},
        "required": ["atoms"],
        "additionalProperties": False,
    }


def _format_vocabulary(vocabulary: dict[str, list[str]], max_tags_per_field: int) -> str:
    lines = []
    for field in LIST_FIELDS:
        tags = sorted(vocabulary.get(field, []))[:max_tags_per_field]
        lines.append(f"{field}: {json.dumps(tags, ensure_ascii=False)}")
    return "\n".join(lines)


def parse_query_atoms(payload: dict, vocabulary: dict[str, list[str]] | None = None) -> list[dict]:
    """Turn the model's JSON into atom dicts ready for classify_atom().

    Tags are stripped and de-duplicated. A tag that matches a vocabulary tag
    apart from capitalisation is replaced by the vocabulary spelling, so it
    can match exactly.
    """
    canonical: dict[tuple[str, str], str] = {}
    for field, tags in (vocabulary or {}).items():
        for tag in tags:
            canonical.setdefault((field, tag.strip().lower()), tag)

    atoms = []
    for i, raw in enumerate(payload.get("atoms", [])):
        atom = {
            "atom_id": f"Q.{i + 1}",
            "article_id": "QUERY",
            "text": raw.get("text", ""),
            "condition_type": raw.get("condition_type", "pre_condition"),
        }
        for field in LIST_FIELDS:
            seen: list[str] = []
            for tag in raw.get(field, []) or []:
                tag = " ".join(str(tag).split())
                if not tag:
                    continue
                tag = canonical.get((field, tag.lower()), tag)
                if tag not in seen:
                    seen.append(tag)
            atom[field] = seen
        atoms.append(atom)
    return atoms


def text_to_query_atoms(
    text: str,
    vocabulary: dict[str, list[str]] | None = None,
    client=None,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    max_tags_per_field: int = 300,
) -> list[dict]:
    """Decompose ``text`` into labelled query atoms with Claude.

    Parameters
    ----------
    text : str
        The question or case description (Dutch or English).
    vocabulary : dict or None
        ``{field: [tags]}``, usually unique_tags_by_field(df). Strongly
        recommended: without it the model invents its own tag spellings and
        exact matching mostly fails.
    client : anthropic.Anthropic or None
        Pass one to reuse a client (or a stub in tests).
    effort : str
        Reasoning effort. "medium" is enough for decomposition; raise it if
        the atoms look sloppy.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    vocabulary = vocabulary or {}
    system = _SYSTEM_PROMPT.format(
        vocabulary=_format_vocabulary(vocabulary, max_tags_per_field)
    )
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=[_FALLBACK_BETA],
        fallbacks="default",
        # The system prompt (instructions + vocabulary) is identical across
        # queries, so cache it.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": _atom_schema()},
        },
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"The model declined to decompose the query: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model's answer was cut off (max_tokens); try a shorter query.")

    payload = json.loads(next(b.text for b in response.content if b.type == "text"))
    return parse_query_atoms(payload, vocabulary)


def embed_vocabulary(vocabulary: dict[str, list[str]], model) -> dict:
    """Embed every corpus tag once: ``{field: (tags, normalised embedding matrix)}``."""
    out = {}
    for field in LIST_FIELDS:
        tags = sorted(vocabulary.get(field, []))
        if tags:
            out[field] = (tags, model.encode(tags, normalize_embeddings=True,
                                             convert_to_numpy=True))
    return out


def extend_semantic_lookup(
    sem_lookup: dict,
    query_atoms: list[dict],
    vocab_embeddings: dict,
    model,
    threshold: float = 0.55,
) -> dict:
    """Add similarity links for query tags the corpus has never seen.

    The precomputed lookup only pairs corpus tags with corpus tags, so a new
    query tag such as "doctor" could never match "care provider"
    semantically. This embeds the query's unseen tags with the same model and
    links each to the corpus tags in the same field at or above ``threshold``.
    Returns a new lookup; the original is not modified.
    """
    lookup = {k: list(v) for k, v in sem_lookup.items()}
    for field, (tags, matrix) in vocab_embeddings.items():
        known = set(tags)
        new = sorted({t for a in query_atoms for t in a.get(field, []) if t not in known})
        if not new:
            continue
        sims = model.encode(new, normalize_embeddings=True, convert_to_numpy=True) @ matrix.T
        for i, tag in enumerate(new):
            for j, sim in enumerate(sims[i]):
                if sim >= threshold:
                    lookup.setdefault((field, tag), []).append((tags[j], float(sim)))
    return lookup


def rank_articles_for_query_atoms(
    query_atoms: list[dict],
    df: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    normalize: str = "cosine",
    sem_lookup: dict | None = None,
) -> pd.DataFrame:
    """Rank articles for a question that was split into several atoms.

    Each query atom ranks the articles on its own (an article scores as its
    best-matching atom). An article's final score is the *sum* over query
    atoms, so an article that answers several parts of the question beats
    one that answers a single part very well.

    Returns columns: article_id, score, coverage, n_query_atoms_matched,
    best_atoms (``{query atom id: best-matching corpus atom}``).

    ``score`` is the sum over query atoms and grows with the number of atoms,
    so it is only comparable within one question. ``coverage`` is the same
    sum divided by the number of query atoms: a 0-1 value ("how well does
    this article cover the question on average") that is comparable across
    questions. Both give the same ranking.
    """
    if weights is None:
        weights = compute_tag_weights(df)
    if sem_lookup is None:
        sem_lookup = build_semantic_lookup(sim_pairs)

    totals: dict[str, float] = {}
    matched: dict[str, int] = {}
    evidence: dict[str, dict[str, str]] = {}
    for q in query_atoms:
        ranking = classify_atom(
            q, df, weights=weights, sem_lookup=sem_lookup,
            alpha=alpha, normalize=normalize,
        )["article_ranking"]
        for row in ranking.itertuples(index=False):
            totals[row.article_id] = totals.get(row.article_id, 0.0) + row.article_score
            if row.article_score > 0:
                matched[row.article_id] = matched.get(row.article_id, 0) + 1
                evidence.setdefault(row.article_id, {})[q["atom_id"]] = row.best_atom

    out = pd.DataFrame({
        "article_id": list(totals),
        "score": [round(v, 4) for v in totals.values()],
        "coverage": [round(v / max(len(query_atoms), 1), 4) for v in totals.values()],
        "n_query_atoms_matched": [matched.get(a, 0) for a in totals],
        "best_atoms": [evidence.get(a, {}) for a in totals],
    })
    return out.sort_values(["score", "article_id"], ascending=[False, True]).reset_index(drop=True)


def retrieve_for_text(
    text: str,
    df: pd.DataFrame,
    weights: dict | None = None,
    sim_pairs: pd.DataFrame | None = None,
    alpha: float = 0.5,
    normalize: str = "cosine",
    client=None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """End to end: question text -> query atoms -> ranked articles.

    Returns ``{"query_atoms": [...], "article_ranking": DataFrame}``.
    """
    atoms = text_to_query_atoms(
        text, vocabulary=unique_tags_by_field(df), client=client, model=model,
    )
    ranking = rank_articles_for_query_atoms(
        atoms, df, weights=weights, sim_pairs=sim_pairs, alpha=alpha, normalize=normalize,
    )
    return {"query_atoms": atoms, "article_ranking": ranking}
