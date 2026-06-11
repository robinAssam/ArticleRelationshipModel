"""Annotation schema — the Atom dataclass for tuple records.

Each atom is one boolean statement extracted from a statutory article,
annotated against the lab's 9-label Legal Preconditions schema.

Usage:
    from src.annotation.schema import Atom, save_atoms, load_atoms

    atom = Atom(
        atom_id="7:454(1).a",
        article_id="7:454",
        lid=1,
        text="De hulpverlener richt een dossier in...",
        condition_type="post_condition",
        actors=["hulpverlener"],
        acts=["richt dossier in"],
        residual=["dossier", "behandeling", "patiënt"],
    )

    save_atoms([atom], "data/annotations/7_454.json")
    atoms = load_atoms("data/annotations/7_454.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

ConditionType = Literal["pre_condition", "post_condition"]


@dataclass
class Atom:
    """One boolean statement extracted from a legal provision, annotated
    against the lab's 9-label schema."""

    # --- metadata ---
    atom_id: str
    article_id: str
    lid: int
    text: str
    source: str = ""

    # --- Task 1: pre vs post ---
    condition_type: ConditionType = "post_condition"

    # --- Tasks 2–9: pre-conditional attributes ---
    actors: list[str] = field(default_factory=list)
    legal_relations: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    geographical_domain: list[str] = field(default_factory=list)
    temporal: list[str] = field(default_factory=list)
    explicit_references: list[str] = field(default_factory=list)
    hierarchies: list[str] = field(default_factory=list)
    residual: list[str] = field(default_factory=list)

    # --- provenance ---
    annotator: str = ""
    annotated_at: str = ""  # ISO-8601 date string
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        return cls(**d)


def save_atoms(atoms: list[Atom], path: str | Path) -> None:
    """Write a list of atoms to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in atoms], f, ensure_ascii=False, indent=2)


def load_atoms(path: str | Path) -> list[Atom]:
    """Load atoms from a JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return [Atom.from_dict(d) for d in data]


def today_iso() -> str:
    """Return today's date as YYYY-MM-DD (useful for `annotated_at`)."""
    return date.today().isoformat()
