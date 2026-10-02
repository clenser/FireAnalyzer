"""Deterministic material identification against ``flame_dataset.json``.

No LLM is involved.  A measured flame colour is compared to the reference
flame colours stored in the database using Euclidean distance in CIELAB space
(the space the database itself is expressed in), and the distances are turned
into a heuristic ``similarity`` score.

The score is **not** a probability::

    similarity = max(0.0, 1.0 - distance / similarity_distance_scale)

where ``distance`` is the smallest distance to any reference colour of that
material.  Multiple representative colours (K-Means, GMM) are averaged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .color_analysis import RepresentativeColor
from .config import Settings
from .errors import MissingDatabaseError
from .schemas import MaterialAnalysis, MaterialScore, SuppressionInformation

__all__ = [
    "MaterialEntry",
    "MaterialDatabase",
    "load_material_database",
    "match_material",
    "rank_materials",
    "suppression_for",
]

logger = logging.getLogger(__name__)

SIMILARITY_DECIMALS = 4
DATABASE_SOURCE = "flame_dataset.json"


@dataclass(frozen=True)
class MaterialEntry:
    """One material record from the database (LAB references drive matching)."""

    name: str
    flame_lab: np.ndarray  # float64, shape (M, 3)
    extinguishers: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True)
class MaterialDatabase:
    """The loaded material database plus its source label."""

    entries: tuple[MaterialEntry, ...]
    source: str = DATABASE_SOURCE

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, name: str) -> MaterialEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None


def _as_array(records: Iterable[Sequence[float]], width: int, dtype) -> np.ndarray:
    array = np.asarray(list(records), dtype=dtype)
    if array.size == 0:
        return np.empty((0, width), dtype=dtype)
    return array.reshape(-1, width)


def load_material_database(path: str | Path) -> MaterialDatabase:
    """Load and validate ``flame_dataset.json``.

    Raises
    ------
    MissingDatabaseError
        If the file is missing, unreadable, not valid JSON, or contains no
        usable material records.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise MissingDatabaseError(f"Material database not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise MissingDatabaseError(f"Material database could not be read: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("materials"), list):
        raise MissingDatabaseError("Material database must be an object with a 'materials' list.")

    entries: list[MaterialEntry] = []
    for index, record in enumerate(payload["materials"]):
        if not isinstance(record, dict) or "name" not in record:
            logger.warning("Skipping malformed material record at index %s", index)
            continue
        lab = _as_array(record.get("flame_lab", []), 3, np.float64)
        if lab.shape[0] == 0:
            logger.warning("Material %r has no flame_lab references; skipping", record["name"])
            continue
        extinguishers = record.get("possible_extinguishers") or []
        if isinstance(extinguishers, str):
            extinguishers = [extinguishers]
        notes = record.get("notes")
        entries.append(
            MaterialEntry(
                name=str(record["name"]),
                flame_lab=lab,
                extinguishers=tuple(str(item) for item in extinguishers),
                notes=str(notes) if notes else None,
            )
        )

    if not entries:
        raise MissingDatabaseError("Material database contains no usable materials.")

    logger.info("Loaded %s materials from %s", len(entries), path)
    return MaterialDatabase(entries=tuple(entries), source=path.name)


def _similarity(lab: np.ndarray, references: np.ndarray, scale: float) -> float:
    """Best heuristic similarity between one LAB colour and a set of references."""
    distances = np.linalg.norm(references - lab.reshape(1, 3), axis=1)
    best = float(np.min(distances)) if distances.size else float("inf")
    if scale <= 0:
        raise ValueError("similarity_distance_scale must be > 0")
    return max(0.0, 1.0 - best / scale)


def _score_entry(
    entry: MaterialEntry,
    colors: Sequence[RepresentativeColor],
    settings: Settings,
) -> float:
    """Average similarity of one material across all representative colours."""
    scores = [
        _similarity(color.lab, entry.flame_lab, settings.similarity_distance_scale)
        for color in colors
    ]
    return float(np.mean(scores))


def rank_materials(
    flame_colors: Sequence[RepresentativeColor],
    database: MaterialDatabase,
    settings: Settings,
) -> list[MaterialScore]:
    """Every canonical material with its similarity, best first.

    Equal similarities keep dataset order (a stable sort) - never alphabetical
    order.  Genuine ties are resolved downstream by
    :mod:`app.material_fusion`, which reports them as uncertain.
    """
    colors = [color for color in flame_colors if color is not None]
    if not colors:
        raise ValueError("rank_materials() requires at least one representative colour")
    return sorted(
        (
            MaterialScore(
                material=entry.name,
                similarity=round(_score_entry(entry, colors, settings), SIMILARITY_DECIMALS),
            )
            for entry in database.entries
        ),
        key=lambda score: -score.similarity,
    )


def match_material(
    flame_colors: Sequence[RepresentativeColor],
    database: MaterialDatabase,
    settings: Settings,
) -> tuple[MaterialAnalysis, SuppressionInformation]:
    """Rank database materials against the measured flame colours.

    Parameters
    ----------
    flame_colors:
        Representative colours (K-Means, GMM, mean).  At least one is required.
    database:
        Database loaded with :func:`load_material_database`.
    settings:
        Active configuration (distance scale, number of alternatives).

    Returns
    -------
    tuple[MaterialAnalysis, SuppressionInformation]
        The ranked material analysis plus the suppression record copied from the
        database for the winning material.
    """
    ranked = rank_materials(flame_colors, database, settings)
    best = ranked[0]
    entry = database.get(best.material)
    analysis = MaterialAnalysis(
        primary_material=best.material,
        similarity=best.similarity,
        alternatives=ranked[1 : 1 + max(0, settings.max_alternatives)],
        database_notes=entry.notes if entry else None,
    )
    suppression = suppression_for(database, best.material)
    return analysis, suppression


def suppression_for(database: MaterialDatabase, material: str | None) -> SuppressionInformation:
    """Return the suppression record stored in the database for ``material``.

    Values are returned verbatim - nothing is generated or invented.
    """
    entry = database.get(material) if material else None
    if entry is None:
        return SuppressionInformation(
            source=database.source, material=None, methods=[], database_notes=None
        )
    return SuppressionInformation(
        source=database.source,
        material=entry.name,
        methods=list(entry.extinguishers),
        database_notes=entry.notes,
    )
