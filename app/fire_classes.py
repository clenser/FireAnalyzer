"""Fire-class and extinguishing-agent derivation.

The fire class is **derived**, never predicted by a model.  The detection model
(``OBJ_best.pt``) predicts a single fire class only; the segmentation model
produces masks only.  Nothing in this project trains a fire-class classifier, so
this module is the single, explicit place where a detected *material* becomes a
*fire class*, and every step is deterministic.

Where the data comes from
-------------------------
``data/flame_dataset.json`` was inspected before this module was written.  It
contains **no** fire-class field and **no** chemical compound data: each of its
18 material records has only ``name``, ``flame_rgb``, ``flame_lab``,
``possible_extinguishers`` and ``notes``.  It records exactly seven distinct
extinguisher names:

    CO2, Class D powder, Dry powder, Foam, Sand, Water, Water spray

Two consequences, both deliberate:

1. The material -> fire-class table below is an **explicit configuration**
   built only from the 18 material categories the project already represents.
   It is not guessed from generic assumptions about fire, and a test asserts it
   covers exactly the materials in the dataset.
2. The extinguisher records are **agent names, not compounds**.  No chemical
   formula is fabricated: every agent carries ``compound: null`` and a note
   pointing at the dataset as the source.  ``type`` is the suppression *mechanism*
   category, which is a property of the agent, not a claim about the burning
   chemistry.

Mapping rule
------------
The correct extinguishing agent is the defining operational property of a fire
class, and every record in the dataset already states which agents are correct
for it.  ``fire_class_for_agents`` therefore derives the class from the
material's own ``possible_extinguishers`` set, in this fixed precedence:

1. A metal-specific agent is present (``Class D powder`` or ``Sand``)
   -> **Class D**, combustible metal.
2. The material name denotes electrical equipment
   -> **Class C**, energized electrical equipment.  The dataset agrees: its
   note for ``Electrical Components`` reads "never use water on live circuits"
   and its only agents are CO2 and Foam.
3. Plain ``Water`` is an acceptable agent
   -> **Class A**, ordinary combustibles.  Water is the correct agent for
   Class A and is not correct for the classes above.
4. Otherwise
   -> **Class B**, flammable material: CO2 / foam / dry powder only, water is
   not listed as acceptable.

``MATERIAL_FIRE_CLASS`` spells the resulting table out for all 18 materials so
the mapping is readable and reviewable rather than implicit, and
``tests/test_pipeline.py`` asserts the table agrees with the rule above for
every material in the dataset.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from .material_matching import MaterialDatabase
from .schemas import ExtinguishingAgent, FireClassResult

__all__ = [
    "CLASS_A",
    "CLASS_B",
    "CLASS_C",
    "CLASS_D",
    "UNKNOWN_CLASS",
    "CLASS_DESCRIPTIONS",
    "MAPPING_SOURCE",
    "MATERIAL_FIRE_CLASS",
    "AGENT_SUPPRESSION_TYPE",
    "UNSPECIFIED_AGENT_TYPE",
    "AGENT_TYPE_BASIS",
    "fire_class_for_agents",
    "fire_class_for_material",
    "agent_suppression_type",
    "extinguishing_agents_for",
    "classify",
]

logger = logging.getLogger(__name__)

CLASS_A = "Class A"
CLASS_B = "Class B"
CLASS_C = "Class C"
CLASS_D = "Class D"
UNKNOWN_CLASS = "Unclassified"

#: Short operational descriptions of the four classes.  These describe the class
#: itself, not the detected sample, and are used for display only.
CLASS_DESCRIPTIONS: dict[str, str] = {
    CLASS_A: "Ordinary combustibles",
    CLASS_B: "Flammable liquid or flammable material",
    CLASS_C: "Energized electrical equipment",
    CLASS_D: "Combustible metal",
    UNKNOWN_CLASS: "Not enough evidence to assign a fire class",
}

#: Where the mapping lives, reported in ``fire_class.mapping_source``.
MAPPING_SOURCE = "app/fire_classes.py (documented material -> fire class mapping)"

#: Agents that only apply to burning metals, and therefore imply Class D.
_METAL_AGENTS = ("Class D powder", "Sand")

#: Substrings that mark a material as electrical equipment.
_ELECTRICAL_TOKENS = ("electrical", "electric")

#: Material -> fire class.  Exactly the 18 categories in ``flame_dataset.json``,
#: each following the rule documented in the module docstring.
MATERIAL_FIRE_CLASS: dict[str, str] = {
    # Class D - combustible metal (Class D powder / sand in the dataset).
    "Metal Objects": CLASS_D,
    # Class C - energized electrical equipment (CO2 / foam only).
    "Electrical Components": CLASS_C,
    # Class A - ordinary combustibles (water is an acceptable agent).
    "Paper Products(Wood material)": CLASS_A,
    "Wood Materials": CLASS_A,
    "Natural Fibers": CLASS_A,
    "Wax Materials": CLASS_A,
    "Carpet Materials": CLASS_A,
    "Organic Waste": CLASS_A,
    "Food Products": CLASS_A,
    "Composite Materials": CLASS_A,
    # Class B - flammable material (CO2 / foam / dry powder; water spray only).
    "Plastic Materials": CLASS_B,
    "Foam Materials": CLASS_B,
    "Rubber Materials": CLASS_B,
    "Alcohol-Based Products": CLASS_B,
    "Adhesive Materials": CLASS_B,
    "Spray Products": CLASS_B,
    "Liquid Fuels": CLASS_B,
    "Chemical Products": CLASS_B,
}

#: Extinguisher name -> suppression mechanism category.  Covers exactly the seven
#: names recorded in ``flame_dataset.json``; a test asserts that.  These are the
#: standard suppression mechanisms for the agents, not claims about compounds.
#:
#: Note on the vocabulary: ``flame_dataset.json`` records agent *names* only and
#: contains no mechanism terminology whatsoever, so the four categories below are
#: this project's own explicit controlled vocabulary, not a value read from the
#: data.  ``blanketing`` means isolating the burning surface from air, and is
#: used for agents that smother the fuel rather than thinning the surrounding
#: atmosphere the way ``oxygen_displacement`` (CO2) does.
AGENT_SUPPRESSION_TYPE: dict[str, str] = {
    "Water": "cooling",
    "Water spray": "cooling",
    "CO2": "oxygen_displacement",
    "Foam": "blanketing",
    "Dry powder": "chemical_chain_break",
    # Class D powder is a special-purpose metal-fire agent: it absorbs heat and
    # forms a protective crust that isolates the burning surface from air. That
    # is blanketing, not oxygen displacement - CO2 is the oxygen-displacement
    # agent, and Class D powder does not act that way.
    "Class D powder": "blanketing",
    "Sand": "oxygen_displacement",
}

#: Used when the dataset ever records an agent this module does not know, so an
#: unknown name is reported as unknown instead of being given a guessed type.
UNSPECIFIED_AGENT_TYPE = "unspecified"

#: Why ``compound`` is null.  Reported per agent so no client has to guess.
AGENT_TYPE_BASIS = (
    "name verbatim from flame_dataset.json; the dataset records no chemical "
    "identity, so no compound formula is asserted"
)

#: Appended to ``fire_class.basis`` to make the derivation chain explicit.
_CLASS_BASIS = (
    "Derived from the matched material via the documented material -> fire class "
    "mapping in app/fire_classes.py, which is itself derived from the "
    "extinguishers recorded in flame_dataset.json. The fire class is not "
    "predicted by the detection model, and the confidence is the material "
    "match's distance-based similarity, not a calibrated probability."
)


def _normalise(agents: Iterable[str]) -> set[str]:
    """Agent names as a lookup set, stripped of case and padding."""
    return {str(agent).strip().casefold() for agent in agents if str(agent).strip()}


#: Pre-lowercased view of the metal agents for fast membership tests.
_METAL_AGENTS_LOWER = _normalise(_METAL_AGENTS)


def _is_electrical(material: str | None) -> bool:
    """True when the material name denotes electrical equipment."""
    if not material:
        return False
    lowered = material.casefold()
    return any(token in lowered for token in _ELECTRICAL_TOKENS)


def fire_class_for_agents(agents: Sequence[str], material: str | None = None) -> str:
    """Derive a fire class from a material's recorded extinguishers.

    Applies the documented precedence in the module docstring.  ``material`` is
    only consulted for the electrical test (step 2); the rest of the rule is
    derived from the extinguisher list alone.  Deterministic and total: any
    input yields one of the four classes or :data:`UNKNOWN_CLASS` for an empty
    agent list.
    """
    names = _normalise(agents)
    if not names:
        return UNKNOWN_CLASS
    if names & _METAL_AGENTS_LOWER:
        return CLASS_D
    if _is_electrical(material):
        return CLASS_C
    if "water" in names:
        return CLASS_A
    return CLASS_B


def fire_class_for_material(material: str | None, agents: Sequence[str]) -> tuple[str, str]:
    """Return ``(fire_class, how_it_was_decided)`` for a material.

    The explicit :data:`MATERIAL_FIRE_CLASS` table wins so the mapping is
    reviewable in one place.  A material that is not in the table - which can
    only happen if the dataset grows - falls back to the agent-based rule, and
    the returned basis string says so.
    """
    if not material:
        return UNKNOWN_CLASS, "No material was matched, so no fire class could be derived."

    configured = MATERIAL_FIRE_CLASS.get(material)
    if configured is not None:
        return configured, f"material '{material}' -> {configured} via {MAPPING_SOURCE}"

    derived = fire_class_for_agents(agents, material)
    if derived == UNKNOWN_CLASS:
        return (
            UNKNOWN_CLASS,
            f"material '{material}' has no extinguisher records in flame_dataset.json, "
            "so no fire class could be derived.",
        )
    return (
        derived,
        f"material '{material}' is not listed in the explicit mapping, so the class was "
        f"derived from its recorded agents ({', '.join(agents) or 'none'}) -> {derived}",
    )


def agent_suppression_type(name: str) -> str:
    """The suppression mechanism category for an agent name.

    Returns :data:`UNSPECIFIED_AGENT_TYPE` for a name this module does not know,
    so an unrecognised agent is surfaced rather than guessed at.
    """
    return AGENT_SUPPRESSION_TYPE.get(str(name).strip(), UNSPECIFIED_AGENT_TYPE)


def extinguishing_agents_for(
    material: str | None, agents: Sequence[str], fire_class: str | None
) -> list[ExtinguishingAgent]:
    """Build the ``extinguishing_agents`` list for a material.

    Every entry is copied verbatim from the dataset: ``name`` is the recorded
    extinguisher string, ``compound`` is always ``None`` because the dataset
    records no chemical identity, and ``type`` is the suppression mechanism
    category from :data:`AGENT_SUPPRESSION_TYPE`.
    """
    source = "flame_dataset.json"
    return [
        ExtinguishingAgent(
            name=str(name),
            compound=None,  # the dataset records agent names only
            type=agent_suppression_type(name),
            source=source,
            fire_class=fire_class,
            compound_basis=AGENT_TYPE_BASIS,
        )
        for name in agents
    ]


def classify(
    database: MaterialDatabase, material: str | None, similarity: float
) -> tuple[FireClassResult, list[ExtinguishingAgent]]:
    """Derive the fire class and extinguishing agents for a matched material.

    Parameters
    ----------
    database:
        The loaded material database; supplies the recorded extinguishers.
    material:
        ``material_analysis.primary_material``, or ``None``.
    similarity:
        ``material_analysis.similarity``, carried through as
        ``fire_class.confidence``.

    Returns
    -------
    tuple[FireClassResult, list[ExtinguishingAgent]]
        The fire class with its derivation basis, plus the agent records.
    """
    entry = database.get(material) if material else None
    agents = list(entry.extinguishers) if entry else []
    fire_class, basis = fire_class_for_material(material, agents)

    return (
        FireClassResult(
            class_=fire_class,
            description=CLASS_DESCRIPTIONS.get(fire_class, ""),
            confidence=round(float(similarity), 4),
            material=material,
            basis=basis,
            mapping_source=MAPPING_SOURCE,
            notes=_CLASS_BASIS,
        ),
        extinguishing_agents_for(material, agents, fire_class),
    )
