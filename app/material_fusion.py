"""Deterministic material fusion - the only place the final material is decided.

Evidence sources
----------------
1. **Deterministic LAB/RGB matcher** (:func:`app.material_matching.rank_materials`):
   CIELAB distance from the merged-mask flame colours to every canonical
   material's reference colours.
2. **Vision evidence** (:mod:`app.vision_providers`): ranked candidates from
   Groq, or from Gemini vision when Groq failed.  Evidence only.

No model is called here; the same inputs always give the same output.

Normalisation
-------------
* Deterministic: LAB distances become a support distribution over the
  canonical materials, ``p_i ∝ exp(-(d_i - d_min) / TAU_LAB)``.  Only the
  *relative* distance matters, so two materials with equally close reference
  colours get equal support (and therefore produce a tie, not a winner).
* Vision: the provider's confidences (already validated to sum to <= 1); the
  unassigned remainder is spread evenly over every material, so a hedging
  provider adds little information.

Weighting
---------
Each source has a fixed base weight (:data:`BASE_WEIGHTS`) scaled by a
reliability in ``[0, 1]``:

* deterministic reliability - flame-region quality (pixels, real mask vs box
  fallback, detector confidence) times colour closeness (how near the best
  reference actually is);
* vision reliability - lower when the provider says it is uncertain or its top
  candidate is high-uncertainty.

``combined_i = (w_det * p_i + w_vis * q_i) / (w_det + w_vis)``.  Neither source
is trusted blindly: colour alone can reach at most a medium confidence, and a
vision answer is diluted by the colour evidence.

Confidence and uncertainty
--------------------------
``confidence = combined_top * (0.5 + 0.5 * coverage)`` where ``coverage`` is the
share of the maximum evidence weight actually available; agreement of the two
sources' leaders adds :data:`AGREEMENT_BONUS`, significant disagreement
multiplies by :data:`DISAGREEMENT_FACTOR`.

``uncertain=true`` - and ``final_material=None`` - when there is no usable
evidence, when the flame region is insufficient, when the runner-up is within
:data:`TIE_RATIO` of the leader (an effective tie), or when the confidence is
below :data:`MIN_CONFIDENCE`.  A winner is never invented; ties are never
broken alphabetically.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from typing import Any, Sequence

from .config import Settings
from .fire_classes import CLASS_DESCRIPTIONS, MAPPING_SOURCE, UNKNOWN_CLASS, classify
from .material_matching import MaterialDatabase
from .schemas import MaterialScore, jsonable

__all__ = [
    "FUSION_VERSION",
    "BASE_WEIGHTS",
    "deterministic_evidence",
    "evidence_quality",
    "fuse_material",
    "aggregate_video_frames",
    "video_fire_class",
    "confidence_level",
]

FUSION_VERSION = "1.0"

#: Fixed base weights of the two evidence sources.
BASE_WEIGHTS = {"deterministic": 0.5, "vision": 0.5}

#: LAB distance scale of the deterministic support distribution.
TAU_LAB = 6.0
#: Best-reference LAB distance at or below which colour closeness is full...
CLOSE_LAB = 15.0
#: ...and at or above which the colour match counts for half.
FAR_LAB = 45.0

#: Reliability of the deterministic evidence per flame-region quality.
QUALITY_RELIABILITY = {"strong": 1.0, "moderate": 0.8, "limited": 0.55, "insufficient": 0.0}

#: Vision reliability multipliers.
VISION_BASE_RELIABILITY = 0.9
VISION_UNCERTAIN_FACTOR = 0.6
VISION_TOP_HIGH_UNCERTAINTY_FACTOR = 0.75

AGREEMENT_BONUS = 0.10
DISAGREEMENT_FACTOR = 0.8
#: A source "has an opinion" in the disagreement test when its reliability is
#: at least this and its own leader holds at least this share of its support.
OPINION_RELIABILITY = 0.5
OPINION_SHARE = 0.35

#: Runner-up / leader ratio at or above which the leaders are effectively tied.
TIE_RATIO = 0.75
#: Below this confidence the result is uncertain.
MIN_CONFIDENCE = 0.45
MAX_CONFIDENCE = 0.95

_STRONG_PIXELS = 5000
_MODERATE_PIXELS = 1000
_LIMITED_PIXELS = 50
_CANDIDATES_REPORTED = 5


def confidence_level(confidence: float, uncertain: bool) -> str:
    if uncertain:
        return "low"
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.50:
        return "medium"
    return "low"


def evidence_quality(
    flame_pixel_count: int,
    real_mask: bool,
    detection_confidence: float,
) -> str:
    """Grade the measured flame region: strong / moderate / limited / insufficient."""
    pixels = int(flame_pixel_count or 0)
    if pixels < _LIMITED_PIXELS:
        return "insufficient"
    if real_mask and pixels >= _STRONG_PIXELS and detection_confidence >= 0.6:
        return "strong"
    if pixels >= _MODERATE_PIXELS and detection_confidence >= 0.45:
        return "moderate"
    return "limited"


def deterministic_evidence(
    ranking: Sequence[MaterialScore],
    settings: Settings,
    *,
    flame_pixel_count: int,
    mask_area_ratio: float,
    real_mask: bool,
    detection_confidence: float,
    detection_count: int,
    mask_count: int,
    mean_rgb: Sequence[int] | None = None,
    mean_lab: Sequence[float] | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """Normalise the LAB/RGB matcher output into the common evidence form.

    ``quality`` overrides the region grading for colours a client measured
    itself (no mask is available to grade).
    """
    scale = float(settings.similarity_distance_scale)
    distances = {score.material: round((1.0 - float(score.similarity)) * scale, 3) for score in ranking}
    if quality not in QUALITY_RELIABILITY:
        quality = evidence_quality(flame_pixel_count, real_mask, detection_confidence)
    support: dict[str, float] = {}
    best_distance = min(distances.values()) if distances else None
    if distances:
        raw = {name: math.exp(-(dist - best_distance) / TAU_LAB) for name, dist in distances.items()}
        total = sum(raw.values())
        support = {name: value / total for name, value in raw.items()}

    if best_distance is None:
        closeness = 0.0
    elif best_distance <= CLOSE_LAB:
        closeness = 1.0
    elif best_distance >= FAR_LAB:
        closeness = 0.5
    else:
        closeness = 1.0 - 0.5 * (best_distance - CLOSE_LAB) / (FAR_LAB - CLOSE_LAB)
    reliability = QUALITY_RELIABILITY[quality] * closeness

    return {
        "source": "deterministic_lab_rgb",
        "available": bool(support) and quality != "insufficient",
        "leader": ranking[0].material if ranking else None,
        "reliability": round(reliability, 4),
        "evidence_quality": quality,
        "best_lab_distance": best_distance,
        "mean_rgb": list(mean_rgb) if mean_rgb is not None else None,
        "mean_lab": list(mean_lab) if mean_lab is not None else None,
        "flame_pixel_count": int(flame_pixel_count),
        "mask_area_ratio": float(mask_area_ratio),
        "real_segmentation_mask": bool(real_mask),
        "detection_confidence": round(float(detection_confidence), 4),
        "detection_count": int(detection_count),
        "mask_count": int(mask_count),
        "ranking": [
            {
                "material": score.material,
                "similarity": float(score.similarity),
                "lab_distance": distances[score.material],
                "support": round(support.get(score.material, 0.0), 4),
            }
            for score in ranking
        ],
        "score_basis": "CIELAB distance of the merged-mask flame colours to flame_dataset.json references",
    }


def _vision_support(vision: dict[str, Any] | None, materials: Sequence[str]) -> tuple[dict[str, float], float, str | None]:
    """``(support, reliability, leader)`` of the vision evidence."""
    if not vision or not vision.get("available") or not vision.get("candidates"):
        return {}, 0.0, None
    allowed = set(materials)
    candidates = [c for c in vision["candidates"] if c.get("material") in allowed]
    if not candidates:
        return {}, 0.0, None
    assigned = {c["material"]: float(c["confidence"]) for c in candidates}
    remainder = max(0.0, 1.0 - sum(assigned.values()))
    spread = remainder / len(materials) if materials else 0.0
    support = {name: assigned.get(name, 0.0) + spread for name in materials}

    reliability = VISION_BASE_RELIABILITY
    if vision.get("uncertain"):
        reliability *= VISION_UNCERTAIN_FACTOR
    if candidates[0].get("uncertainty") == "high":
        reliability *= VISION_TOP_HIGH_UNCERTAINTY_FACTOR
    if sum(assigned.values()) <= 0.0:
        reliability = 0.0
    return support, reliability, candidates[0]["material"]


def _ordered(scores: dict[str, float], order: Sequence[str]) -> list[tuple[str, float]]:
    """Highest score first; exact ties keep dataset order (never alphabetical)."""
    position = {name: index for index, name in enumerate(order)}
    return sorted(scores.items(), key=lambda item: (-item[1], position.get(item[0], len(order))))


def fuse_material(
    deterministic: dict[str, Any],
    vision: dict[str, Any] | None,
    database: MaterialDatabase,
) -> dict[str, Any]:
    """Combine deterministic and vision evidence into the final material decision."""
    materials = [entry.name for entry in database.entries]
    det_support = {
        row["material"]: float(row["support"])
        for row in deterministic.get("ranking", [])
        if row.get("material") in materials
    }
    det_reliability = float(deterministic.get("reliability", 0.0)) if deterministic.get("available") else 0.0
    vis_support, vis_reliability, vis_leader = _vision_support(vision, materials)

    w_det = BASE_WEIGHTS["deterministic"] * det_reliability
    w_vis = BASE_WEIGHTS["vision"] * vis_reliability
    total_weight = w_det + w_vis
    coverage = total_weight / sum(BASE_WEIGHTS.values())

    combined: dict[str, float] = {}
    if total_weight > 0:
        for name in materials:
            combined[name] = (
                w_det * det_support.get(name, 0.0) + w_vis * vis_support.get(name, 0.0)
            ) / total_weight
    ranked = _ordered(combined, materials)

    det_ranked = _ordered(det_support, materials) if det_support else []
    det_leader = det_ranked[0][0] if det_ranked and w_det > 0 else None
    det_share = det_ranked[0][1] if det_ranked else 0.0
    vis_share = max(vis_support.values()) if vis_support else 0.0

    agreement: bool | None = None
    disagreement = False
    if det_leader and vis_leader:
        agreement = det_leader == vis_leader
        disagreement = (
            not agreement
            and det_reliability >= OPINION_RELIABILITY
            and vis_reliability >= OPINION_RELIABILITY
            and det_share >= OPINION_SHARE
            and vis_share >= OPINION_SHARE
        )

    reasons: list[str] = []
    top_name, top_score = ranked[0] if ranked else (None, 0.0)
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = top_score * (0.5 + 0.5 * coverage) if ranked else 0.0
    if agreement:
        confidence += AGREEMENT_BONUS
    if disagreement:
        confidence *= DISAGREEMENT_FACTOR
        reasons.append(
            f"colour evidence favours {det_leader} but vision evidence favours {vis_leader}"
        )
    confidence = round(max(0.0, min(MAX_CONFIDENCE, confidence)), 4)

    if total_weight <= 0 or not ranked:
        reasons.append("no usable material evidence")
    if deterministic.get("evidence_quality") == "insufficient":
        reasons.append("the segmented flame region is too small")
    tie_ratio = (runner_score / top_score) if top_score > 0 else 1.0
    leading = [name for name, score in ranked if top_score > 0 and score / top_score >= TIE_RATIO]
    if len(leading) > 1:
        reasons.append("leading materials are effectively tied: " + ", ".join(leading))
    if ranked and confidence < MIN_CONFIDENCE:
        reasons.append(f"fused confidence {confidence:.2f} is below {MIN_CONFIDENCE:.2f}")

    uncertain = bool(
        total_weight <= 0
        or not ranked
        or deterministic.get("evidence_quality") == "insufficient"
        or len(leading) > 1
        or confidence < MIN_CONFIDENCE
    )
    final = None if uncertain else top_name

    candidates = [
        {
            "material": name,
            "score": round(score, 4),
            "deterministic_support": round(det_support.get(name, 0.0), 4),
            "vision_support": round(vis_support.get(name, 0.0), 4) if vis_support else None,
        }
        for name, score in ranked[:_CANDIDATES_REPORTED]
    ]
    return {
        "final_material": final,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 1),
        "confidence_level": confidence_level(confidence, uncertain),
        "uncertain": uncertain,
        "uncertainty_reasons": reasons if uncertain or disagreement else [],
        "leading_candidates": leading if leading else ([top_name] if top_name else []),
        "candidate_materials": candidates,
        "supporting_evidence": {
            "deterministic_leader": det_leader,
            "vision_leader": vis_leader,
            "vision_provider": (vision or {}).get("vision_provider", "none"),
            "agreement": agreement,
            "significant_disagreement": disagreement,
            "tie_ratio": round(tie_ratio, 4),
            "weights": {
                "deterministic": round(w_det, 4),
                "vision": round(w_vis, 4),
                "deterministic_reliability": round(det_reliability, 4),
                "vision_reliability": round(vis_reliability, 4),
                "coverage": round(coverage, 4),
            },
        },
        "decided_by": "python_deterministic_fusion",
        "fusion_version": FUSION_VERSION,
    }


# ---------------------------------------------------------------------------
# Video aggregation
# ---------------------------------------------------------------------------
#: Minimum lead of the winning material over the runner-up, as a share of the
#: frames that reached a material decision.
VIDEO_MIN_VOTE_MARGIN = 0.2
#: The winner must be supported by at least this share of analysed frames.
VIDEO_MIN_SUPPORT_SHARE = 0.4
#: Below this share of sampled frames with a detection, confidence is reduced.
VIDEO_MIN_DETECTION_RATE = 0.5


def aggregate_video_frames(
    frames: Sequence[dict[str, Any]],
    sampled_count: int,
    database: MaterialDatabase,
) -> dict[str, Any]:
    """Majority/consistency vote over per-frame fused results.

    ``frames`` are the per-frame results of successfully analysed frames, each
    carrying ``final_material`` (``None`` when that frame was uncertain),
    ``confidence`` and ``fire_class``.  No AI output is averaged here: only
    each frame's Python-fused decision votes.
    """
    order = [entry.name for entry in database.entries]
    position = {name: index for index, name in enumerate(order)}
    analysed = len(frames)
    decided = [frame for frame in frames if frame.get("final_material")]
    uncertain_frames = analysed - len(decided)

    counts = Counter(frame["final_material"] for frame in decided)
    conf_sum: dict[str, float] = {}
    for frame in decided:
        conf_sum[frame["final_material"]] = conf_sum.get(frame["final_material"], 0.0) + float(
            frame.get("confidence") or 0.0
        )
    votes = sorted(
        counts.items(),
        key=lambda item: (-item[1], -conf_sum.get(item[0], 0.0), position.get(item[0], len(order))),
    )
    detection_rate = analysed / float(sampled_count) if sampled_count else 0.0

    reasons: list[str] = []
    leader = votes[0][0] if votes else None
    leader_count = votes[0][1] if votes else 0
    runner_count = votes[1][1] if len(votes) > 1 else 0
    margin = (leader_count - runner_count) / float(len(decided)) if decided else 0.0
    support_share = leader_count / float(analysed) if analysed else 0.0

    if analysed == 0:
        reasons.append("no frame contained an analysable flame")
    elif not decided:
        reasons.append("every analysed frame was individually uncertain")
    if votes and leader_count == runner_count:
        reasons.append(f"tie between {votes[0][0]} and {votes[1][0]}")
    elif decided and margin < VIDEO_MIN_VOTE_MARGIN:
        reasons.append(
            f"leading materials are too close ({leader_count} vs {runner_count} frames)"
        )
    if decided and support_share < VIDEO_MIN_SUPPORT_SHARE:
        reasons.append(
            f"{leader} is supported by only {leader_count} of {analysed} analysed frames"
        )
    uncertain = bool(reasons)

    mean_leader_conf = conf_sum.get(leader, 0.0) / leader_count if leader_count else 0.0
    confidence = mean_leader_conf * (0.5 + 0.5 * support_share)
    notes: list[str] = []
    if detection_rate < VIDEO_MIN_DETECTION_RATE:
        confidence *= 0.85
        notes.append(f"flame detected in only {analysed} of {sampled_count} sampled frames")
    confidence = round(max(0.0, min(MAX_CONFIDENCE, confidence)), 4)

    # Fire class: majority over the frames' own deterministic fire classes.
    class_counts = Counter(
        (frame.get("fire_class") or {}).get("class")
        for frame in frames
        if (frame.get("fire_class") or {}).get("class") not in (None, "Unclassified")
    )
    class_votes = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    fire_class_vote = None
    if class_votes and (len(class_votes) == 1 or class_votes[0][1] > class_votes[1][1]):
        fire_class_vote = class_votes[0][0]

    return {
        "final_material": None if uncertain else leader,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 1),
        "confidence_level": confidence_level(confidence, uncertain),
        "uncertain": uncertain,
        "uncertainty_reasons": reasons,
        "consistency_notes": notes,
        "leading_candidates": [name for name, count in votes if count == leader_count] if votes else [],
        "votes": [
            {
                "material": name,
                "frames": count,
                "mean_confidence": round(conf_sum.get(name, 0.0) / count, 4),
            }
            for name, count in votes
        ],
        "uncertain_frames": uncertain_frames,
        "frames_sampled": int(sampled_count),
        "frames_analyzed": analysed,
        "frames_with_decision": len(decided),
        "detection_rate": round(detection_rate, 4),
        "vote_margin": round(margin, 4),
        "support_share": round(support_share, 4),
        "fire_class_votes": [{"class": name, "frames": count} for name, count in class_votes],
        "fire_class_vote": fire_class_vote,
        "decided_by": "python_majority_vote",
        "fusion_version": FUSION_VERSION,
    }


def video_fire_class(
    database: MaterialDatabase,
    aggregate: dict[str, Any],
    frame_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministic video fire class and agents from the frame results."""
    final = aggregate["final_material"]
    if final:
        fire_class, agents = classify(database, final, aggregate["confidence"])
        payload = jsonable(dataclasses.asdict(fire_class))
        payload["class"] = payload.pop("class_")
        payload["basis"] = (
            f"video material {final} (majority of frame decisions) -> {payload['class']} "
            f"via {MAPPING_SOURCE}"
        )
        return payload, [jsonable(dataclasses.asdict(agent)) for agent in agents]

    voted = aggregate.get("fire_class_vote")
    if not voted:
        return (
            {
                "class": UNKNOWN_CLASS,
                "description": CLASS_DESCRIPTIONS[UNKNOWN_CLASS],
                "confidence": 0.0,
                "material": None,
                "basis": "the frames do not agree on one fire class",
                "mapping_source": MAPPING_SOURCE,
                "notes": "",
            },
            [],
        )
    # Agents every frame of the winning class recommends, in first-seen order.
    agent_lists = [
        r.get("extinguishing_agents") or []
        for r in frame_results
        if r.get("success") and (r.get("fire_class") or {}).get("class") == voted
    ]
    shared = None
    for agents in agent_lists:
        names = {a.get("name") for a in agents}
        shared = names if shared is None else shared & names
    first = agent_lists[0] if agent_lists else []
    agents = [dict(a) for a in first if a.get("name") in (shared or set())]
    votes = {item["class"]: item["frames"] for item in aggregate.get("fire_class_votes", [])}
    classified = sum(votes.values()) or 1
    return (
        {
            "class": voted,
            "description": CLASS_DESCRIPTIONS.get(voted, ""),
            "confidence": round(votes.get(voted, 0) / float(classified), 4),
            "material": None,
            "basis": (
                f"material is uncertain; {votes.get(voted, 0)} of {classified} classified "
                f"frames map to {voted} via {MAPPING_SOURCE}"
            ),
            "mapping_source": MAPPING_SOURCE,
            "notes": "Confidence is the share of classified frames in the winning class.",
        },
        agents,
    )
