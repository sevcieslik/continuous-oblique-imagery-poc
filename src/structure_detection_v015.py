#!/usr/bin/env python
r"""
Structure-detection helpers for continuous_oblique_v015.py.

Pure numpy/python functions only (no torch, no geopandas) so they can be unit-tested
without model weights or project data.

Main ideas introduced in v0.15
------------------------------
1. Competitor-aware assignment.
   The v0.14 geometry prior can be offset by 1000-1500 px (EO/OPK error). On corridors with
   parallel circuits (e.g. NYPA DH1 + DH2, ~50 m apart) that offset is comparable to the spacing
   between towers, so "closest detection to the prior" picks the neighbouring circuit.
   v0.15 projects *every* plausible tower in the neighbourhood (other target structures in the
   image, LiDAR tower clusters, optional neighbour-circuit structure CSV) with the same camera
   model. A detection is owned by the hypothesis it is closest to. A common projection error
   shifts all hypotheses together, so relative geometry still identifies the right tower.

   With offsets larger than half the tower spacing nearest-neighbour ownership is not enough, so
   estimate_image_translation() first registers the whole hypothesis set to the detection set
   (common translation supported by several hypotheses), then ownership is computed.

2. Online prior-bias calibration.
   Confident, unambiguous detections measure the projection offset (detected centre - prior).
   A robust rolling median per acquisition pass is applied to later priors in the same pass.

3. Detector clean-up.
   Tile-height artefacts (shadow/conductor strips clipped at tile edges) are rejected and
   nested duplicate boxes are suppressed with an intersection-over-minimum criterion.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DETECTION_MODULE_VERSION = "0.15.0"


# -----------------------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------------------

def box_area(box) -> float:
    x0, y0, x1, y1 = map(float, box)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_center(box) -> tuple[float, float]:
    x0, y0, x1, y1 = map(float, box)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def box_intersection(a, b) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def box_iou(a, b) -> float:
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    den = box_area(a) + box_area(b) - inter
    return inter / den if den > 0 else 0.0


def box_iomin(a, b) -> float:
    """Intersection over the smaller box area. 1.0 when one box is inside the other."""
    inter = box_intersection(a, b)
    if inter <= 0:
        return 0.0
    den = min(box_area(a), box_area(b))
    return inter / den if den > 0 else 0.0


def point_to_box_distance(pt, box) -> float:
    try:
        px, py = float(pt[0]), float(pt[1])
        x0, y0, x1, y1 = map(float, box)
    except Exception:
        return float("nan")
    if not all(math.isfinite(v) for v in (px, py, x0, y0, x1, y1)):
        return float("nan")
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def containment_nms(
    dets: list[dict],
    iou_threshold: float = 0.40,
    iomin_threshold: float = 0.80,
    max_boxes: int = 24,
) -> list[dict]:
    """Greedy NMS that also suppresses boxes nested inside a higher-scoring box.

    Grounding DINO with several synonymous prompts and overlapping tiles produces nested
    boxes (whole tower, crossarm only, base only). Plain IoU-NMS keeps all of them because
    their IoU is low. Suppression happens when IoU >= iou_threshold OR IoMin >= iomin_threshold.
    """
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    keep: list[dict] = []
    for d in ordered:
        box = d.get("box")
        if box is None or box_area(box) <= 0:
            continue
        suppressed = False
        for k in keep:
            if box_iou(box, k["box"]) >= float(iou_threshold) or box_iomin(box, k["box"]) >= float(iomin_threshold):
                suppressed = True
                k.setdefault("suppressed_count", 0)
                k["suppressed_count"] += 1
                break
        if suppressed:
            continue
        keep.append(dict(d))
        if len(keep) >= int(max_boxes):
            break
    return keep


def tile_artifact_reason(det: dict, full_span_fraction: float = 0.96, edge_px: float = 6.0) -> str:
    """Return a non-empty reason when a box looks like a tile-clipping artefact.

    Long shadows and conductor bundles are often returned as boxes that run from the top to the
    bottom (or left to right) edge of a detector tile. A real tower inside a 2304 px tile almost
    never touches two opposite tile edges.
    """
    tb = det.get("tile_box")
    if tb is None:
        return ""
    try:
        tx0, ty0, tx1, ty1 = map(float, tb)
        x0, y0, x1, y1 = map(float, det["box"])
    except Exception:
        return ""
    tw, th = max(1.0, tx1 - tx0), max(1.0, ty1 - ty0)
    touches_top = y0 <= ty0 + edge_px
    touches_bottom = y1 >= ty1 - edge_px
    touches_left = x0 <= tx0 + edge_px
    touches_right = x1 >= tx1 - edge_px
    if touches_top and touches_bottom and (y1 - y0) >= full_span_fraction * th:
        return "TILE_FULL_HEIGHT_ARTEFACT"
    if touches_left and touches_right and (x1 - x0) >= full_span_fraction * tw:
        return "TILE_FULL_WIDTH_ARTEFACT"
    return ""


def filter_tile_artifacts(dets: list[dict], full_span_fraction: float = 0.96) -> tuple[list[dict], list[dict]]:
    keep, reject = [], []
    for d in dets:
        reason = tile_artifact_reason(d, full_span_fraction)
        if reason:
            dd = dict(d)
            dd["plausibility_rejection"] = reason
            reject.append(dd)
        else:
            keep.append(d)
    return keep, reject


# -----------------------------------------------------------------------------
# LiDAR tower clustering (competitor hypotheses)
# -----------------------------------------------------------------------------

def cluster_tower_points(
    xyz: np.ndarray,
    cell_size: float = 6.0,
    min_points: int = 25,
    min_vertical_range: float = 15.0,
) -> list[dict]:
    """Group structure-class LiDAR points into tower-like clusters.

    A simple occupancy-grid connected-components clustering in XY. Each returned cluster has
    centroid XY, base/top Z, point count and the member points (for projection).
    Clusters with a small vertical range (e.g. conductor fragments) are dropped.
    """
    if xyz is None or len(xyz) == 0:
        return []
    xyz = np.asarray(xyz, dtype=float)
    cs = max(float(cell_size), 1e-6)
    ij = np.floor(xyz[:, :2] / cs).astype(np.int64)
    cell_members: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (i, j) in enumerate(ij):
        cell_members[(int(i), int(j))].append(idx)

    seen: set[tuple[int, int]] = set()
    clusters: list[dict] = []
    for start in cell_members:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            c = stack.pop()
            members.extend(cell_members[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (c[0] + di, c[1] + dj)
                    if n in cell_members and n not in seen:
                        seen.add(n)
                        stack.append(n)
        pts = xyz[members]
        if len(pts) < int(min_points):
            continue
        z2, z98 = float(np.percentile(pts[:, 2], 2)), float(np.percentile(pts[:, 2], 98))
        if (z98 - z2) < float(min_vertical_range):
            continue
        clusters.append({
            "x": float(np.median(pts[:, 0])),
            "y": float(np.median(pts[:, 1])),
            "base_z": z2,
            "top_z": z98,
            "height": z98 - z2,
            "n": int(len(pts)),
            "xyz": pts,
        })
    clusters.sort(key=lambda c: -c["n"])
    return clusters


def split_target_and_competitors(
    clusters: list[dict],
    target_xy: tuple[float, float],
    target_radius: float,
    known_target_xys: Sequence[tuple[float, float]] = (),
    known_radius: float | None = None,
) -> tuple[dict | None, list[dict]]:
    """Return (target cluster, competitor clusters).

    The target cluster is the nearest cluster to the staking XY within target_radius.
    Clusters that coincide with *other* known target structures (same circuit) are not
    returned as competitors because they are already separate hypotheses.
    """
    tx, ty = map(float, target_xy)
    known_radius = float(known_radius if known_radius is not None else target_radius)
    target = None
    best = float("inf")
    for c in clusters:
        d = math.hypot(c["x"] - tx, c["y"] - ty)
        if d <= float(target_radius) and d < best:
            best, target = d, c
    competitors = []
    for c in clusters:
        if c is target:
            continue
        if any(math.hypot(c["x"] - kx, c["y"] - ky) <= known_radius for kx, ky in known_target_xys):
            continue
        competitors.append(c)
    return target, competitors


# -----------------------------------------------------------------------------
# Online prior-bias calibration
# -----------------------------------------------------------------------------

@dataclass
class PriorBiasTracker:
    """Robust rolling median of (detected centre - geometry prior) per key (pass/run).

    Only confident, unambiguous assignments should be recorded. The correction is applied when
    at least min_samples observations exist for a key. A global fallback (all keys) is used when a
    key has too few samples but the global pool is large enough.
    """
    window: int = 15
    min_samples: int = 3
    max_abs_offset_px: float = 4000.0
    _data: dict = field(default_factory=lambda: defaultdict(deque))
    _global: deque = field(default_factory=deque)

    def add(self, key, dx: float, dy: float) -> None:
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return
        if math.hypot(dx, dy) > self.max_abs_offset_px:
            return
        q = self._data[key]
        q.append((float(dx), float(dy)))
        while len(q) > self.window:
            q.popleft()
        self._global.append((float(dx), float(dy)))
        while len(self._global) > self.window * 4:
            self._global.popleft()

    @staticmethod
    def _robust(samples) -> tuple[float, float, float]:
        arr = np.asarray(samples, dtype=float)
        med = np.median(arr, axis=0)
        spread = float(np.median(np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1]))) * 1.4826
        return float(med[0]), float(med[1]), spread

    def get(self, key) -> dict:
        q = self._data.get(key)
        if q and len(q) >= self.min_samples:
            dx, dy, spread = self._robust(q)
            return {"dx": dx, "dy": dy, "n": len(q), "spread": spread, "source": "PASS"}
        if len(self._global) >= self.min_samples * 2:
            dx, dy, spread = self._robust(self._global)
            return {"dx": dx, "dy": dy, "n": len(self._global), "spread": spread, "source": "GLOBAL"}
        return {"dx": 0.0, "dy": 0.0, "n": 0, "spread": float("nan"), "source": "NONE"}


def apply_bias(prior_xy, bias: dict):
    if prior_xy is None:
        return None
    return (float(prior_xy[0]) + float(bias.get("dx", 0.0)), float(prior_xy[1]) + float(bias.get("dy", 0.0)))


# -----------------------------------------------------------------------------
# Hypothesis ownership and scoring
# -----------------------------------------------------------------------------

@dataclass
class Hypothesis:
    key: str               # unique id, e.g. "T:233" (target) or "C:233:1" (competitor)
    xy: tuple[float, float]
    kind: str              # TARGET or COMPETITOR
    label: str = ""


def ownership_probabilities(det_box, hypotheses: Sequence[Hypothesis], sigma_px: float) -> dict[str, float]:
    """Soft assignment of a detection to hypotheses by centre distance (Gaussian kernel).

    Uses the detection centre (the prior is the median of projected tower points, i.e. roughly the
    tower's visual centre).
    """
    if not hypotheses:
        return {}
    cx, cy = box_center(det_box)
    s = max(1.0, float(sigma_px))
    logits = []
    for h in hypotheses:
        d = math.hypot(cx - h.xy[0], cy - h.xy[1])
        logits.append(-0.5 * (d / s) ** 2)
    m = max(logits)
    w = [math.exp(l - m) for l in logits]
    tot = sum(w) or 1.0
    return {h.key: wi / tot for h, wi in zip(hypotheses, w)}


@dataclass
class Registration:
    dx: float = 0.0
    dy: float = 0.0
    support: float = 0.0          # summed match strength of the best translation
    matched_hypotheses: int = 0   # hypotheses with a detection within ~2 sigma after translation
    second_support: float = 0.0   # best support of a translation that is clearly different
    accepted: bool = False
    status: str = "NOT_RUN"


def estimate_image_translation(
    hypotheses: Sequence[Hypothesis],
    det_boxes: Sequence,
    det_scores: Sequence[float],
    sigma_px: float,
    max_shift_px: float,
    min_matched: int = 2,
    min_margin: float = 0.35,
    expected_shift=(0.0, 0.0),
    expected_sigma_px: float | None = None,
) -> Registration:
    """Find the common pixel translation that best aligns projected hypotheses with detections.

    Every (hypothesis, detection) pair proposes a translation. Each proposal is scored by how many
    hypotheses land on *some* detection after the shift (Gaussian kernel, sigma_px, weighted by
    sqrt(detector score)). Because the target *and* its neighbours (parallel circuit, adjacent
    structures) move together under a common EO/OPK error, the correct translation is supported by
    several hypotheses while a wrong one (target snapped onto the neighbour tower) is supported by
    one. This is what makes the assignment invariant to offsets larger than half the tower spacing.

    The translation is accepted only when >= min_matched hypotheses match and the best support
    exceeds the best clearly-different alternative by min_margin. Otherwise status explains why and
    the caller keeps the (bias-corrected) priors.
    """
    reg = Registration()
    hyps = [h for h in hypotheses if h.xy is not None and all(math.isfinite(float(v)) for v in h.xy)]
    centres = [box_center(b) for b in det_boxes]
    if len(hyps) < 2:
        reg.status = "TOO_FEW_HYPOTHESES"
        return reg
    if len(centres) < 1:
        reg.status = "NO_DETECTIONS"
        return reg
    s = max(1.0, float(sigma_px))
    w = [math.sqrt(max(0.0, float(x))) for x in det_scores]
    H = np.asarray([h.xy for h in hyps], dtype=float)
    D = np.asarray(centres, dtype=float)
    W = np.asarray(w, dtype=float)
    ex = np.asarray(expected_shift, dtype=float)

    def support(t):
        moved = H + t
        d2 = ((moved[:, None, :] - D[None, :, :]) ** 2).sum(axis=2)
        k = np.exp(-0.5 * d2 / (s * s)) * W[None, :]
        best = k.max(axis=1)
        matched = int((d2.min(axis=1) <= (2.0 * s) ** 2).sum())
        val = float(best.sum())
        if expected_sigma_px and math.isfinite(float(expected_sigma_px)):
            # Mild preference for shifts consistent with the pass-level bias estimate.
            val *= 0.85 + 0.15 * math.exp(-0.5 * float(((t - ex) ** 2).sum()) / float(expected_sigma_px) ** 2)
        return val, matched

    proposals = []
    for i in range(len(H)):
        for j in range(len(D)):
            t = D[j] - H[i]
            if math.hypot(t[0], t[1]) > float(max_shift_px):
                continue
            proposals.append(t)
    if not proposals:
        reg.status = "NO_SHIFT_WITHIN_LIMIT"
        return reg
    scored = []
    for t in proposals:
        val, matched = support(t)
        scored.append((val, matched, t))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_val, best_matched, best_t = scored[0]
    # Refine: mean shift of matched pairs.
    moved = H + best_t
    d2 = ((moved[:, None, :] - D[None, :, :]) ** 2).sum(axis=2)
    nn = d2.argmin(axis=1)
    ok = d2[np.arange(len(H)), nn] <= (2.0 * s) ** 2
    if ok.any():
        best_t = (D[nn[ok]] - H[ok]).mean(axis=0)
        best_val, best_matched = support(best_t)
    second = 0.0
    for val, matched, t in scored[1:]:
        if math.hypot(*(t - best_t)) > 2.0 * s:
            second = val
            break
    reg.dx, reg.dy = float(best_t[0]), float(best_t[1])
    reg.support, reg.matched_hypotheses, reg.second_support = best_val, int(best_matched), float(second)
    if best_matched < int(min_matched):
        reg.status = "INSUFFICIENT_MATCHES"
    elif best_val - second < float(min_margin):
        reg.status = "AMBIGUOUS_TRANSLATION"
    else:
        reg.accepted = True
        reg.status = "ACCEPTED"
    return reg


def shift_hypotheses(hypotheses: Sequence[Hypothesis], dx: float, dy: float) -> list[Hypothesis]:
    return [Hypothesis(h.key, (h.xy[0] + dx, h.xy[1] + dy), h.kind, h.label) for h in hypotheses]


def score_candidate_v15(
    det_score: float,
    ownership: float,
    geom_abs: float,
    dino_sim: float | None = None,
    edge_touch: bool = False,
) -> float:
    """Combined score for a (structure, detection) pair.

    ownership dominates geometry because it is invariant to a common projection offset.
    """
    det_score = float(det_score)
    ownership = float(ownership)
    geom_abs = float(geom_abs)
    if dino_sim is not None and math.isfinite(float(dino_sim)):
        dino01 = 0.5 * (float(dino_sim) + 1.0)
        final = 0.30 * det_score + 0.40 * ownership + 0.10 * geom_abs + 0.20 * dino01
    else:
        final = 0.38 * det_score + 0.47 * ownership + 0.15 * geom_abs
    return final * (0.94 if edge_touch else 1.0)


def gaussian_geometry(dist_px: float, sigma_px: float) -> float:
    if not math.isfinite(dist_px):
        return 0.0
    return math.exp(-0.5 * (float(dist_px) / max(1.0, float(sigma_px))) ** 2)


@dataclass
class PairCandidate:
    target_key: str
    det_index: int
    final: float
    ownership: float
    owner_key: str
    det_score: float
    centre_dist: float
    box_dist: float
    dino_sim: float = float("nan")
    det_center: tuple[float, float] = (float("nan"), float("nan"))


def resolve_assignments(
    pairs: list[PairCandidate],
    min_score: float,
    min_ownership: float,
    ambiguity_margin: float,
    ambiguity_min_separation_px: float = 300.0,
) -> tuple[dict[str, PairCandidate], dict[str, str]]:
    """One-to-one greedy assignment with explicit competitor/ambiguity statuses.

    Returns (assignments by target, status by target for targets with no assignment).
    Status values:
      REJECTED_COMPETITOR_OWNED  every in-gate detection is owned by a competitor hypothesis;
      AMBIGUOUS_ASSIGNMENT        two candidate detections are too close in score;
      LOW_ASSIGNMENT_SCORE        candidates exist but none reaches min_score.
    """
    by_target: dict[str, list[PairCandidate]] = defaultdict(list)
    for p in pairs:
        by_target[p.target_key].append(p)

    status: dict[str, str] = {}
    eligible: list[PairCandidate] = []
    for t, plist in by_target.items():
        owned = [p for p in plist if p.owner_key == t and p.ownership >= float(min_ownership)]
        if not owned:
            status[t] = "REJECTED_COMPETITOR_OWNED"
            continue
        owned.sort(key=lambda p: p.final, reverse=True)
        if owned[0].final < float(min_score):
            status[t] = "LOW_ASSIGNMENT_SCORE"
            continue
        if len(owned) > 1 and (owned[0].final - owned[1].final) < float(ambiguity_margin):
            # Two separate visual objects with near-equal evidence. Duplicate boxes of the same
            # tower (centres close together) are not treated as ambiguous.
            a, b = owned[0], owned[1]
            sep = math.hypot(a.det_center[0] - b.det_center[0], a.det_center[1] - b.det_center[1])
            if not math.isfinite(sep) or sep > float(ambiguity_min_separation_px):
                status[t] = "AMBIGUOUS_ASSIGNMENT"
                continue
        eligible.extend(p for p in owned if p.final >= float(min_score))

    eligible.sort(key=lambda p: p.final, reverse=True)
    used_t: set[str] = set()
    used_d: set[int] = set()
    out: dict[str, PairCandidate] = {}
    for p in eligible:
        if p.target_key in used_t or p.det_index in used_d or p.target_key in status:
            continue
        out[p.target_key] = p
        used_t.add(p.target_key)
        used_d.add(p.det_index)
    for t in by_target:
        if t not in out and t not in status:
            status[t] = "DETECTION_TAKEN_BY_OTHER_TARGET"
    return out, status

