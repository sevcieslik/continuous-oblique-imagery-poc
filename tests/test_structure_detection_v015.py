import math

import numpy as np
import pytest

import structure_detection_v015 as sd


def test_containment_nms_removes_nested_boxes():
    dets = [
        {"score": 0.50, "box": (100, 100, 300, 400)},   # whole tower
        {"score": 0.45, "box": (150, 300, 250, 400)},   # base only (nested, IoU low)
        {"score": 0.40, "box": (100, 100, 300, 160)},   # crossarm only (nested)
        {"score": 0.30, "box": (800, 100, 1000, 400)},  # separate tower
    ]
    plain = sd.containment_nms(dets, iou_threshold=0.4, iomin_threshold=1.01)
    assert len(plain) == 4  # IoU alone keeps nested boxes
    kept = sd.containment_nms(dets, iou_threshold=0.4, iomin_threshold=0.8)
    assert [d["box"] for d in kept] == [(100, 100, 300, 400), (800, 100, 1000, 400)]
    assert kept[0]["suppressed_count"] == 2


def test_tile_artifact_full_height_rejected_but_tower_kept():
    tile = (0, 0, 2304, 2304)
    strip = {"score": 0.6, "box": (500, 0, 560, 2304), "tile_box": tile}
    tower = {"score": 0.4, "box": (900, 800, 1300, 1400), "tile_box": tile}
    keep, rej = sd.filter_tile_artifacts([strip, tower])
    assert keep == [tower]
    assert rej[0]["plausibility_rejection"] == "TILE_FULL_HEIGHT_ARTEFACT"


def _tower(x, y, n=200, h=120.0, r=4.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        x + rng.uniform(-r, r, n), y + rng.uniform(-r, r, n), rng.uniform(5, 5 + h, n)
    ])


def test_cluster_tower_points_separates_parallel_towers_and_drops_wires():
    a = _tower(0, 0, seed=1)
    b = _tower(165, 0, seed=2)                    # parallel circuit ~50 m away (ft units)
    wire = np.column_stack([np.linspace(20, 140, 80), np.zeros(80), np.full(80, 90.0)])  # flat conductor
    clusters = sd.cluster_tower_points(np.vstack([a, b, wire]), cell_size=6.0, min_points=25, min_vertical_range=15)
    xs = sorted(round(c["x"]) for c in clusters)
    assert len(clusters) == 2
    assert abs(xs[0] - 0) < 5 and abs(xs[1] - 165) < 5


def test_split_target_and_competitors():
    clusters = [{"x": 0, "y": 0}, {"x": 165, "y": 0}, {"x": 0, "y": 800}]
    target, comps = sd.split_target_and_competitors(
        clusters, (3, 2), target_radius=55, known_target_xys=[(3, 2), (0, 805)])
    assert target is clusters[0]
    assert comps == [clusters[1]]  # cluster at y=800 is another known target, not a competitor


def test_bias_tracker_robust_median_and_global_fallback():
    t = sd.PriorBiasTracker(window=5, min_samples=3)
    assert t.get("p1")["source"] == "NONE"
    for dx in (100, 110, 105, 3000):  # one outlier
        t.add("p1", dx, -1000)
    b = t.get("p1")
    assert b["source"] == "PASS" and 100 <= b["dx"] <= 110 and b["dy"] == -1000
    for _ in range(2):
        t.add("p2", 0, 0)
    assert t.get("p3")["source"] == "GLOBAL"
    t.add("p1", 10_000, 0)  # beyond max_abs_offset_px, ignored
    assert t.get("p1")["n"] == 4


def test_ownership_is_invariant_to_common_offset():
    # True image positions: target tower at (1000, 1000), parallel tower at (1000 - 1850, 1000).
    # Both priors are shifted by the same EO error (+400, -1100).
    off = (400, -1100)
    hyps = [
        sd.Hypothesis("T", (1000 + off[0], 1000 + off[1]), "TARGET"),
        sd.Hypothesis("C", (-850 + off[0], 1000 + off[1]), "COMPETITOR"),
    ]
    target_box = (900, 850, 1100, 1150)
    comp_box = (-950, 850, -750, 1150)
    own_t = sd.ownership_probabilities(target_box, hyps, sigma_px=800)
    own_c = sd.ownership_probabilities(comp_box, hyps, sigma_px=800)
    assert own_t["T"] > 0.8
    assert own_c["C"] > 0.8


def _pair(t, d, final, own, owner, centre=(0.0, 0.0)):
    return sd.PairCandidate(t, d, final, own, owner, 0.4, 0.0, 0.0, float("nan"), centre)


def test_resolve_assignments_statuses():
    pairs = [
        _pair("T:a", 0, 0.70, 0.90, "T:a", (0, 0)),
        _pair("T:b", 1, 0.80, 0.10, "C:x", (5000, 0)),          # only competitor-owned
        _pair("T:c", 2, 0.60, 0.70, "T:c", (0, 3000)),
        _pair("T:c", 3, 0.58, 0.70, "T:c", (2000, 3000)),       # separated, near-equal -> ambiguous
        _pair("T:d", 4, 0.20, 0.90, "T:d", (0, 9000)),          # low score
        _pair("T:e", 5, 0.60, 0.70, "T:e", (0, 7000)),
        _pair("T:e", 6, 0.59, 0.70, "T:e", (50, 7000)),         # duplicate boxes, not ambiguous
    ]
    out, status = sd.resolve_assignments(pairs, min_score=0.3, min_ownership=0.55, ambiguity_margin=0.05,
                                         ambiguity_min_separation_px=300)
    assert set(out) == {"T:a", "T:e"}
    assert out["T:e"].det_index == 5
    assert status["T:b"] == "REJECTED_COMPETITOR_OWNED"
    assert status["T:c"] == "AMBIGUOUS_ASSIGNMENT"
    assert status["T:d"] == "LOW_ASSIGNMENT_SCORE"


def test_resolve_one_to_one():
    pairs = [
        _pair("T:a", 0, 0.9, 0.9, "T:a"),
        _pair("T:b", 0, 0.8, 0.9, "T:b"),
    ]
    out, status = sd.resolve_assignments(pairs, 0.3, 0.55, 0.05)
    assert list(out) == ["T:a"]
    assert status["T:b"] == "DETECTION_TAKEN_BY_OTHER_TARGET"


def test_score_prefers_ownership_over_detector_confidence():
    right = sd.score_candidate_v15(det_score=0.40, ownership=0.85, geom_abs=0.85)
    wrong = sd.score_candidate_v15(det_score=0.50, ownership=0.15, geom_abs=0.80)
    assert right > wrong
    assert math.isclose(sd.gaussian_geometry(0, 10), 1.0)
