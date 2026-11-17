"""End-to-end test of render_candidate_products_and_crops_v15 on a synthetic parallel-circuit corridor.

Scenario (mirrors DH1 + DH2 near SR1-60/04-05):
- two target towers of the processed circuit 800 ft apart along the line;
- a parallel circuit 165 ft to the side (two more towers);
- the EO-derived prior is offset by (-250, -300) px, i.e. more than half of the ~344 px tower spacing,
  so the target prior is *closer* to the neighbouring circuit's tower;
- the detector scores the neighbouring towers higher than the target towers;
- a tile-height shadow artefact is also returned.

v0.14-style assignment (no competitors, no registration) picks the wrong circuit. v0.15 must pick
the correct towers, record competitor/registration evidence and learn the pass bias.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import box as shp_box  # noqa: E402

import continuous_oblique_v015 as co  # noqa: E402

W, H = 4000, 3000
F = 3000.0
CAM = (0.0, 0.0, 1500.0)
OFFSET = np.array([-250.0, -300.0])
TARGETS = {"SR1-60/04": (0.0, -400.0), "SR1-60/05": (0.0, 400.0)}
NEIGHBOURS = {"DH2-A": (165.0, -400.0), "DH2-B": (165.0, 400.0)}


def _cam():
    return {"rotation": np.eye(3), "M": np.array([[F, 0.0, W / 2], [0.0, -F, H / 2]]),
            "quality": "HIGH", "rms_px": 1.0, "rms_fraction": 0.0002, "center_offset_fraction": 0.0,
            "corner_order": "TL,TR,BR,BL"}


def _tower_pts(x, y, seed):
    rng = np.random.default_rng(seed)
    n = 300
    return np.column_stack([x + rng.uniform(-4, 4, n), y + rng.uniform(-4, 4, n), rng.uniform(5, 125, n)])


ALL_TOWERS = {**TARGETS, **NEIGHBOURS}
POINTS = {k: _tower_pts(x, y, i) for i, (k, (x, y)) in enumerate(ALL_TOWERS.items())}


def _project_median(xyz):
    uv, valid = co.project_xyz_array_to_pixels(xyz, CAM, _cam())
    return np.median(uv[valid], axis=0)


TRUE_PX = {k: _project_median(v) + OFFSET for k, v in POINTS.items()}


class FakeAccessor:
    def query_radius(self, x, y, r, classes=None):
        allp = np.vstack(list(POINTS.values()))
        m = (allp[:, 0] - x) ** 2 + (allp[:, 1] - y) ** 2 <= r * r
        p = allp[m]
        return {"x": p[:, 0], "y": p[:, 1], "z": p[:, 2], "classification": np.full(len(p), 215)}

    def ground_z(self, x, y, ground_class=2, radius=75.0):
        return 0.0, 100


LIDAR = {"accessor": FakeAccessor(), "ground_class": 2, "ground_radius": 75.0, "structure_class": 215,
         "structure_radius": 55.0, "min_height": 15.0}


def fake_prior(cr, footprint, sw, sh, lidar_context, opk_convention, cloud_cache):
    sid = str(cr["structure_id"])
    xyz = POINTS[sid]
    cloud = {"status": "OK", "xyz": xyz, "ground_z": 0.0, "base_z": 5.0, "top_z": 125.0, "height": 120.0}
    return tuple(_project_median(xyz)), _cam(), cloud


def fake_detector(detector, im, prompts, thr, text_thr, tile, overlap, batch, max_tiles, rois, nms, maxb):
    tb = (0, 0, 2304, 2304)
    dets = []
    for k, (u, v) in TRUE_PX.items():
        score = 0.50 if k.startswith("DH2") else 0.40
        dets.append({"score": score, "label": "lattice tower", "box": (u - 90, v - 150, u + 90, v + 150),
                     "tile_index": 0, "tile_box": tb, "edge_touch": 0})
        # nested crossarm-only duplicate
        dets.append({"score": score - 0.05, "label": "tower", "box": (u - 90, v - 150, u + 90, v - 80),
                     "tile_index": 0, "tile_box": tb, "edge_touch": 0})
    dets.append({"score": 0.60, "label": "tower", "box": (1500, 0, 1560, 2304),
                 "tile_index": 0, "tile_box": tb, "edge_touch": 1})  # shadow strip artefact
    return dets, {"tile_count": 4, "tiles_with_detections": 1, "raw_detection_count": len(dets)}


@pytest.fixture()
def scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "_load_tower_detector_v12", lambda *a, **k: {"backend": "grounding_dino", "device_name": "cpu"})
    monkeypatch.setattr(co, "_run_local_roi_tower_detector_v14", fake_detector)
    monkeypatch.setattr(co, "_rough_geometry_prior_for_structure", fake_prior)
    rows, man_rows, fps = [], [], []
    for i in range(3):
        fn = f"IMG_{i:03d}.jpg"
        p = tmp_path / fn
        Image.new("RGB", (W, H), (90, 110, 80)).save(p, "JPEG", quality=70)
        fps.append({"filename": fn, "geometry": shp_box(-900, -700, 900, 700)})
        man_rows.append({"filename": fn, "raw_path": str(p), "generated_filename": f"DH1_x_{i}.jpg",
                         "product_direction": "backward", "reference_id": "233-234", "capture_date_code": "042226"})
        for j, (sid, (x, y)) in enumerate(TARGETS.items()):
            rows.append({"structure_idx": 233 + j, "reference_idx": 233 + j, "structure_id": sid, "x": x, "y": y,
                         "filename": fn, "raw_path": str(p), "pass_id": "P1", "circuit_run_id": "R1",
                         "render_by_default": True, "crop_bucket": "backward", "camera_x": CAM[0], "camera_y": CAM[1],
                         "camera_z": CAM[2], "omega": 0.0, "phi": 0.0, "kappa": 0.0,
                         "crop_render_status": "NOT_RUN", "crop_projection_status": "NOT_RUN"})
    crops = pd.DataFrame(rows)
    man = pd.DataFrame(man_rows)
    fp = geopandas.GeoDataFrame(fps, geometry="geometry")
    return tmp_path, man, crops, fp


def _cfg(**over):
    c = {
        "model": "fake", "device": "cpu", "backend": "grounding_dino", "threshold": 0.18, "text_threshold": 0.15,
        "tile_size": 2304, "tile_overlap": 0.2, "tile_batch_size": 2, "max_tiles": 64,
        "prompts": ["steel lattice transmission tower"], "nms_iou": 0.40, "nms_iomin": 0.80, "max_boxes": 24,
        "tile_artifact_fraction": 0.96, "min_aspect": 0.18, "max_aspect": 3.5, "max_area_fraction": 0.04,
        "max_width_fraction": 0.16, "max_height_fraction": 0.30, "roi_radius_fraction": 0.16,
        "roi_min_radius": 1400, "roi_max_radius": 3400, "max_roi_area_fraction": 0.35,
        "max_roi_width_fraction": 0.85, "max_roi_height_fraction": 0.95, "gate_fraction": 0.30,
        "min_assignment_score": 0.24, "min_ownership": 0.55, "ownership_sigma_fraction": 0.10,
        "ambiguity_margin": 0.05, "ambiguity_min_separation_px": 300.0, "lidar_competitors": True,
        "competitor_radius": 302.5, "cluster_cell_size": None, "cluster_min_points": 25,
        "competitor_default_half_height": 50.0, "registration": True, "registration_sigma_fraction": 0.04,
        "registration_min_matched": 2, "registration_min_margin": 0.35, "bias_calibration": True,
        "bias_window": 15, "bias_min_samples": 3, "bias_min_ownership": 0.80, "bias_min_detector_score": 0.30,
        "dino_enabled": False, "dino_model": "", "dino_device": "cpu",
    }
    c.update(over)
    return c


def _run(scn, cfg, competitors=None):
    out, man, crops, fp = scn
    return co.render_candidate_products_and_crops_v15(
        man, crops, fp, out / "run", "DH1", "span", "photogrammetry", "none", 2048, 90, 1.0, 400, 800,
        False, False, None, 50, 50, LIDAR, cfg, competitors)


def _chosen_centres(crops):
    r = crops[crops["crop_render_status"].eq("RENDERED")]
    return {(row.filename, row.structure_id): ((row.tower_box_xmin + row.tower_box_xmax) / 2,
                                                (row.tower_box_ymin + row.tower_box_ymax) / 2)
            for row in r.itertuples()}


def test_v14_assignment_picks_parallel_circuit(scenario):
    """The real v0.14 function, same detections and priors: it crops the neighbouring circuit."""
    out, man, crops, fp = scenario
    crops = co._ensure_v15_crop_columns(crops)  # columns the real crop-candidate builder provides
    _, res, _ = co.render_candidate_products_and_crops_v14(
        man, crops, fp, out / "v14", "DH1", "span", "photogrammetry", "none", 2048, 90, 1.0, 400, 800,
        False, False, None, 0, LIDAR, "fake", "cpu", "grounding_dino", 0.18, 0.15, 2304, 0.2, 2, 64,
        ["tower"], 0.40, 24, 0.18, 3.5, 0.04, 0.16, 0.30, 0.16, 1400, 3400, 0.35, 0.85, 0.95,
        0.30, 0.24, False, False, "", "cpu")
    centres = _chosen_centres(res)
    assert len(centres) == 6
    wrong = sum(1 for (fn, sid), c in centres.items() if np.hypot(*(np.array(c) - TRUE_PX[sid])) > 150)
    assert wrong == 6


def test_v15_without_competitors_refuses_rather_than_guesses(scenario):
    _, crops, _ = _run(scenario, _cfg(lidar_competitors=False, registration=False, bias_calibration=False))
    for (fn, sid), c in _chosen_centres(crops).items():
        assert np.hypot(*(np.array(c) - TRUE_PX[sid])) < 30
    assert crops["tower_assignment_status"].isin(["AMBIGUOUS_ASSIGNMENT", "ASSIGNED_COMPETITOR_AWARE"]).all()


def test_v15_assigns_correct_circuit_and_learns_bias(scenario):
    man, crops, meta = _run(scenario, _cfg())
    centres = _chosen_centres(crops)
    assert len(centres) == 6, crops[["filename", "structure_id", "crop_render_status", "image_registration_status"]]
    for (fn, sid), c in centres.items():
        assert np.hypot(*(np.array(c) - TRUE_PX[sid])) < 30, (fn, sid, c, TRUE_PX[sid])
    assert set(crops["image_registration_status"]) == {"ACCEPTED"}
    assert (crops["competitor_hypothesis_count"] >= 1).all()
    assert set(crops["tower_assignment_confidence"]) == {"HIGH"}
    # Registration recovers the injected EO offset on the first image.
    first = crops[crops["filename"].eq("IMG_000.jpg")].iloc[0]
    assert abs(first["image_registration_dx_px"] - OFFSET[0]) < 20
    assert abs(first["image_registration_dy_px"] - OFFSET[1]) < 20
    # The pass bias is learnt and applied by the third image.
    third = crops[crops["filename"].eq("IMG_002.jpg")].iloc[0]
    assert third["prior_bias_source"] == "PASS"
    assert abs(third["prior_bias_dx_px"] - OFFSET[0]) < 20 and abs(third["prior_bias_dy_px"] - OFFSET[1]) < 20
    # Residuals for georeference validation are populated.
    assert crops["prior_residual_px"].notna().all()
    assert crops["georef_residual_px"].notna().all()
    assert meta["tile_artifacts_rejected_total"] == 3
    assert meta["crop_yield_percent"] == 100.0


def test_csv_competitors_without_lidar_clusters(scenario):
    comp = pd.DataFrame({"competitor_id": list(NEIGHBOURS), "circuit": "DH2",
                         "x": [v[0] for v in NEIGHBOURS.values()], "y": [v[1] for v in NEIGHBOURS.values()]})
    _, crops, _ = _run(scenario, _cfg(lidar_competitors=False), comp)
    centres = _chosen_centres(crops)
    assert len(centres) == 6
    for (fn, sid), c in centres.items():
        assert np.hypot(*(np.array(c) - TRUE_PX[sid])) < 30
    assert crops["competitor_nearest_label"].astype(str).str.startswith("CSV:").all()


def test_render_limit_marks_unprocessed_rows(scenario):
    out, man, crops, fp = scenario
    _, res, _ = co.render_candidate_products_and_crops_v15(
        man, crops, fp, out / "run2", "DH1", "span", "photogrammetry", "none", 2048, 90, 1.0, 400, 800,
        False, False, 1, 0, 0, LIDAR, _cfg(), None)
    assert (res.loc[res["filename"].ne("IMG_000.jpg"), "crop_render_status"] == "NOT_IN_RENDER_SUBSET").all()
    assert (res.loc[res["filename"].eq("IMG_000.jpg"), "crop_render_status"] == "RENDERED").all()


def test_load_competitor_structures_from_fli_step00_csv():
    """FLI_frame_maker step-00 rows (lat/lon) for DH1+DH2 near SR1-60/04; DH1 rows must be dropped."""
    from pathlib import Path
    from pyproj import CRS
    fx = Path(__file__).parent / "fixtures" / "fli_structures_dh1_dh2_sample.csv"
    comp = co.load_competitor_structures(fx, CRS.from_epsg(2262), "DH1")
    assert set(comp["circuit"]) == {"DH2"} and len(comp) == 2
    sr1_60_04 = (1335528.925, 1117316.843)  # DH1 staking XY (EPSG:2262, ftUS)
    d = np.hypot(comp["x"] - sr1_60_04[0], comp["y"] - sr1_60_04[1]).min()
    assert 140 < d < 190  # parallel circuit ~163 ft away: inside competitor radius, outside structure radius
