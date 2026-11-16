import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import build_structure_catalog_v01 as cat
import evaluate_structure_detector_v01 as ev


def _jpeg(color=(120, 130, 90), size=(64, 48), exif=None) -> bytes:
    buf = io.BytesIO()
    im = Image.new("RGB", size, color)
    if exif is not None:
        im.save(buf, "JPEG", quality=90, exif=exif)
    else:
        im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_inject_exif_lossless_keeps_pixels_and_adds_time():
    crop = _jpeg()
    e = Image.Exif()
    e[0x0112] = 6
    e.get_ifd(0x8769)[0x9003] = "2026:04:22 10:11:12"
    raw = _jpeg(exif=e.tobytes())
    payload_src = Path("/tmp/_raw_exif_test.jpg")
    payload_src.write_bytes(raw)
    payload, info = cat.raw_exif_bytes(payload_src)
    assert info["image_exif_time"] == "2026:04:22 10:11:12"
    out = cat.inject_exif_lossless(crop, payload)
    a = np.asarray(Image.open(io.BytesIO(crop)))
    b = np.asarray(Image.open(io.BytesIO(out)))
    assert np.array_equal(a, b), "pixels must be untouched"
    ex = Image.open(io.BytesIO(out)).getexif()
    assert ex.get(0x0112) == 1  # orientation reset: crops are already upright
    assert ex.get_ifd(0x8769).get(0x9003) == "2026:04:22 10:11:12"
    # re-injection replaces, not duplicates
    out2 = cat.inject_exif_lossless(out, payload)
    assert out2.count(b"Exif\x00\x00") == 1


def _make_run(tmp: Path) -> Path:
    run = tmp / "run"
    (run / "structure_crops").mkdir(parents=True)
    rows = []
    # pass P1: 4 consistent observations + 1 outlier (neighbour circuit)
    specs = [
        ("SR1-60/03", 232, "IMG_1.jpg", (100, 500), "HIGH"),
        ("SR1-60/03", 232, "IMG_2.jpg", (120, 520), "HIGH"),
        ("SR1-60/04", 233, "IMG_1.jpg", (90, 480), "HIGH"),
        ("SR1-60/04", 233, "IMG_2.jpg", (110, 530), "MEDIUM"),
        ("SR1-60/05", 234, "IMG_1.jpg", (-1500, 470), "HIGH"),
    ]
    for i, (sid, ref, fn, (dx, dy), conf) in enumerate(specs):
        folder = run / "structure_crops" / f"{ref:03d}_{cat.clean_token(sid)}" / "backward"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{Path(fn).stem}.jpg"
        p.write_bytes(_jpeg(size=(80 + i, 60)))
        prior = (5000.0, 4000.0)
        cx, cy = prior[0] + dx, prior[1] + dy
        rows.append({
            "structure_idx": ref + 1, "reference_idx": ref, "structure_id": sid, "x": 1334800.0 + 730 * (ref - 232),
            "y": 1117340.0, "centerline_z": 640.0, "structure_type": "SUSPENSION", "structure_height": 56.5,
            "chainage": 185700.0 + 730 * (ref - 232), "filename": fn,
            "raw_path": rf"D:\Sev\Obliques\DH1\130_raw\{fn}", "pass_id": "P1", "circuit_run_id": "R1",
            "pass_direction": "BACKWARD_PASS", "view_direction": "BACKWARD", "crop_bucket": "backward",
            "crop_role": "PRODUCT_VIEW", "camera_x": 1336400.0, "camera_y": 1117200.0, "camera_z": 2600.0,
            "omega": 0.0, "phi": 0.0, "kappa": 0.0, "crop_render_status": "RENDERED",
            "tower_detector_status": "OK_VISUAL_DETECTION", "tower_assignment_confidence": conf,
            "tower_assignment_status": "ASSIGNED_COMPETITOR_AWARE", "tower_detector_score": 0.45,
            "tower_detector_final_score": 0.7 - 0.01 * i, "tower_ownership": 0.9,
            "geometry_prior_raw_x_px": prior[0], "geometry_prior_raw_y_px": prior[1],
            "tower_box_xmin": cx - 300, "tower_box_xmax": cx + 300, "tower_box_ymin": cy - 400, "tower_box_ymax": cy + 400,
            "source_width": 14204, "source_height": 10652, "tower_edge_touch": 0, "lidar_ground_z": 600.0,
            "camera_model_quality": "HIGH", "image_registration_status": "ACCEPTED",
            # Windows path from another machine: builder must fall back to the run folder
            "crop_path": rf"D:\elsewhere\run\structure_crops\{ref:03d}_{cat.clean_token(sid)}\backward\{Path(fn).stem}.jpg",
        })
    rows.append({**rows[0], "filename": "IMG_9.jpg", "crop_render_status": "NO_PLAUSIBLE_DETECTIONS_IN_ROI",
                 "tower_detector_status": "NO_PLAUSIBLE_DETECTIONS_IN_ROI", "tower_assignment_confidence": None,
                 "crop_path": None})
    pd.DataFrame(rows).to_csv(run / "structure_crop_candidates.csv", index=False)
    topo = pd.DataFrame({"structure_idx": [233, 234, 235, 236], "reference_idx": [232, 233, 234, 235],
                         "structure_id": ["SR1-60/03", "SR1-60/04", "SR1-60/05", "SR1-60/06"],
                         "x": [1334800.0, 1335530.0, 1336260.0, 1336990.0], "y": [1117340.0] * 4,
                         "station": [1.0, 2.0, 3.0, 4.0], "is_reference_structure": [True] * 4,
                         "structure_type": ["SUSPENSION"] * 4})
    topo.to_csv(run / "reference_topology.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"working_epsg": 2262, "circuit": "DH1", "poc_version": "0.15"}))
    return run


def _args(run, **kw):
    a = dict(run_output=run, circuit="DH1", epsg=None, catalog_root=None, min_confidence="MEDIUM", raw_dir=None,
             fli_all_images=False, bias_flag_px=400.0, outlier_k=3.0, outlier_min_px=300.0, outlier_min_n=4)
    a.update(kw)
    return argparse.Namespace(**a)


def test_catalog_end_to_end(tmp_path):
    run = _make_run(tmp_path)
    s = cat.build_catalog(_args(run))
    out = run / "structure_catalog"
    assert s["observations"] == 5 and s["missing_crop_files"] == 0
    obs = pd.read_csv(out / "structure_catalog_observations.csv")
    o5 = obs[obs.structure_id.eq("SR1-60/05")].iloc[0]
    assert o5.residual_consistency == "OUTLIER" and o5.quality_grade == "C"
    assert set(obs.loc[obs.structure_id.ne("SR1-60/05"), "residual_consistency"]) == {"OK"}
    assert obs.groupby("structure_id")["is_best_view"].sum().eq(1).all()
    assert obs["structure_lat"].between(43.0, 43.1).all() and obs["structure_lon"].between(-78.0, -77.8).all()
    for f in obs["catalog_crop_file"]:
        assert (out / f).exists()
        assert re.match(r"crops/\d{3}_[^/]+/DH1__STR_\d{3}_.+__BACKWARD__OBS_\d{3}__SRC_IMG_\d\.jpg$", f)
    st = pd.read_csv(out / "structure_catalog_structures.csv")
    assert dict(zip(st.structure_id, st.catalog_status)) == {
        "SR1-60/03": "CATALOGUED", "SR1-60/04": "CATALOGUED", "SR1-60/05": "CATALOGUED", "SR1-60/06": "NOT_ATTEMPTED"}
    # FLI contract: exact column order of FLI_frame_maker CSV_FIELDS, one best view per structure, images present
    with open(out / "fli" / "structure_photo_match_qa.csv", encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh))
    assert header == cat.FLI_CSV_FIELDS
    fli = pd.read_csv(out / "fli" / "structure_photo_match_qa.csv", encoding="utf-8-sig")
    assert sorted(fli.structure_id) == ["SR1-60/03", "SR1-60/04", "SR1-60/05"]
    assert (fli.anchor == "yes").all()
    assert all((out / "fli" / "images" / f).exists() for f in fli.image_file)
    assert "residual_outlier_review" in fli.loc[fli.structure_id.eq("SR1-60/05"), "warning"].iloc[0]
    assert fli["source_epsg"].eq("EPSG:2262").all()
    tp = pd.read_csv(out / "georef" / "tie_points.csv")
    assert {"detected_base_u_px", "detected_base_v_px", "structure_x", "structure_y", "structure_ground_z"} <= set(tp.columns)
    gj = json.loads((out / "structures.geojson").read_text())
    assert len(gj["features"]) == 4
    # rerun overwrites its own catalogue
    cat.build_catalog(_args(run))


def test_catalog_confidence_filter_and_overwrite_guard(tmp_path):
    run = _make_run(tmp_path)
    s = cat.build_catalog(_args(run, min_confidence="HIGH"))
    assert s["observations"] == 4 and s["excluded_by_confidence"] == 1
    other = tmp_path / "not_a_catalog"
    other.mkdir()
    (other / "precious.txt").write_text("x")
    with pytest.raises(SystemExit):
        cat.build_catalog(_args(run, catalog_root=other))
    assert (other / "precious.txt").exists()


def test_evaluate_detector_with_adjudication(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "eval"
    s = ev.evaluate(run, None, out, 50)
    assert s["attempted_pairs"] == 6 and s["pairs_with_crop"] == 5
    t = pd.read_csv(out / "adjudication_template.csv")
    t["label"] = np.where(t.structure_id.eq("SR1-60/05"), "WRONG_STRUCTURE", np.where(t.has_crop, "CORRECT", "MISSED"))
    t.to_csv(tmp_path / "adj.csv", index=False)
    s = ev.evaluate(run, tmp_path / "adj.csv", out, 50)
    a = s["adjudication"]["all"]
    assert a["precision"] == 0.8 and a["wrong_structure_rate"] == 0.2
    assert a["missed_visible_towers"] == 1 and a["recall_estimate"] == pytest.approx(4 / 5)
    assert (out / "adjudication.html").exists()
