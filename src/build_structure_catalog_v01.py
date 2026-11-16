#!/usr/bin/env python
r"""
Build a georeferenced structure catalogue from a continuous-oblique run.

The catalogue is the hand-off between continuous-oblique structure detection and downstream
consumers:

* FLI_frame_maker (https://github.com/sevcieslik/FLI_frame_maker) - an assignment CSV in the
  `structure_photo_match_qa.csv` schema plus an image folder, so `03_frame_generator_v4.py --batch`
  can frame catalogue crops directly;
* georeference validation - every accepted detection is an image/world tie point
  (pixel u,v of the detected tower base <-> staking X,Y + LiDAR ground Z). Residual tables per
  observation, image and pass show EO/OPK bias and can be used to validate another
  georeferencing process;
* GIS review - GeoJSON of structures and camera->structure observation rays.

The builder never changes detection results. It only selects, ranks, tags and copies.

Inputs (from continuous_oblique_v015.py; v0.14 runs are accepted with --min-confidence ANY)
------
<run>/structure_crop_candidates.csv   required
<run>/summary.json                    working EPSG / circuit
<run>/reference_topology.csv          optional, structure list for coverage
<run>/product_manifest.csv            optional, capture date per frame

Outputs (default <run>/structure_catalog)
-------
crops/<ref>_<structure>/<CIRCUIT>__STR_<ref>_<structure>__<BUCKET>__OBS_<n>__SRC_<stem>.jpg
structure_catalog_observations.csv    one row per catalogued observation, fully tagged
structure_catalog_structures.csv      one row per reference structure incl. coverage and best views
structures.geojson / observations.geojson
georef/tie_points.csv                 image <-> world tie points
georef/residuals_by_image.csv / residuals_by_pass.csv
fli/structure_photo_match_qa.csv      best view per structure, FLI schema
fli/structure_photo_match_all_views.csv
fli/images/                           crops with raw EXIF (time/camera/GPS) inserted losslessly
catalog_summary.json, README.txt

Example
-------
python build_structure_catalog_v01.py "D:\Sev\Obliques\DH1\_continuous_oblique_DH1_full_v015" --circuit DH1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG_VERSION = "0.1.0"
CONFIDENCE_ORDER = {"HIGH": 3, "MEDIUM": 2, "UNGRADED": 1}

# Column order of FLI_frame_maker current/01-02_*_manual_structure_match_v1.py CSV_FIELDS.
FLI_CSV_FIELDS = [
    "structure_sequence", "match_sequence", "manual_row", "manual_structure_id", "manual_image_name",
    "structure_id", "effective_structure_id", "auto_structure_id", "manual_structure_id_override",
    "display_structure_id", "tower_name", "structure_number", "structure_type", "structure_description",
    "structure_model", "station", "structure_height", "structure_elevation", "transverse_axis_azimuth",
    "image_id", "image_file", "image_path", "manual_image_id", "image_exif_time", "focal_length_mm",
    "photo_lat", "photo_lon", "photo_altitude", "structure_lat", "structure_lon", "distance_m",
    "bearing_photo_to_structure_deg", "easting", "northing", "source_crs_label", "source_epsg",
    "target_crs_label", "target_epsg", "crs_label", "coordinate_units", "circuit", "match_method",
    "match_status", "effective_match_status", "anchor", "review_note", "warning",
]


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def clean_token(value, fallback="NA") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return fallback
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-._") or fallback


def fnum(v, default=float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def path_parts(p: str) -> list[str]:
    return [x for x in re.split(r"[\\/]+", str(p)) if x]


def resolve_crop_path(crop_path, run: Path) -> Path | None:
    """Crop paths in CSVs are absolute on the machine that ran the pipeline. Fall back to the run folder."""
    s = str(crop_path or "")
    if not s or s.lower() == "nan":
        return None
    p = Path(s)
    if p.exists():
        return p
    parts = path_parts(s)
    if "structure_crops" in parts:
        rel = parts[parts.index("structure_crops"):]
        q = run.joinpath(*rel)
        if q.exists():
            return q
    if len(parts) >= 3:
        q = run / "structure_crops" / parts[-3] / parts[-2] / parts[-1]
        if q.exists():
            return q
    return None


def resolve_raw_path(raw_path, raw_dirs: list[Path]) -> Path | None:
    s = str(raw_path or "")
    if s and s.lower() != "nan" and Path(s).exists():
        return Path(s)
    name = path_parts(s)[-1] if s else ""
    for d in raw_dirs:
        q = d / name
        if name and q.exists():
            return q
    return None


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    if not all(math.isfinite(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def crs_info(epsg: int | None):
    """Return (transformer to WGS84 or None, crs label, unit name, metres per unit)."""
    if not epsg:
        return None, "", "", float("nan")
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_epsg(int(epsg))
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        ax = crs.axis_info[0]
        return tr, crs.name, ax.unit_name or "", float(ax.unit_conversion_factor or float("nan"))
    except Exception:
        return None, f"EPSG:{epsg}", "", float("nan")


# -----------------------------------------------------------------------------
# EXIF (lossless insertion so catalogue crops keep capture time/camera for FLI)
# -----------------------------------------------------------------------------

def raw_exif_bytes(raw: Path | None) -> tuple[bytes | None, dict]:
    """Read EXIF from the raw frame without decoding pixels. Orientation forced to 1, thumbnail dropped."""
    info = {"image_exif_time": "", "focal_length_mm": "", "make": "", "model": ""}
    if raw is None:
        return None, info
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(raw) as im:
            exif = im.getexif()
            if not exif:
                return None, info
            sub = exif.get_ifd(0x8769)
            dto = sub.get(0x9003) or exif.get(0x0132) or ""
            fl = sub.get(0x920A)
            info["image_exif_time"] = str(dto)
            info["focal_length_mm"] = f"{float(fl):.2f}" if fl is not None else ""
            info["make"] = str(exif.get(0x010F, "") or "")
            info["model"] = str(exif.get(0x0110, "") or "")
            exif[0x0112] = 1  # crops were written after exif_transpose
            for tag in (0x0201, 0x0202):  # JPEGInterchangeFormat(Length) of IFD0 thumbnails
                exif.pop(tag, None)
            return exif.tobytes(), info
    except Exception:
        return None, info


def inject_exif_lossless(jpeg: bytes, exif_payload: bytes) -> bytes:
    """Insert an APP1/Exif segment after SOI (and APP0) without re-encoding pixels.

    exif_payload must start with b"Exif\\x00\\x00" (Pillow's Exif.tobytes()). Existing APP1 Exif
    segments are removed.
    """
    if not jpeg.startswith(b"\xff\xd8") or not exif_payload or len(exif_payload) + 2 > 0xFFFF:
        return jpeg
    i = 2
    head = [b"\xff\xd8"]
    rest_start = 2
    while i + 4 <= len(jpeg) and jpeg[i] == 0xFF:
        marker = jpeg[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(jpeg[i + 2:i + 4], "big")
        seg = jpeg[i:i + 2 + seg_len]
        if marker == 0xE0:              # keep JFIF APP0 first
            head.append(seg)
        elif marker == 0xE1 and seg[4:10] == b"Exif\x00\x00":
            pass                        # drop old Exif
        else:
            break
        i += 2 + seg_len
        rest_start = i
    app1 = b"\xff\xe1" + (len(exif_payload) + 2).to_bytes(2, "big") + exif_payload
    return b"".join(head) + app1 + jpeg[rest_start:]


# -----------------------------------------------------------------------------
# catalogue construction
# -----------------------------------------------------------------------------

def observation_confidence(r: pd.Series) -> str:
    c = str(r.get("tower_assignment_confidence") or "").upper()
    if c in ("HIGH", "MEDIUM"):
        return c
    return "UNGRADED"  # v0.14 or older runs: no competitor-aware confidence available


def flag_residual_outliers(obs: pd.DataFrame, k: float = 3.0, min_px: float = 300.0, min_n: int = 4) -> pd.DataFrame:
    """Per-pass robust consistency check of detection-minus-prior residuals.

    Within one acquisition pass the EO/OPK projection error is locally similar, so an observation whose
    residual differs from the pass median by more than max(k * MAD, min_px) in either image axis is
    suspicious - typically a tower of a neighbouring circuit or an adjacent structure. Evaluated
    component-wise because along-track error varies more with range than cross-track error.
    """
    obs = obs.copy()
    obs["residual_consistency"] = "NOT_EVALUATED"
    obs["residual_robust_z"] = np.nan
    key = obs["pass_id"].astype(str) if "pass_id" in obs.columns else pd.Series("ALL", index=obs.index)
    for _, idx in obs.groupby(key).groups.items():
        g = obs.loc[idx]
        dx = pd.to_numeric(g["prior_residual_dx_px"], errors="coerce")
        dy = pd.to_numeric(g["prior_residual_dy_px"], errors="coerce")
        ok = dx.notna() & dy.notna()
        if int(ok.sum()) < int(min_n):
            continue
        zs = []
        for comp in (dx, dy):
            med = float(comp[ok].median())
            mad = float((comp[ok] - med).abs().median()) * 1.4826
            thr = max(float(k) * mad, float(min_px))
            zs.append((comp - med).abs() / max(thr / float(k), 1e-9))
        z = pd.concat(zs, axis=1).max(axis=1)
        obs.loc[idx, "residual_robust_z"] = z
        obs.loc[idx, "residual_consistency"] = np.where(ok, np.where(z > float(k), "OUTLIER", "OK"), "NOT_EVALUATED")
    return obs


def quality_grade(r: pd.Series) -> str:
    if r.get("residual_consistency") == "OUTLIER":
        return "C"
    conf = r["catalog_confidence"]
    cam = str(r.get("camera_model_quality") or "").upper()
    reg = str(r.get("image_registration_status") or "")
    if conf == "HIGH" and cam in ("HIGH", "MEDIUM", "") and reg in ("ACCEPTED", "INSUFFICIENT_MATCHES", "TOO_FEW_HYPOTHESES", ""):
        return "A"
    if conf in ("HIGH", "MEDIUM"):
        return "B"
    return "C"


def view_rank_score(r: pd.Series) -> float:
    """Score for best-view ranking. Metadata only; all views remain in the catalogue."""
    final = fnum(r.get("tower_detector_final_score"), 0.0)
    h = fnum(r.get("tower_box_ymax")) - fnum(r.get("tower_box_ymin"))
    sh = fnum(r.get("source_height"))
    size = min(1.0, h / (0.12 * sh)) if math.isfinite(h) and math.isfinite(sh) and sh > 0 else 0.5
    edge = 0.0 if int(fnum(r.get("tower_edge_touch"), 0)) else 1.0
    x0, y0, x1, y1 = (fnum(r.get(k)) for k in ("tower_box_xmin", "tower_box_ymin", "tower_box_xmax", "tower_box_ymax"))
    sw = fnum(r.get("source_width"))
    margin = 1.0
    if all(math.isfinite(v) for v in (x0, y0, x1, y1, sw, sh)):
        m = min(x0, y0, sw - x1, sh - y1)
        margin = 1.0 if m > 0.03 * min(sw, sh) else 0.3
    conf = CONFIDENCE_ORDER.get(r["catalog_confidence"], 0) / 3.0
    return 0.40 * final + 0.25 * size + 0.15 * edge + 0.10 * margin + 0.10 * conf


def build_catalog(args) -> dict:
    run = Path(args.run_output).resolve()
    crops_csv = run / "structure_crop_candidates.csv"
    if not crops_csv.exists():
        raise SystemExit(f"Missing {crops_csv}")
    summary = {}
    if (run / "summary.json").exists():
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    circuit = clean_token(args.circuit or summary.get("circuit"), "CIRCUIT")
    epsg = args.epsg or summary.get("working_epsg")
    tr, crs_label, unit_name, m_per_unit = crs_info(epsg)
    out = Path(args.catalog_root).resolve() if args.catalog_root else run / "structure_catalog"
    if out.exists():
        marker = out / "catalog_summary.json"
        if not marker.exists() and any(out.iterdir()):
            raise SystemExit(f"Refusing to overwrite non-catalogue folder: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    crops = read_csv(crops_csv)
    topo = read_csv(run / "reference_topology.csv")
    manifest = read_csv(run / "product_manifest.csv")
    raw_dirs = [Path(p) for p in (args.raw_dir or [])]

    rendered = crops[crops.get("crop_render_status", pd.Series("", index=crops.index)).astype(str).eq("RENDERED")].copy()
    rendered["catalog_confidence"] = rendered.apply(observation_confidence, axis=1)
    min_conf = CONFIDENCE_ORDER.get(args.min_confidence, 0) if args.min_confidence != "ANY" else 0
    keep = rendered["catalog_confidence"].map(lambda c: CONFIDENCE_ORDER.get(c, 0) >= min_conf)
    excluded_conf = int((~keep).sum())
    sel = rendered[keep].copy()

    # topology lookups (station, elevation, azimuth) when not already on the crop rows
    if not topo.empty and "structure_idx" in topo.columns:
        extra = [c for c in ["station", "transverse_axis_azimuth", "structure_name", "tin_z"] if c in topo.columns]
        if extra:
            sel = sel.drop(columns=[c for c in extra if c in sel.columns], errors="ignore").merge(
                topo[["structure_idx"] + extra].drop_duplicates("structure_idx"), on="structure_idx", how="left")
    date_by_file = {}
    if not manifest.empty and {"filename", "capture_date_code"} <= set(manifest.columns):
        date_by_file = manifest.drop_duplicates("filename").set_index("filename")["capture_date_code"].astype(str).to_dict()

    rows = []
    missing_files = 0
    obs_counter: Counter = Counter()
    exif_cache: dict = {}
    for _, r in sel.sort_values(["reference_idx", "filename"], na_position="last").iterrows():
        src = resolve_crop_path(r.get("crop_path"), run)
        if src is None:
            missing_files += 1
            continue
        sid = str(r.get("structure_id"))
        sid_tok = clean_token(sid, "STRUCTURE")
        try:
            ref = int(float(r.get("reference_idx")))
        except Exception:
            ref = int(float(r.get("structure_idx"))) if pd.notna(r.get("structure_idx")) else 0
        ref_txt = f"{ref:03d}"
        bucket = clean_token(r.get("crop_bucket"), "other_views").lower()
        key = (ref_txt, sid_tok, bucket)
        obs_counter[key] += 1
        stem = clean_token(Path(path_parts(str(r.get("filename") or src.name))[-1]).stem, "SOURCE")
        name = f"{circuit}__STR_{ref_txt}_{sid_tok}__{bucket.upper()}__OBS_{obs_counter[key]:03d}__SRC_{stem}.jpg"
        rel = Path("crops") / f"{ref_txt}_{sid_tok}" / name
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dest.write_bytes(data)

        sx, sy = fnum(r.get("x")), fnum(r.get("y"))
        cx, cy = fnum(r.get("camera_x")), fnum(r.get("camera_y"))
        lon = lat = clon = clat = float("nan")
        if tr is not None and math.isfinite(sx) and math.isfinite(sy):
            lon, lat = tr.transform(sx, sy)
        if tr is not None and math.isfinite(cx) and math.isfinite(cy):
            clon, clat = tr.transform(cx, cy)
        dist_units = math.hypot(cx - sx, cy - sy) if all(math.isfinite(v) for v in (cx, cy, sx, sy)) else float("nan")
        ground_z = fnum(r.get("lidar_ground_z"), fnum(r.get("centerline_z")))
        bx0, by0, bx1, by1 = (fnum(r.get(k)) for k in ("tower_box_xmin", "tower_box_ymin", "tower_box_xmax", "tower_box_ymax"))
        det_base_u = fnum(r.get("detected_base_x_px"), (bx0 + bx1) / 2.0)
        det_base_v = fnum(r.get("detected_base_y_px"), by1)
        pr_u = fnum(r.get("geometry_prior_raw_x_px"), fnum(r.get("geometry_prior_x_px")))
        pr_v = fnum(r.get("geometry_prior_raw_y_px"), fnum(r.get("geometry_prior_y_px")))
        prior_dx = fnum(r.get("prior_residual_dx_px"), (bx0 + bx1) / 2.0 - pr_u)
        prior_dy = fnum(r.get("prior_residual_dy_px"), (by0 + by1) / 2.0 - pr_v)
        rec = r.to_dict()
        rec.update({
            "catalog_version": CATALOG_VERSION,
            "catalog_obs_id": f"{circuit}-{ref_txt}-{sid_tok}-{bucket}-{obs_counter[key]:03d}",
            "circuit": circuit,
            "catalog_crop_file": str(rel).replace("\\", "/"),
            "catalog_crop_sha1": hashlib.sha1(data).hexdigest(),
            "source_image": path_parts(str(r.get("filename") or ""))[-1] if r.get("filename") else "",
            "capture_date_code": date_by_file.get(str(r.get("filename")), ""),
            "epsg": int(epsg) if epsg else None,
            "crs_label": crs_label,
            "coordinate_units": unit_name,
            "structure_x": sx, "structure_y": sy, "structure_ground_z": ground_z,
            "structure_lat": lat, "structure_lon": lon,
            "camera_lat": clat, "camera_lon": clon,
            "camera_distance_m": dist_units * m_per_unit if math.isfinite(m_per_unit) else float("nan"),
            "bearing_camera_to_structure_deg": bearing_deg(clat, clon, lat, lon),
            "detected_base_u_px": det_base_u, "detected_base_v_px": det_base_v,
            "prior_residual_dx_px": prior_dx, "prior_residual_dy_px": prior_dy,
            "prior_residual_px": math.hypot(prior_dx, prior_dy) if math.isfinite(prior_dx) and math.isfinite(prior_dy) else float("nan"),
        })
        rows.append(rec)

    obs = pd.DataFrame(rows)
    if not obs.empty:
        obs = flag_residual_outliers(obs, args.outlier_k, args.outlier_min_px, args.outlier_min_n)
        obs["quality_grade"] = obs.apply(quality_grade, axis=1)
        obs["view_rank_score"] = obs.apply(view_rank_score, axis=1)
        obs.loc[obs["residual_consistency"].eq("OUTLIER"), "view_rank_score"] *= 0.5
        obs["best_view_rank_in_bucket"] = obs.groupby(["structure_id", "crop_bucket"])["view_rank_score"].rank(ascending=False, method="first").astype(int)
        obs["best_view_rank_overall"] = obs.groupby("structure_id")["view_rank_score"].rank(ascending=False, method="first").astype(int)
        obs["is_best_view"] = obs["best_view_rank_overall"].eq(1)

    lead = ["catalog_obs_id", "circuit", "structure_id", "reference_idx", "structure_idx", "structure_type",
            "structure_description", "structure_height", "chainage", "structure_x", "structure_y", "structure_ground_z",
            "structure_lat", "structure_lon", "epsg", "source_image", "capture_date_code", "pass_id", "pass_direction",
            "view_direction", "circuit_run_id", "crop_bucket", "crop_role", "catalog_confidence", "quality_grade",
            "is_best_view", "residual_consistency", "residual_robust_z", "best_view_rank_overall", "best_view_rank_in_bucket", "view_rank_score",
            "catalog_crop_file", "catalog_crop_sha1", "tower_box_xmin", "tower_box_ymin", "tower_box_xmax",
            "tower_box_ymax", "crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax", "source_width", "source_height",
            "detected_base_u_px", "detected_base_v_px", "tower_detector_score", "tower_detector_final_score",
            "tower_ownership", "tower_assignment_status", "image_registration_status", "camera_x", "camera_y",
            "camera_z", "omega", "phi", "kappa", "camera_lat", "camera_lon", "camera_distance_m",
            "bearing_camera_to_structure_deg", "camera_model_quality", "prior_residual_dx_px",
            "prior_residual_dy_px", "prior_residual_px", "georef_residual_dx_px", "georef_residual_dy_px",
            "georef_residual_px"]
    if not obs.empty:
        cols = [c for c in lead if c in obs.columns] + [c for c in obs.columns if c not in lead]
        obs = obs[cols]
    obs.to_csv(out / "structure_catalog_observations.csv", index=False)

    # ---- per-structure table ---------------------------------------------------------------------
    struct_rows = []
    if not topo.empty:
        refs = topo[topo["is_reference_structure"].astype(str).str.lower().isin({"true", "1"})] if "is_reference_structure" in topo.columns else topo
    else:
        refs = crops.drop_duplicates("structure_id")
    attempts = crops[~crops.get("crop_render_status", pd.Series("", index=crops.index)).astype(str).isin(
        ["NOT_IN_RENDER_SUBSET", "NOT_RENDERED_OTHER_VIEW", "NOT_RUN"])]
    for _, s in refs.iterrows():
        sid = str(s.get("structure_id"))
        o = obs[obs["structure_id"].astype(str).eq(sid)] if not obs.empty else pd.DataFrame()
        a = attempts[attempts["structure_id"].astype(str).eq(sid)]
        sx, sy = fnum(s.get("x")), fnum(s.get("y"))
        lon = lat = float("nan")
        if tr is not None and math.isfinite(sx) and math.isfinite(sy):
            lon, lat = tr.transform(sx, sy)
        best = o[o["is_best_view"]].iloc[0] if not o.empty else None
        rec = {
            "circuit": circuit, "structure_id": sid, "reference_idx": s.get("reference_idx"),
            "structure_idx": s.get("structure_idx"), "structure_type": s.get("structure_type"),
            "structure_description": s.get("structure_description"), "structure_height": s.get("structure_height"),
            "chainage": s.get("chainage"), "station": s.get("station"), "x": sx, "y": sy, "lat": lat, "lon": lon,
            "epsg": int(epsg) if epsg else None,
            "observations": int(len(o)),
            "observations_grade_A": int(o["quality_grade"].eq("A").sum()) if not o.empty else 0,
            "observations_forward": int(o["crop_bucket"].eq("forward").sum()) if not o.empty else 0,
            "observations_backward": int(o["crop_bucket"].eq("backward").sum()) if not o.empty else 0,
            "observations_other_views": int(o["crop_bucket"].eq("other_views").sum()) if not o.empty else 0,
            "detection_attempts": int(len(a)),
            "attempt_status_counts": json.dumps(dict(Counter(a.get("tower_detector_status", pd.Series(dtype=object)).fillna("NONE").astype(str))), sort_keys=True),
            "catalog_status": "CATALOGUED" if len(o) else ("NO_ACCEPTED_DETECTION" if len(a) else "NOT_ATTEMPTED"),
            "best_view_obs_id": best["catalog_obs_id"] if best is not None else "",
            "best_view_crop_file": best["catalog_crop_file"] if best is not None else "",
            "best_view_source_image": best["source_image"] if best is not None else "",
        }
        for b in ("forward", "backward"):
            ob = o[o["crop_bucket"].eq(b) & o["best_view_rank_in_bucket"].eq(1)] if not o.empty else pd.DataFrame()
            rec[f"best_{b}_crop_file"] = ob.iloc[0]["catalog_crop_file"] if not ob.empty else ""
        struct_rows.append(rec)
    structs = pd.DataFrame(struct_rows)
    structs.to_csv(out / "structure_catalog_structures.csv", index=False)

    # ---- GeoJSON -----------------------------------------------------------------------------------
    def jsonable(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if not math.isfinite(float(v)) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    feats = []
    for _, s in structs.iterrows():
        if math.isfinite(fnum(s["lat"])):
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                          "properties": {k: jsonable(s[k]) for k in ["structure_id", "reference_idx", "structure_type", "observations",
                                                                    "observations_grade_A", "catalog_status", "best_view_crop_file"]}})
    (out / "structures.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    feats = []
    if not obs.empty:
        for _, o in obs.iterrows():
            if all(math.isfinite(fnum(o[k])) for k in ["camera_lon", "camera_lat", "structure_lon", "structure_lat"]):
                feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [
                    [o["camera_lon"], o["camera_lat"]], [o["structure_lon"], o["structure_lat"]]]},
                    "properties": {k: jsonable(o.get(k)) for k in ["catalog_obs_id", "structure_id", "source_image", "crop_bucket",
                                                                  "catalog_confidence", "quality_grade", "is_best_view", "catalog_crop_file"]}})
    (out / "observations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")

    # ---- georeference validation -----------------------------------------------------------------
    georef = out / "georef"
    georef.mkdir()
    if not obs.empty:
        tp_cols = ["catalog_obs_id", "source_image", "pass_id", "circuit_run_id", "source_width", "source_height",
                   "detected_base_u_px", "detected_base_v_px", "structure_id", "structure_x", "structure_y",
                   "structure_ground_z", "epsg", "catalog_confidence", "quality_grade",
                   "camera_x", "camera_y", "camera_z", "omega", "phi", "kappa", "camera_model_quality",
                   "staking_base_x_px", "staking_base_y_px", "georef_residual_dx_px", "georef_residual_dy_px",
                   "georef_residual_px", "prior_residual_dx_px", "prior_residual_dy_px", "prior_residual_px"]
        obs[[c for c in tp_cols if c in obs.columns]].to_csv(georef / "tie_points.csv", index=False)

        def resid_table(group_col):
            res = []
            for k, g in obs.groupby(group_col, dropna=False):
                dx = pd.to_numeric(g["prior_residual_dx_px"], errors="coerce").dropna()
                dy = pd.to_numeric(g["prior_residual_dy_px"], errors="coerce").dropna()
                rr = np.hypot(dx, dy) if len(dx) == len(dy) else pd.Series(dtype=float)
                gdx = pd.to_numeric(g.get("georef_residual_dx_px", pd.Series(dtype=float)), errors="coerce").dropna()
                gdy = pd.to_numeric(g.get("georef_residual_dy_px", pd.Series(dtype=float)), errors="coerce").dropna()
                med = (float(dx.median()) if len(dx) else float("nan"), float(dy.median()) if len(dy) else float("nan"))
                mad = float(np.median(np.hypot(dx - med[0], dy - med[1]))) * 1.4826 if len(dx) and len(dx) == len(dy) else float("nan")
                flag = ""
                if len(dx) >= 3 and math.hypot(*med) > args.bias_flag_px and (not math.isfinite(mad) or math.hypot(*med) > 2 * mad):
                    flag = "EO_BIAS_SUSPECT"
                res.append({group_col: k, "n": int(len(g)), "prior_dx_median_px": med[0], "prior_dy_median_px": med[1],
                            "prior_residual_mad_px": mad, "prior_residual_p90_px": float(rr.quantile(0.9)) if len(rr) else float("nan"),
                            "georef_dx_median_px": float(gdx.median()) if len(gdx) else float("nan"),
                            "georef_dy_median_px": float(gdy.median()) if len(gdy) else float("nan"),
                            "flag": flag})
            return pd.DataFrame(res)
        resid_table("source_image").to_csv(georef / "residuals_by_image.csv", index=False)
        resid_table("pass_id").to_csv(georef / "residuals_by_pass.csv", index=False)

    # ---- FLI export -----------------------------------------------------------------------------------
    fli = out / "fli"
    fli_img = fli / "images"
    fli_img.mkdir(parents=True)
    fli_rows = []
    exif_found = 0
    if not obs.empty:
        units_m = m_per_unit if math.isfinite(m_per_unit) else float("nan")
        for seq, (_, o) in enumerate(obs.sort_values(["reference_idx", "best_view_rank_overall"]).iterrows(), start=1):
            src = out / o["catalog_crop_file"]
            fli_name = f"{circuit}_{clean_token(o['structure_id'])}_{str(o['crop_bucket']).upper()}_{clean_token(o['source_image'] and Path(o['source_image']).stem)}.jpg"
            raw = resolve_raw_path(o.get("raw_path"), raw_dirs)
            key = str(raw) if raw else None
            if key not in exif_cache:
                exif_cache[key] = raw_exif_bytes(raw)
            payload, info = exif_cache[key]
            data = src.read_bytes()
            if payload:
                data = inject_exif_lossless(data, payload)
                exif_found += 1
            (fli_img / fli_name).write_bytes(data)
            warn = []
            if o["catalog_confidence"] != "HIGH":
                warn.append(f"confidence_{str(o['catalog_confidence']).lower()}")
            if not payload:
                warn.append("no_raw_exif")
            if o.get("residual_consistency") == "OUTLIER":
                warn.append("residual_outlier_review")
            cam_alt = fnum(o.get("camera_z")) * units_m if math.isfinite(units_m) else ""
            fli_rows.append({
                "structure_sequence": int(fnum(o.get("reference_idx"), 0)),
                "match_sequence": seq, "manual_row": "", "manual_structure_id": "", "manual_image_name": "",
                "structure_id": o["structure_id"], "effective_structure_id": o["structure_id"],
                "auto_structure_id": o["structure_id"], "manual_structure_id_override": "",
                "display_structure_id": o["structure_id"], "tower_name": "", "structure_number": o["structure_id"],
                "structure_type": o.get("structure_type", ""), "structure_description": o.get("structure_description", ""),
                "structure_model": "", "station": o.get("station", ""), "structure_height": o.get("structure_height", ""),
                "structure_elevation": o.get("centerline_z", ""), "transverse_axis_azimuth": o.get("transverse_axis_azimuth", ""),
                "image_id": fli_name, "image_file": fli_name, "image_path": str((fli_img / fli_name).resolve()),
                "manual_image_id": fli_name, "image_exif_time": info.get("image_exif_time", ""),
                "focal_length_mm": info.get("focal_length_mm", ""),
                "photo_lat": o.get("camera_lat"), "photo_lon": o.get("camera_lon"), "photo_altitude": cam_alt,
                "structure_lat": o.get("structure_lat"), "structure_lon": o.get("structure_lon"),
                "distance_m": round(fnum(o.get("camera_distance_m")), 2) if math.isfinite(fnum(o.get("camera_distance_m"))) else "",
                "bearing_photo_to_structure_deg": round(fnum(o.get("bearing_camera_to_structure_deg")), 2) if math.isfinite(fnum(o.get("bearing_camera_to_structure_deg"))) else "",
                "easting": o.get("structure_x"), "northing": o.get("structure_y"),
                "source_crs_label": crs_label, "source_epsg": f"EPSG:{epsg}" if epsg else "",
                "target_crs_label": crs_label, "target_epsg": f"EPSG:{epsg}" if epsg else "",
                "crs_label": crs_label, "coordinate_units": unit_name, "circuit": circuit,
                "match_method": "continuous_oblique_auto_detection",
                "match_status": f"auto_detected_{str(o['catalog_confidence']).lower()}",
                "effective_match_status": f"auto_detected_{str(o['catalog_confidence']).lower()}",
                "anchor": "yes" if bool(o["is_best_view"]) else "no",
                "review_note": f"obs={o['catalog_obs_id']} grade={o['quality_grade']} bucket={o['crop_bucket']} src={o['source_image']}",
                "warning": ";".join(warn),
            })
    all_views = pd.DataFrame(fli_rows, columns=FLI_CSV_FIELDS)
    all_views.to_csv(fli / "structure_photo_match_all_views.csv", index=False, encoding="utf-8-sig")
    best = all_views[all_views["anchor"].eq("yes")] if not all_views.empty else all_views
    best.to_csv(fli / "structure_photo_match_qa.csv", index=False, encoding="utf-8-sig")
    if not args.fli_all_images and not all_views.empty:
        keep_names = set(best["image_file"])
        for p in fli_img.iterdir():
            if p.name not in keep_names:
                p.unlink()

    # ---- summary ----------------------------------------------------------------------------------
    summ = {
        "catalog_version": CATALOG_VERSION,
        "circuit": circuit,
        "source_run": str(run),
        "source_poc_version": summary.get("poc_version"),
        "epsg": int(epsg) if epsg else None,
        "crs_label": crs_label,
        "min_confidence": args.min_confidence,
        "rendered_crop_rows": int(len(rendered)),
        "excluded_by_confidence": excluded_conf,
        "missing_crop_files": int(missing_files),
        "observations": int(len(obs)),
        "observation_confidence_counts": dict(Counter(obs["catalog_confidence"])) if not obs.empty else {},
        "quality_grade_counts": dict(Counter(obs["quality_grade"])) if not obs.empty else {},
        "residual_consistency_counts": dict(Counter(obs["residual_consistency"])) if not obs.empty else {},
        "reference_structures": int(len(structs)),
        "structures_catalogued": int(structs["catalog_status"].eq("CATALOGUED").sum()) if not structs.empty else 0,
        "structure_status_counts": dict(Counter(structs["catalog_status"])) if not structs.empty else {},
        "fli_best_view_rows": int(len(best)),
        "fli_rows_with_raw_exif": int(exif_found),
        "georef_prior_residual_median_px": float(pd.to_numeric(obs.get("prior_residual_px"), errors="coerce").median()) if not obs.empty else None,
    }
    (out / "catalog_summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    (out / "README.txt").write_text(README.format(**{k: summ[k] for k in ["circuit", "observations", "structures_catalogued",
                                                                           "reference_structures", "min_confidence"]}), encoding="utf-8")
    return summ


README = """Structure catalogue - circuit {circuit}
=====================================
Observations: {observations}   Structures catalogued: {structures_catalogued}/{reference_structures}
Minimum confidence: {min_confidence}

crops/                               clean, unmodified detector crops (one folder per structure)
structure_catalog_observations.csv   every observation, tagged with structure id/type/location,
                                     source frame, pass/view direction, camera EO, pixel boxes,
                                     detector/ownership scores, grade and best-view rank
structure_catalog_structures.csv     per reference structure: location, counts, best views, coverage
structures.geojson, observations.geojson   WGS84 points / camera->structure rays for GIS review
georef/tie_points.csv                image (u,v of detected tower base) <-> world (X,Y,ground Z) pairs
georef/residuals_by_*.csv            detection - projection residuals; EO_BIAS_SUSPECT flags
fli/structure_photo_match_qa.csv     FLI_frame_maker assignment CSV (best view per structure)
fli/structure_photo_match_all_views.csv
fli/images/                          crops with the raw frame's EXIF inserted (no pixel re-encoding)

FLI framing (FLI_frame_maker/current):
  python 03_frame_generator_v4.py --batch --csv fli/structure_photo_match_qa.csv --images fli/images ...

Grades: A = competitor-aware HIGH confidence; B = HIGH/MEDIUM with weaker evidence; C = ungraded
(pre-v0.15 run) or per-pass residual OUTLIER (detection inconsistent with the rest of the pass -
typically a neighbouring circuit or adjacent structure; review before use).
Best-view ranks are metadata only - every observation is kept.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a georeferenced structure catalogue from a continuous-oblique run.")
    ap.add_argument("run_output", type=Path, help="Output folder of continuous_oblique_v015.py (or v014 with --min-confidence ANY)")
    ap.add_argument("--circuit", default=None)
    ap.add_argument("--epsg", type=int, default=None, help="Working EPSG. Default: summary.json working_epsg")
    ap.add_argument("--catalog-root", type=Path, default=None, help="Default: <run>/structure_catalog")
    ap.add_argument("--min-confidence", choices=["HIGH", "MEDIUM", "ANY"], default="MEDIUM")
    ap.add_argument("--raw-dir", action="append", default=None,
                    help="Folder with raw frames, used when raw_path in the CSV is not reachable (EXIF for FLI). Repeatable.")
    ap.add_argument("--fli-all-images", action="store_true", help="Keep FLI images for every view, not only best views.")
    ap.add_argument("--bias-flag-px", type=float, default=400.0, help="Flag passes/images whose median prior residual exceeds this. Default: 400.")
    ap.add_argument("--outlier-k", type=float, default=3.0, help="Robust-z threshold for per-pass residual outliers. Default: 3.")
    ap.add_argument("--outlier-min-px", type=float, default=300.0, help="Minimum residual deviation (px) before an outlier is flagged. Default: 300.")
    ap.add_argument("--outlier-min-n", type=int, default=4, help="Observations per pass needed for the consistency check. Default: 4.")
    args = ap.parse_args(argv)
    s = build_catalog(args)
    print("\nStructure catalogue complete")
    for k in ["observations", "observation_confidence_counts", "quality_grade_counts", "residual_consistency_counts", "structures_catalogued",
              "reference_structures", "excluded_by_confidence", "missing_crop_files", "fli_best_view_rows",
              "fli_rows_with_raw_exif", "georef_prior_residual_median_px"]:
        print(f"  {k:32s} {s[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
