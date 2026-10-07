#!/usr/bin/env python
"""
Continuous Oblique Imagery POC v0.1

Purpose
-------
Build a geometry-first continuous-oblique QA model from an NM-style project root.
This version deliberately does NOT use DINOv3 and does NOT alter/copy/rename source images.

Implemented in v0.1
-------------------
- Auto-discovery of NM product folders and highest vN folders
- PLS-CADD staking XML -> structures, spans, alignment, chainage
- DXF image footprints -> image polygons keyed by filename
- EO TXT -> camera positions/orientations keyed by filename
- Raw image index
- Existing 163 naming index for benchmark/reference only
- Flight-pass segmentation from EO trajectory
- FORWARD_PASS / BACKWARD_PASS / CROSSING_PASS classification
- Per-image alignment coverage and primary span / primary structure
- Per-image geometric FORWARD / BACKWARD / SIDE view classification
- Structure observations for every footprint that contains a structure
- Pairwise transition QA and hard-gap detection per pass
- GeoPackage + CSV outputs

Not implemented yet
-------------------
- LiDAR structure-position refinement
- Exact camera projection using camera calibration
- Structure crops
- DINOv3 visual continuity / FLI matching

Example
-------
python continuous_oblique_v01.py "D:\\Sev\\Obliques\\NYPA" --circuit DH1

Dependencies
------------
pandas, numpy, shapely>=2, geopandas, pyproj, ezdxf, pyogrio
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import xml.etree.ElementTree as ET

try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import ezdxf
    from pyproj import CRS
    from shapely.geometry import (
        Point,
        LineString,
        MultiLineString,
        GeometryCollection,
        Polygon,
    )
    from shapely.ops import substring
    from shapely.strtree import STRtree
except ImportError as exc:
    missing = getattr(exc, "name", str(exc))
    raise SystemExit(
        f"Missing Python dependency: {missing}\n"
        "Install requirements first:\n"
        "  python -m pip install pandas numpy shapely geopandas pyproj ezdxf pyogrio"
    ) from exc


# -----------------------------------------------------------------------------
# Configuration / utilities
# -----------------------------------------------------------------------------

VERSION_RE = re.compile(r"^v(?P<n>\d+)$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
FINAL_163_RE = re.compile(
    r"^(?P<circuit>[^_]+)_"
    r"(?P<span_from>\d+)-(?P<span_to>\d+)_"
    r"(?P<output_id>\d+)_"
    r"(?P<date>\d{6})_"
    r"(?P<version>v\d+)\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)

# PLS-CADD State Plane zone -> common NAD83 ftUS EPSG mapping.
# Extend this dictionary for other project templates when encountered.
PLS_ZONE_TO_EPSG = {
    "3101": 2260,  # NAD83 / New York East (ftUS)
    "3102": 2261,  # NAD83 / New York Central (ftUS)
    "3103": 2262,  # NAD83 / New York West (ftUS)
    "3104": 2263,  # NAD83 / New York Long Island (ftUS)
}


@dataclass
class ProjectPaths:
    root: Path
    raw_dir: Path | None
    final_163_dir: Path | None
    eo_dir: Path | None
    lidar_dir: Path | None
    staking_dir: Path | None
    staking_xml: Path | None
    footprint_dxfs: list[Path]


def latest_version_dir(parent: Path) -> Path | None:
    if not parent.exists() or not parent.is_dir():
        return None
    versions: list[tuple[int, Path]] = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        m = VERSION_RE.match(p.name)
        if m:
            versions.append((int(m.group("n")), p))
    if not versions:
        return None
    versions.sort(key=lambda x: x[0])
    return versions[-1][1]


def discover_project(root: Path) -> ProjectPaths:
    product_dirs = {
        "raw": root / "130_raw_overlapping_oblique_imagery",
        "final": root / "163_continuous_oblique_imagery",
        "eo": root / "277_camera_eo",
        "lidar": root / "30_classified_tiled_point_cloud",
        "staking": root / "40_pls_cadd_staking_table",
    }

    raw_dir = latest_version_dir(product_dirs["raw"])
    final_dir = latest_version_dir(product_dirs["final"])
    eo_dir = latest_version_dir(product_dirs["eo"])
    lidar_dir = latest_version_dir(product_dirs["lidar"])
    staking_dir = latest_version_dir(product_dirs["staking"])

    staking_xml = None
    if staking_dir:
        xmls = sorted(staking_dir.glob("*.xml"))
        if xmls:
            staking_xml = xmls[0]

    dxfs = sorted(root.glob("*footprint*.dxf")) + sorted(root.glob("*Footprint*.dxf"))
    # De-duplicate on case-insensitive Windows-like name semantics.
    unique: dict[str, Path] = {}
    for p in dxfs:
        unique[str(p).lower()] = p

    return ProjectPaths(
        root=root,
        raw_dir=raw_dir,
        final_163_dir=final_dir,
        eo_dir=eo_dir,
        lidar_dir=lidar_dir,
        staking_dir=staking_dir,
        staking_xml=staking_xml,
        footprint_dxfs=list(unique.values()),
    )


def text_or_none(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    s = el.text.strip().replace("\xa0", " ")
    return s if s else None


def float_or_nan(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def circular_angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def unit_defaults(crs: CRS) -> dict[str, float]:
    """Threshold defaults in CRS working units."""
    unit_name = ""
    try:
        unit_name = (crs.axis_info[0].unit_name or "").lower()
    except Exception:
        pass

    if "foot" in unit_name or "feet" in unit_name:
        return {
            "max_eo_distance": 20000.0,
            "gap_tolerance": 5.0,
            "min_overlap": 50.0,
            "structure_tolerance": 10.0,
        }
    return {
        "max_eo_distance": 6000.0,
        "gap_tolerance": 2.0,
        "min_overlap": 15.0,
        "structure_tolerance": 3.0,
    }


# -----------------------------------------------------------------------------
# Staking / alignment
# -----------------------------------------------------------------------------


def parse_staking_xml(path: Path, epsg_override: int | None = None):
    root = ET.parse(path).getroot()
    creator = root.find("creator")
    creator_attrs = dict(creator.attrib) if creator is not None else {}

    if epsg_override is not None:
        epsg = epsg_override
    else:
        zone = creator_attrs.get("zone", "")
        epsg = PLS_ZONE_TO_EPSG.get(zone)
        if epsg is None:
            raise ValueError(
                f"Cannot infer EPSG from PLS-CADD zone={zone!r}. "
                "Run again with --epsg <EPSG>."
            )
    crs = CRS.from_epsg(epsg)

    table = root.find(".//table[@tagname='staking_table']")
    if table is None:
        raise ValueError("No PLS-CADD staking_table found in XML")

    rows: list[dict] = []
    for row in table.findall("staking_table"):
        def get(tag: str) -> str | None:
            return text_or_none(row.find(tag))

        rownum = int(row.attrib.get("rownum", len(rows)))
        rec = {
            "rownum": rownum,
            "structure_idx": rownum + 1,
            "structure_id": get("structure_number") or f"STRUCTURE_{rownum + 1}",
            "station": float_or_nan(get("station")),
            "x": float_or_nan(get("x_easting")),
            "y": float_or_nan(get("y_northing")),
            "centerline_z": float_or_nan(get("centerline_z_elevation")),
            "tin_z": float_or_nan(get("tin_z_elevation")),
            "ahead_span": float_or_nan(get("ahead_span")),
            "line_angle": float_or_nan(get("line_angle")),
            "transverse_axis_azimuth": float_or_nan(get("transverse_axis_azimuth")),
            "structure_height": float_or_nan(get("struct_height")),
            "structure_description": get("structure_description"),
            "structure_type": get("structure_comment_2"),
        }
        if math.isfinite(rec["x"]) and math.isfinite(rec["y"]):
            rows.append(rec)

    if len(rows) < 2:
        raise ValueError("Fewer than two coordinate-bearing staking rows found")

    df = pd.DataFrame(rows).sort_values("rownum").reset_index(drop=True)

    # PLS row 0 can have a blank station; for an ordered single circuit this is 0.
    if pd.isna(df.loc[0, "station"]):
        df.loc[0, "station"] = 0.0

    points = [Point(xy) for xy in zip(df["x"], df["y"])]
    alignment = LineString([(p.x, p.y) for p in points])

    # Geometric chainage is used consistently for spatial projection.
    geom_chainages = [alignment.project(p) for p in points]
    df["chainage"] = geom_chainages

    structures = gpd.GeoDataFrame(df, geometry=points, crs=crs)

    span_rows: list[dict] = []
    for i in range(len(structures) - 1):
        a = structures.iloc[i]
        b = structures.iloc[i + 1]
        geom = LineString([a.geometry, b.geometry])
        span_rows.append(
            {
                "span_id": f"{int(a.structure_idx)}-{int(b.structure_idx)}",
                "from_idx": int(a.structure_idx),
                "to_idx": int(b.structure_idx),
                "from_structure": a.structure_id,
                "to_structure": b.structure_id,
                "chainage_from": float(a.chainage),
                "chainage_to": float(b.chainage),
                "geom_length": float(geom.length),
                "pls_station_from": float(a.station) if pd.notna(a.station) else np.nan,
                "pls_station_to": float(b.station) if pd.notna(b.station) else np.nan,
                "pls_ahead_span": float(a.ahead_span) if pd.notna(a.ahead_span) else np.nan,
                "geometry": geom,
            }
        )
    spans = gpd.GeoDataFrame(span_rows, geometry="geometry", crs=crs)
    alignment_gdf = gpd.GeoDataFrame(
        [{"alignment_id": "PRIMARY", "length": alignment.length, "geometry": alignment}],
        geometry="geometry",
        crs=crs,
    )

    metadata = {
        "creator": creator_attrs,
        "epsg": epsg,
        "crs": crs,
        "alignment_length": float(alignment.length),
    }
    return structures, spans, alignment_gdf, metadata


def local_tangent(alignment: LineString, chainage: float) -> np.ndarray:
    eps = max(alignment.length * 1e-6, 0.1)
    a = max(0.0, chainage - eps)
    b = min(alignment.length, chainage + eps)
    if b <= a:
        return np.array([1.0, 0.0], dtype=float)
    p0 = alignment.interpolate(a)
    p1 = alignment.interpolate(b)
    v = np.array([p1.x - p0.x, p1.y - p0.y], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0], dtype=float)


def classify_camera_view(
    camera: Point,
    target: Point,
    tangent: np.ndarray,
    forward_max_deg: float = 60.0,
    backward_min_deg: float = 120.0,
) -> tuple[str, float, float, float]:
    """
    Geometric view class based on camera->target vector vs increasing alignment tangent.

    FORWARD  : camera is behind target and line of sight points with increasing chainage.
    BACKWARD : camera is ahead of target and line of sight points against increasing chainage.
    SIDE     : line of sight is largely perpendicular to alignment.
    """
    v = np.array([target.x - camera.x, target.y - camera.y], dtype=float)
    dist = float(np.linalg.norm(v))
    if dist == 0:
        return "UNKNOWN", float("nan"), 0.0, 0.0
    u = v / dist
    dot = float(np.clip(np.dot(u, tangent), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    along = float(np.dot(v, tangent))
    cross = float(tangent[0] * v[1] - tangent[1] * v[0])
    if angle <= forward_max_deg:
        cls = "FORWARD"
    elif angle >= backward_min_deg:
        cls = "BACKWARD"
    else:
        cls = "SIDE"
    return cls, angle, along, cross


# -----------------------------------------------------------------------------
# EO parser / pass segmentation
# -----------------------------------------------------------------------------


def parse_eo_file(path: Path) -> pd.DataFrame:
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\s+", s)

            # Observed NM format:
            # time X Y Z O P K filename
            if len(parts) >= 8 and parts[-1].lower().endswith(tuple(IMAGE_EXTS)):
                try:
                    nums = [float(x) for x in parts[:7]]
                except ValueError:
                    continue
                records.append(
                    {
                        "eo_source": path.name,
                        "eo_line": line_no,
                        "eo_order": len(records),
                        "time": nums[0],
                        "camera_x": nums[1],
                        "camera_y": nums[2],
                        "camera_z": nums[3],
                        "omega": nums[4],
                        "phi": nums[5],
                        "kappa": nums[6],
                        "filename": Path(parts[-1]).name,
                    }
                )
                continue

            # Fallback documented format: Image X Y Z O P K
            if len(parts) >= 7 and parts[0].lower().endswith(tuple(IMAGE_EXTS)):
                try:
                    nums = [float(x) for x in parts[1:7]]
                except ValueError:
                    continue
                records.append(
                    {
                        "eo_source": path.name,
                        "eo_line": line_no,
                        "eo_order": len(records),
                        "time": np.nan,
                        "camera_x": nums[0],
                        "camera_y": nums[1],
                        "camera_z": nums[2],
                        "omega": nums[3],
                        "phi": nums[4],
                        "kappa": nums[5],
                        "filename": Path(parts[0]).name,
                    }
                )

    return pd.DataFrame(records)


def load_all_eo(eo_dir: Path | None) -> pd.DataFrame:
    if eo_dir is None:
        return pd.DataFrame()
    frames = []
    for path in sorted(eo_dir.glob("*.txt")):
        if "_EO_" not in path.name.upper() and "EO" not in path.stem.upper():
            continue
        df = parse_eo_file(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def project_eo_to_alignment(eo: pd.DataFrame, alignment: LineString) -> pd.DataFrame:
    eo = eo.copy()
    chainages = []
    distances = []
    for x, y in zip(eo["camera_x"], eo["camera_y"]):
        p = Point(float(x), float(y))
        chainages.append(float(alignment.project(p)))
        distances.append(float(p.distance(alignment)))
    eo["camera_chainage"] = chainages
    eo["distance_to_alignment"] = distances
    return eo


def eo_file_compatibility(
    eo: pd.DataFrame, max_distance: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark complete EO files as spatially compatible/incompatible with the alignment."""
    eo = eo.copy()
    summaries = []
    compat_map = {}
    for source, grp in eo.groupby("eo_source", sort=False):
        min_dist = float(grp["distance_to_alignment"].min())
        med_dist = float(grp["distance_to_alignment"].median())
        compatible = min_dist <= max_distance
        compat_map[source] = compatible
        summaries.append(
            {
                "eo_source": source,
                "records": len(grp),
                "min_distance_to_alignment": min_dist,
                "median_distance_to_alignment": med_dist,
                "spatial_compatible": compatible,
            }
        )
    eo["spatial_compatible"] = eo["eo_source"].map(compat_map).fillna(False)
    return eo, pd.DataFrame(summaries)


def _sustained_opposite(move_classes: Sequence[int], idx: int, current_sign: int, n: int = 2) -> bool:
    vals = []
    for j in range(idx, min(len(move_classes), idx + n)):
        if move_classes[j] != 0:
            vals.append(move_classes[j])
    return len(vals) >= n and all(v == -current_sign for v in vals[:n])


def segment_passes_for_source(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("eo_order").copy().reset_index(drop=True)
    n = len(grp)
    if n == 0:
        return grp
    if n == 1:
        grp["pass_local"] = 1
        return grp

    x = grp["camera_x"].to_numpy(float)
    y = grp["camera_y"].to_numpy(float)
    ch = grp["camera_chainage"].to_numpy(float)
    t = grp["time"].to_numpy(float)

    steps = np.hypot(np.diff(x), np.diff(y))
    dch = np.diff(ch)
    ratio = np.divide(np.abs(dch), steps, out=np.zeros_like(dch), where=steps > 1e-9)

    finite_dt = np.diff(t)
    positive_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0)]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else np.nan
    time_break = max(30.0, median_dt * 5.0) if math.isfinite(median_dt) else float("inf")

    positive_steps = steps[steps > 0]
    med_step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
    spatial_break = med_step * 5.0 if med_step > 0 else float("inf")

    # 0 = crossing/indeterminate, +/-1 = along-alignment movement.
    move_classes = [0 if r < 0.45 or abs(dc) < 1e-9 else (1 if dc > 0 else -1) for dc, r in zip(dch, ratio)]

    pass_ids = [1]
    current_pass = 1
    current_sign = 0

    for i in range(n - 1):
        hard_break = False
        if math.isfinite(t[i]) and math.isfinite(t[i + 1]):
            dt = t[i + 1] - t[i]
            if dt > time_break:
                hard_break = True
        if steps[i] > spatial_break:
            hard_break = True
        if math.isfinite(grp.loc[i, "kappa"]) and math.isfinite(grp.loc[i + 1, "kappa"]):
            heading_jump = circular_angle_diff_deg(float(grp.loc[i, "kappa"]), float(grp.loc[i + 1, "kappa"]))
            if heading_jump > 100.0 and move_classes[i] != 0:
                hard_break = True

        mc = move_classes[i]
        if current_sign == 0 and mc != 0:
            current_sign = mc
        elif current_sign != 0 and mc == -current_sign:
            if _sustained_opposite(move_classes, i, current_sign, n=2):
                hard_break = True

        if hard_break:
            current_pass += 1
            current_sign = mc if mc != 0 else 0
        pass_ids.append(current_pass)

    grp["pass_local"] = pass_ids
    return grp


def classify_passes(eo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eo.empty:
        return eo, pd.DataFrame()

    compatible = eo[eo["spatial_compatible"]].copy()
    incompatible = eo[~eo["spatial_compatible"]].copy()

    segmented = []
    for source, grp in compatible.groupby("eo_source", sort=False):
        seg = segment_passes_for_source(grp)
        seg["pass_id"] = [f"{Path(source).stem}_P{int(v):02d}" for v in seg["pass_local"]]
        segmented.append(seg)

    if segmented:
        compatible = pd.concat(segmented, ignore_index=True)
    else:
        compatible = pd.DataFrame(columns=eo.columns)

    pass_rows = []
    direction_map: dict[str, str] = {}
    for pass_id, grp in compatible.groupby("pass_id", sort=False):
        grp = grp.sort_values("eo_order")
        if len(grp) < 2:
            direction = "UNKNOWN_PASS"
            ratio_med = np.nan
            delta = 0.0
        else:
            dx = np.diff(grp["camera_x"].to_numpy(float))
            dy = np.diff(grp["camera_y"].to_numpy(float))
            step = np.hypot(dx, dy)
            dch = np.diff(grp["camera_chainage"].to_numpy(float))
            along_ratio = np.divide(np.abs(dch), step, out=np.zeros_like(dch), where=step > 1e-9)
            ratio_med = float(np.median(along_ratio)) if len(along_ratio) else np.nan
            delta = float(grp.iloc[-1]["camera_chainage"] - grp.iloc[0]["camera_chainage"])

            if math.isfinite(ratio_med) and ratio_med < 0.45:
                direction = "CROSSING_PASS"
            elif abs(delta) < max(1.0, 0.02 * float(np.nansum(step))):
                direction = "CROSSING_PASS"
            else:
                direction = "FORWARD_PASS" if delta > 0 else "BACKWARD_PASS"

        direction_map[pass_id] = direction
        coords = list(zip(grp["camera_x"].astype(float), grp["camera_y"].astype(float)))
        geom = LineString(coords) if len(coords) >= 2 else Point(coords[0]) if coords else None
        pass_rows.append(
            {
                "pass_id": pass_id,
                "eo_source": grp.iloc[0]["eo_source"] if len(grp) else None,
                "image_count": len(grp),
                "pass_direction": direction,
                "chainage_start": float(grp.iloc[0]["camera_chainage"]) if len(grp) else np.nan,
                "chainage_end": float(grp.iloc[-1]["camera_chainage"]) if len(grp) else np.nan,
                "alignment_progression_ratio": ratio_med,
                "geometry": geom,
            }
        )

    if not compatible.empty:
        compatible["pass_direction"] = compatible["pass_id"].map(direction_map)

    if not incompatible.empty:
        incompatible["pass_local"] = np.nan
        incompatible["pass_id"] = None
        incompatible["pass_direction"] = "OUTSIDE_ALIGNMENT"

    combined = pd.concat([compatible, incompatible], ignore_index=True, sort=False)
    return combined, pd.DataFrame(pass_rows)


# -----------------------------------------------------------------------------
# DXF footprints
# -----------------------------------------------------------------------------


def _entity_text(entity) -> str | None:
    t = entity.dxftype()
    if t == "TEXT":
        return str(entity.dxf.text).strip()
    if t == "MTEXT":
        try:
            return entity.plain_text().strip()
        except Exception:
            return str(entity.text).strip()
    return None


def parse_footprints_dxf(path: Path, crs: CRS) -> gpd.GeoDataFrame:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    polygons: list[Polygon] = []
    poly_meta: list[dict] = []
    text_points: list[Point] = []
    text_values: list[str] = []

    for entity in msp:
        typ = entity.dxftype()
        if typ == "LWPOLYLINE":
            pts = [(float(x), float(y)) for x, y, *_ in entity.get_points("xy")]
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.area > 0:
                    polygons.append(poly)
                    poly_meta.append(
                        {
                            "dxf_source": path.name,
                            "dxf_handle": entity.dxf.handle,
                            "layer": entity.dxf.layer,
                        }
                    )
        elif typ in {"TEXT", "MTEXT"}:
            val = _entity_text(entity)
            if not val or not val.lower().endswith(tuple(IMAGE_EXTS)):
                continue
            try:
                ins = entity.dxf.insert
                text_points.append(Point(float(ins.x), float(ins.y)))
                text_values.append(Path(val).name)
            except Exception:
                continue

    if not polygons:
        return gpd.GeoDataFrame(columns=["filename", "geometry"], geometry="geometry", crs=crs)

    # Associate filename labels with polygons using spatial containment first.
    tree = STRtree(polygons)
    assignments: dict[int, tuple[str, float]] = {}
    for p, name in zip(text_points, text_values):
        candidates = list(tree.query(p, predicate="intersects"))
        if candidates:
            best = min(candidates, key=lambda idx: polygons[int(idx)].centroid.distance(p))
            assignments[int(best)] = (name, 0.0)
        else:
            # Fallback: nearest polygon. DXF text is expected near footprint centre.
            nearest = tree.query_nearest(p)
            if np.isscalar(nearest):
                idx = int(nearest)
            else:
                idx = int(np.asarray(nearest).ravel()[0])
            d = polygons[idx].centroid.distance(p)
            current = assignments.get(idx)
            if current is None or d < current[1]:
                assignments[idx] = (name, float(d))

    records = []
    for i, poly in enumerate(polygons):
        meta = dict(poly_meta[i])
        pair = assignments.get(i)
        meta["filename"] = pair[0] if pair else None
        meta["label_distance"] = pair[1] if pair else np.nan
        meta["geometry"] = poly
        records.append(meta)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    # Multiple DXFs can overlap; filename is the natural key.
    return gdf


def load_footprints(paths: Sequence[Path], crs: CRS) -> gpd.GeoDataFrame:
    frames = [parse_footprints_dxf(p, crs) for p in paths]
    frames = [x for x in frames if not x.empty]
    if not frames:
        return gpd.GeoDataFrame(columns=["filename", "geometry"], geometry="geometry", crs=crs)
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    gdf = gdf[gdf["filename"].notna()].copy()
    gdf = gdf.drop_duplicates(subset=["filename"], keep="first")
    return gdf.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Image indexing / existing 163 reference
# -----------------------------------------------------------------------------


def index_raw_images(raw_dir: Path | None) -> pd.DataFrame:
    if raw_dir is None:
        return pd.DataFrame(columns=["filename", "raw_path"])
    rows = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rows.append({"filename": p.name, "raw_path": str(p)})
    return pd.DataFrame(rows)


def parse_mmddyy(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%m%d%y").date().isoformat()
    except ValueError:
        return None


def index_existing_163(final_dir: Path | None) -> pd.DataFrame:
    if final_dir is None:
        return pd.DataFrame()
    rows = []
    for p in sorted(final_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        m = FINAL_163_RE.match(p.name)
        if not m:
            rows.append({"filename": p.name, "parse_status": "UNPARSED", "path": str(p)})
            continue
        d = m.groupdict()
        rows.append(
            {
                "filename": p.name,
                "parse_status": "OK",
                "circuit": d["circuit"],
                "span_from": int(d["span_from"]),
                "span_to": int(d["span_to"]),
                "span_id": f"{d['span_from']}-{d['span_to']}",
                "output_id": int(d["output_id"]),
                "capture_date_code": d["date"],
                "capture_date": parse_mmddyy(d["date"]),
                "version": d["version"],
                "path": str(p),
            }
        )
    return pd.DataFrame(rows)


def infer_circuit(existing_163: pd.DataFrame, staking_xml: Path | None) -> str:
    if not existing_163.empty and "circuit" in existing_163:
        vals = existing_163["circuit"].dropna()
        if len(vals):
            return vals.mode().iloc[0]
    if staking_xml:
        stem = staking_xml.stem
        stem = re.sub(r"_STAKING$", "", stem, flags=re.IGNORECASE)
        return stem.replace("-", "")
    return "CIRCUIT"


# -----------------------------------------------------------------------------
# Coverage / attribution
# -----------------------------------------------------------------------------


def iter_line_parts(geom) -> Iterator[LineString]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from iter_line_parts(g)


def geometry_to_intervals(alignment: LineString, geom) -> list[tuple[float, float]]:
    intervals = []
    for part in iter_line_parts(geom):
        coords = list(part.coords)
        if len(coords) < 2:
            continue
        chainages = [alignment.project(Point(xy)) for xy in coords]
        a, b = float(min(chainages)), float(max(chainages))
        if b > a:
            intervals.append((a, b))
    return merge_intervals(intervals, tolerance=1e-7)


def merge_intervals(intervals: Iterable[tuple[float, float]], tolerance: float = 0.0) -> list[tuple[float, float]]:
    vals = sorted((min(a, b), max(a, b)) for a, b in intervals if math.isfinite(a) and math.isfinite(b))
    if not vals:
        return []
    merged = [list(vals[0])]
    for a, b in vals[1:]:
        if a <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def intervals_length(intervals: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, b - a) for a, b in merge_intervals(intervals)))


def intervals_overlap_length(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    aa = merge_intervals(a)
    bb = merge_intervals(b)
    i = j = 0
    total = 0.0
    while i < len(aa) and j < len(bb):
        left = max(aa[i][0], bb[j][0])
        right = min(aa[i][1], bb[j][1])
        if right > left:
            total += right - left
        if aa[i][1] < bb[j][1]:
            i += 1
        else:
            j += 1
    return float(total)


def intervals_gap(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    if not a or not b:
        return float("nan")
    if intervals_overlap_length(a, b) > 0:
        return 0.0
    return float(min(abs(x - y) for ia in a for ib in b for x in ia for y in ib))


def interval_string(intervals: Sequence[tuple[float, float]]) -> str:
    return ";".join(f"{a:.3f}-{b:.3f}" for a, b in intervals)


def spans_for_intervals(spans: gpd.GeoDataFrame, intervals: Sequence[tuple[float, float]]) -> list[str]:
    out = []
    for _, s in spans.iterrows():
        sf = float(s.chainage_from)
        st = float(s.chainage_to)
        if any(min(b, st) - max(a, sf) > 0 for a, b in intervals):
            out.append(str(s.span_id))
    return out


def primary_span_for_chainage(spans: gpd.GeoDataFrame, chainage: float) -> str | None:
    hit = spans[(spans["chainage_from"] <= chainage) & (spans["chainage_to"] >= chainage)]
    if len(hit):
        return str(hit.iloc[0]["span_id"])
    return None


def primary_structure_for_chainage(structures: gpd.GeoDataFrame, chainage: float) -> tuple[str, int] | tuple[None, None]:
    if structures.empty:
        return None, None
    idx = (structures["chainage"].astype(float) - chainage).abs().idxmin()
    s = structures.loc[idx]
    return str(s.structure_id), int(s.structure_idx)


def build_image_model(
    raw: pd.DataFrame,
    eo: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    spans: gpd.GeoDataFrame,
    alignment: LineString,
    crs: CRS,
    reference_mode: str,
    allow_side: bool,
):
    # Raw is left-side master so missing metadata remains visible in CSV.
    images = raw.copy()
    if eo.empty:
        for c in ["camera_x", "camera_y", "camera_z", "omega", "phi", "kappa", "pass_id", "pass_direction"]:
            images[c] = np.nan
    else:
        eo_cols = [
            "filename", "eo_source", "eo_order", "time", "camera_x", "camera_y", "camera_z",
            "omega", "phi", "kappa", "camera_chainage", "distance_to_alignment",
            "spatial_compatible", "pass_id", "pass_direction",
        ]
        images = images.merge(eo[eo_cols], on="filename", how="left")

    fp = footprints[["filename", "dxf_source", "geometry"]].rename(columns={"geometry": "footprint_geometry"})
    images = images.merge(pd.DataFrame(fp), on="filename", how="left")

    records = []
    coverage_geoms = []
    for _, r in images.iterrows():
        footprint = r.get("footprint_geometry")
        intervals: list[tuple[float, float]] = []
        coverage_geom = None
        if footprint is not None and not pd.isna(footprint):
            try:
                coverage_geom = alignment.intersection(footprint)
                intervals = geometry_to_intervals(alignment, coverage_geom)
            except Exception:
                intervals = []

        cov_len = intervals_length(intervals)
        if intervals:
            # Centre of total min/max extent; preserves circuit progression even for multi-part intersections.
            centre = (min(a for a, _ in intervals) + max(b for _, b in intervals)) / 2.0
        else:
            centre = np.nan

        primary_span = primary_span_for_chainage(spans, centre) if math.isfinite(centre) else None
        pstruct, pstruct_idx = primary_structure_for_chainage(structures, centre) if math.isfinite(centre) else (None, None)
        visible_spans = spans_for_intervals(spans, intervals) if intervals else []

        view = "UNKNOWN"
        view_angle = np.nan
        along_offset = np.nan
        cross_offset = np.nan
        if math.isfinite(centre) and pd.notna(r.get("camera_x")) and pd.notna(r.get("camera_y")):
            camera = Point(float(r.camera_x), float(r.camera_y))
            if reference_mode == "structure" and pstruct_idx is not None:
                target_row = structures[structures["structure_idx"] == pstruct_idx].iloc[0]
                target = target_row.geometry
                target_chainage = float(target_row.chainage)
            else:
                target = alignment.interpolate(float(centre))
                target_chainage = float(centre)
            tangent = local_tangent(alignment, target_chainage)
            view, view_angle, along_offset, cross_offset = classify_camera_view(camera, target, tangent)

        pass_direction = r.get("pass_direction")
        reason = None
        valid = True
        if not intervals:
            valid = False
            reason = "NO_ALIGNMENT_COVERAGE"
        elif pd.isna(r.get("eo_source")):
            valid = False
            reason = "MISSING_EO"
        elif pass_direction in {"CROSSING_PASS", "OUTSIDE_ALIGNMENT", "UNKNOWN_PASS", None} or pd.isna(pass_direction):
            valid = False
            reason = str(pass_direction) if pd.notna(pass_direction) else "NO_PASS"
        elif view == "SIDE" and not allow_side:
            valid = False
            reason = "SIDE_VIEW"

        reference_id = primary_span if reference_mode == "span" else pstruct
        rec = r.drop(labels=["footprint_geometry"], errors="ignore").to_dict()
        rec.update(
            {
                "coverage_intervals": interval_string(intervals),
                "coverage_from": min((a for a, _ in intervals), default=np.nan),
                "coverage_to": max((b for _, b in intervals), default=np.nan),
                "coverage_length": cov_len,
                "centre_chainage": centre,
                "visible_spans": ";".join(visible_spans),
                "primary_span": primary_span,
                "primary_structure": pstruct,
                "primary_structure_idx": pstruct_idx,
                "reference_mode": reference_mode,
                "reference_id": reference_id,
                "view_direction": view,
                "view_angle_deg": view_angle,
                "along_track_offset": along_offset,
                "cross_track_offset": cross_offset,
                "valid_for_product": bool(valid),
                "rejection_reason": reason,
            }
        )
        records.append(rec)
        coverage_geoms.append(coverage_geom)

    images_df = pd.DataFrame(records)

    # Spatial images layer contains only camera records that are spatially compatible with this alignment.
    spatial_mask = (
        images_df["camera_x"].notna()
        & images_df["camera_y"].notna()
        & images_df.get("spatial_compatible", pd.Series(False, index=images_df.index)).fillna(False)
    )
    image_points = [Point(x, y) for x, y in zip(images_df.loc[spatial_mask, "camera_x"], images_df.loc[spatial_mask, "camera_y"])]
    images_gdf = gpd.GeoDataFrame(images_df.loc[spatial_mask].copy(), geometry=image_points, crs=crs)

    cov_rows = []
    for rec, geom in zip(records, coverage_geoms):
        if geom is None or geom.is_empty:
            continue
        cov_rows.append(
            {
                "filename": rec["filename"],
                "pass_id": rec.get("pass_id"),
                "pass_direction": rec.get("pass_direction"),
                "primary_span": rec.get("primary_span"),
                "reference_id": rec.get("reference_id"),
                "coverage_from": rec.get("coverage_from"),
                "coverage_to": rec.get("coverage_to"),
                "coverage_length": rec.get("coverage_length"),
                "valid_for_product": rec.get("valid_for_product"),
                "geometry": geom,
            }
        )
    coverage_gdf = gpd.GeoDataFrame(cov_rows, geometry="geometry", crs=crs) if cov_rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)

    return images_df, images_gdf, coverage_gdf


def build_structure_observations(
    images_df: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    alignment: LineString,
    crs: CRS,
    tolerance: float,
) -> gpd.GeoDataFrame:
    if footprints.empty or structures.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)

    image_lookup = images_df.set_index("filename").to_dict("index")
    fp_lookup = {r.filename: r.geometry for _, r in footprints.iterrows() if pd.notna(r.filename)}
    rows = []

    for _, s in structures.iterrows():
        sp = s.geometry
        test_geom = sp.buffer(tolerance) if tolerance > 0 else sp
        tangent = local_tangent(alignment, float(s.chainage))
        for filename, poly in fp_lookup.items():
            if not poly.intersects(test_geom):
                continue
            img = image_lookup.get(filename)
            if img is None or pd.isna(img.get("camera_x")) or pd.isna(img.get("camera_y")):
                view, angle, along, cross = "UNKNOWN", np.nan, np.nan, np.nan
                cam_dist = np.nan
            else:
                camera = Point(float(img["camera_x"]), float(img["camera_y"]))
                view, angle, along, cross = classify_camera_view(camera, sp, tangent)
                cam_dist = float(camera.distance(sp))

            rows.append(
                {
                    "structure_idx": int(s.structure_idx),
                    "structure_id": s.structure_id,
                    "structure_chainage": float(s.chainage),
                    "filename": filename,
                    "pass_id": img.get("pass_id") if img else None,
                    "pass_direction": img.get("pass_direction") if img else None,
                    "view_direction": view,
                    "view_angle_deg": angle,
                    "camera_distance_2d": cam_dist,
                    "along_track_offset": along,
                    "cross_track_offset": cross,
                    "image_valid_for_product": bool(img.get("valid_for_product")) if img else False,
                    "geometry": sp,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


# -----------------------------------------------------------------------------
# Transition QA / pass gaps
# -----------------------------------------------------------------------------


def parse_interval_string(s: str | float | None) -> list[tuple[float, float]]:
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(";"):
        try:
            a, b = part.split("-", 1)
            out.append((float(a), float(b)))
        except ValueError:
            continue
    return out


def build_transitions(
    images_df: pd.DataFrame,
    crs: CRS,
    gap_tolerance: float,
    min_overlap: float,
) -> gpd.GeoDataFrame:
    rows = []
    candidates = images_df[images_df["pass_id"].notna()].copy()
    for pass_id, grp in candidates.groupby("pass_id", sort=False):
        grp = grp.sort_values("eo_order")
        vals = list(grp.to_dict("records"))
        for a, b in zip(vals[:-1], vals[1:]):
            ia = parse_interval_string(a.get("coverage_intervals"))
            ib = parse_interval_string(b.get("coverage_intervals"))
            overlap = intervals_overlap_length(ia, ib)
            gap = intervals_gap(ia, ib)
            progression = float(b.get("centre_chainage", np.nan)) - float(a.get("centre_chainage", np.nan)) if pd.notna(a.get("centre_chainage")) and pd.notna(b.get("centre_chainage")) else np.nan
            direction = a.get("pass_direction")

            if not ia or not ib:
                status = "MISSING_COVERAGE"
            elif math.isfinite(gap) and gap > gap_tolerance:
                status = "HARD_GAP"
            elif direction == "FORWARD_PASS" and math.isfinite(progression) and progression < -gap_tolerance:
                status = "SEQUENCE_JUMP"
            elif direction == "BACKWARD_PASS" and math.isfinite(progression) and progression > gap_tolerance:
                status = "SEQUENCE_JUMP"
            elif overlap < min_overlap:
                status = "WEAK_OVERLAP"
            else:
                status = "OK"

            geom = None
            if pd.notna(a.get("camera_x")) and pd.notna(a.get("camera_y")) and pd.notna(b.get("camera_x")) and pd.notna(b.get("camera_y")):
                geom = LineString([(float(a["camera_x"]), float(a["camera_y"])), (float(b["camera_x"]), float(b["camera_y"]))])

            rows.append(
                {
                    "pass_id": pass_id,
                    "pass_direction": direction,
                    "image_a": a.get("filename"),
                    "image_b": b.get("filename"),
                    "reference_a": a.get("reference_id"),
                    "reference_b": b.get("reference_id"),
                    "chainage_a": a.get("centre_chainage"),
                    "chainage_b": b.get("centre_chainage"),
                    "progression": progression,
                    "coverage_overlap": overlap,
                    "coverage_gap": gap,
                    "transition_status": status,
                    "geometry": geom,
                }
            )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


def build_pass_gaps(
    images_df: pd.DataFrame,
    alignment: LineString,
    crs: CRS,
    gap_tolerance: float,
) -> gpd.GeoDataFrame:
    rows = []
    candidates = images_df[(images_df["pass_id"].notna()) & (images_df["valid_for_product"])].copy()

    for pass_id, grp in candidates.groupby("pass_id", sort=False):
        direction = grp["pass_direction"].dropna().iloc[0] if grp["pass_direction"].notna().any() else None
        all_intervals = []
        for s in grp["coverage_intervals"]:
            all_intervals.extend(parse_interval_string(s))
        merged = merge_intervals(all_intervals, tolerance=gap_tolerance)
        if not merged:
            continue

        # A continuous pass is expected to cover the whole circuit, not only the
        # portion reached by its first/last accepted image. Therefore start/end
        # gaps are genuine QA gaps as well as internal gaps.
        cursor = 0.0
        gap_no = 0
        for a, b in merged:
            a = max(0.0, a)
            b = min(float(alignment.length), b)
            if a - cursor > gap_tolerance:
                gap_no += 1
                rows.append(
                    {
                        "gap_id": f"{pass_id}_G{gap_no:03d}",
                        "pass_id": pass_id,
                        "pass_direction": direction,
                        "gap_from": cursor,
                        "gap_to": a,
                        "gap_length": a - cursor,
                        "gap_type": "HARD_GAP",
                        "geometry": substring(alignment, cursor, a),
                    }
                )
            cursor = max(cursor, b)
        if float(alignment.length) - cursor > gap_tolerance:
            gap_no += 1
            rows.append(
                {
                    "gap_id": f"{pass_id}_G{gap_no:03d}",
                    "pass_id": pass_id,
                    "pass_direction": direction,
                    "gap_from": cursor,
                    "gap_to": float(alignment.length),
                    "gap_length": float(alignment.length) - cursor,
                    "gap_type": "HARD_GAP",
                    "geometry": substring(alignment, cursor, float(alignment.length)),
                }
            )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def safe_csv(df: pd.DataFrame, path: Path):
    out = df.copy()
    if isinstance(out, gpd.GeoDataFrame):
        out = pd.DataFrame(out.drop(columns="geometry"))
    out.to_csv(path, index=False)


def write_gpkg_layer(gdf: gpd.GeoDataFrame, gpkg: Path, layer: str, first: bool = False):
    if gdf is None or gdf.empty:
        return
    # Avoid unsupported Python list/dict values.
    out = gdf.copy()
    for col in out.columns:
        if col == out.geometry.name:
            continue
        if out[col].map(lambda x: isinstance(x, (list, tuple, dict, set))).any():
            out[col] = out[col].map(lambda x: json.dumps(x) if isinstance(x, (list, tuple, dict, set)) else x)
    out.to_file(gpkg, layer=layer, driver="GPKG", mode="w" if first else "a")


def build_summary(
    paths: ProjectPaths,
    metadata: dict,
    raw: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    eo: pd.DataFrame,
    eo_summary: pd.DataFrame,
    structures: gpd.GeoDataFrame,
    spans: gpd.GeoDataFrame,
    passes: pd.DataFrame,
    images: pd.DataFrame,
    transitions: gpd.GeoDataFrame,
    gaps: gpd.GeoDataFrame,
    existing: pd.DataFrame,
    circuit: str,
) -> dict:
    pass_summary = []
    if not passes.empty:
        for _, p in passes.iterrows():
            pid = p["pass_id"]
            im = images[images["pass_id"] == pid]
            valid = im[im["valid_for_product"]]
            pass_summary.append(
                {
                    "pass_id": pid,
                    "direction": p["pass_direction"],
                    "images": int(len(im)),
                    "valid_images": int(len(valid)),
                    "coverage_from": float(valid["coverage_from"].min()) if len(valid) else None,
                    "coverage_to": float(valid["coverage_to"].max()) if len(valid) else None,
                    "hard_gaps": int(len(gaps[gaps["pass_id"] == pid])) if not gaps.empty else 0,
                    "transition_flags": dict(Counter(transitions.loc[transitions["pass_id"] == pid, "transition_status"])) if not transitions.empty else {},
                }
            )

    existing_dates = {}
    if not existing.empty and "capture_date" in existing:
        existing_dates = {str(k): int(v) for k, v in existing["capture_date"].value_counts(dropna=False).items()}

    return {
        "poc_version": "0.1",
        "circuit": circuit,
        "project_root": str(paths.root),
        "working_epsg": metadata["epsg"],
        "alignment_length": metadata["alignment_length"],
        "inputs": {
            "raw_dir": str(paths.raw_dir) if paths.raw_dir else None,
            "eo_dir": str(paths.eo_dir) if paths.eo_dir else None,
            "lidar_dir": str(paths.lidar_dir) if paths.lidar_dir else None,
            "staking_xml": str(paths.staking_xml) if paths.staking_xml else None,
            "footprint_dxfs": [str(x) for x in paths.footprint_dxfs],
            "existing_163_dir": str(paths.final_163_dir) if paths.final_163_dir else None,
        },
        "counts": {
            "raw_images": int(len(raw)),
            "footprints": int(len(footprints)),
            "eo_records": int(len(eo)),
            "structures": int(len(structures)),
            "spans": int(len(spans)),
            "passes": int(len(passes)),
            "valid_product_images": int(images["valid_for_product"].sum()) if len(images) else 0,
            "existing_163_images": int(len(existing)),
        },
        "eo_sources": eo_summary.to_dict("records") if not eo_summary.empty else [],
        "passes": pass_summary,
        "existing_163_capture_dates": existing_dates,
    }


def validate_required_paths(paths: ProjectPaths):
    missing = []
    if paths.raw_dir is None:
        missing.append("130_raw_overlapping_oblique_imagery/vN")
    if paths.eo_dir is None:
        missing.append("277_camera_eo/vN")
    if paths.staking_xml is None:
        missing.append("40_pls_cadd_staking_table/vN/*.xml")
    if not paths.footprint_dxfs:
        missing.append("*footprint*.dxf in project root")
    if missing:
        raise FileNotFoundError("Required project inputs not found:\n  - " + "\n  - ".join(missing))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geometry-first continuous oblique imagery POC v0.1")
    parser.add_argument("project_root", type=Path, help="Project root, e.g. D:\\Sev\\Obliques\\NYPA")
    parser.add_argument("--circuit", default=None, help="Circuit label. Default: infer from existing 163/staking filename")
    parser.add_argument("--epsg", type=int, default=None, help="Override working EPSG if PLS zone cannot be mapped")
    parser.add_argument("--reference-mode", choices=["span", "structure"], default="span")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <root>\\_continuous_oblique_poc_v01")
    parser.add_argument("--max-eo-distance", type=float, default=None, help="Max camera-to-alignment distance in working units")
    parser.add_argument("--gap-tolerance", type=float, default=None, help="Ignore gaps smaller than this, working units")
    parser.add_argument("--min-overlap", type=float, default=None, help="Transition overlap below this is WEAK_OVERLAP")
    parser.add_argument("--structure-tolerance", type=float, default=None, help="Buffer around structure when testing footprint visibility")
    parser.add_argument("--allow-side", action="store_true", help="Allow SIDE images as product candidates")
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    if not root.exists():
        parser.error(f"Project root does not exist: {root}")

    paths = discover_project(root)
    validate_required_paths(paths)

    out_dir = args.output.resolve() if args.output else root / "_continuous_oblique_poc_v01"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "continuous_oblique_poc.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    print("[1/8] Parsing staking and building alignment...")
    structures, spans, alignment_gdf, metadata = parse_staking_xml(paths.staking_xml, args.epsg)
    alignment = alignment_gdf.iloc[0].geometry
    crs = metadata["crs"]
    defaults = unit_defaults(crs)
    max_eo_distance = args.max_eo_distance if args.max_eo_distance is not None else defaults["max_eo_distance"]
    gap_tolerance = args.gap_tolerance if args.gap_tolerance is not None else defaults["gap_tolerance"]
    min_overlap = args.min_overlap if args.min_overlap is not None else defaults["min_overlap"]
    structure_tolerance = args.structure_tolerance if args.structure_tolerance is not None else defaults["structure_tolerance"]

    print("[2/8] Indexing raw images and existing 163...")
    raw = index_raw_images(paths.raw_dir)
    existing = index_existing_163(paths.final_163_dir)
    circuit = args.circuit or infer_circuit(existing, paths.staking_xml)

    print("[3/8] Parsing image footprints...")
    footprints = load_footprints(paths.footprint_dxfs, crs)

    print("[4/8] Parsing EO and detecting flight passes...")
    eo = load_all_eo(paths.eo_dir)
    if eo.empty:
        raise ValueError("No EO records parsed")
    eo = project_eo_to_alignment(eo, alignment)
    eo, eo_summary = eo_file_compatibility(eo, max_eo_distance)
    eo, passes_df = classify_passes(eo)
    passes_gdf = gpd.GeoDataFrame(passes_df, geometry="geometry", crs=crs) if not passes_df.empty else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)

    print("[5/8] Building image coverage / attribution model...")
    images_df, images_gdf, coverage_gdf = build_image_model(
        raw, eo, footprints, structures, spans, alignment, crs, args.reference_mode, args.allow_side
    )

    print("[6/8] Building structure observations...")
    structure_obs = build_structure_observations(
        images_df, footprints, structures, alignment, crs, structure_tolerance
    )

    print("[7/8] Checking transitions and coverage gaps...")
    transitions = build_transitions(images_df, crs, gap_tolerance, min_overlap)
    gaps = build_pass_gaps(images_df, alignment, crs, gap_tolerance)

    print("[8/8] Writing GeoPackage and CSV QA outputs...")
    layers = [
        (structures, "structures"),
        (spans, "spans"),
        (alignment_gdf, "alignment"),
        (footprints, "image_footprints"),
        (passes_gdf, "flight_passes"),
        (images_gdf, "images"),
        (coverage_gdf, "image_coverage"),
        (structure_obs, "structure_observations"),
        (transitions, "transitions"),
        (gaps, "coverage_gaps"),
    ]
    first = True
    for gdf, layer in layers:
        if gdf is not None and not gdf.empty:
            write_gpkg_layer(gdf, gpkg, layer, first=first)
            first = False

    safe_csv(raw, out_dir / "raw_images.csv")
    safe_csv(eo, out_dir / "eo_index.csv")
    safe_csv(eo_summary, out_dir / "eo_sources.csv")
    safe_csv(images_df, out_dir / "image_index.csv")
    safe_csv(passes_df, out_dir / "flight_passes.csv")
    safe_csv(transitions, out_dir / "transitions.csv")
    safe_csv(gaps, out_dir / "coverage_gaps.csv")
    safe_csv(structure_obs, out_dir / "structure_observations.csv")
    if not existing.empty:
        safe_csv(existing, out_dir / "existing_163_index.csv")

    summary = build_summary(
        paths, metadata, raw, footprints, eo, eo_summary, structures, spans,
        passes_df, images_df, transitions, gaps, existing, circuit,
    )
    summary["settings"] = {
        "reference_mode": args.reference_mode,
        "allow_side": args.allow_side,
        "max_eo_distance": max_eo_distance,
        "gap_tolerance": gap_tolerance,
        "min_overlap": min_overlap,
        "structure_tolerance": structure_tolerance,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Circuit:              {circuit}")
    print(f"CRS:                  EPSG:{metadata['epsg']}")
    print(f"Structures:           {len(structures)}")
    print(f"Spans:                {len(spans)}")
    print(f"Raw images:           {len(raw)}")
    print(f"Footprints:           {len(footprints)}")
    print(f"EO records:           {len(eo)}")
    print(f"Detected passes:      {len(passes_df)}")
    print(f"Valid candidates:     {int(images_df['valid_for_product'].sum()) if len(images_df) else 0}")
    print(f"Existing 163 images:  {len(existing)}")
    print(f"Output:               {out_dir}")

    if not eo_summary.empty:
        print("\nEO source compatibility:")
        for _, r in eo_summary.iterrows():
            print(
                f"  {r['eo_source']}: records={int(r['records'])}, "
                f"min_distance={r['min_distance_to_alignment']:.1f}, "
                f"compatible={bool(r['spatial_compatible'])}"
            )

    if not passes_df.empty:
        print("\nPasses:")
        for _, p in passes_df.iterrows():
            pid = p["pass_id"]
            n_valid = int(((images_df["pass_id"] == pid) & images_df["valid_for_product"]).sum())
            n_gaps = int((gaps["pass_id"] == pid).sum()) if not gaps.empty else 0
            print(f"  {pid}: {p['pass_direction']}, images={int(p['image_count'])}, valid={n_valid}, hard_gaps={n_gaps}")

    print("\nNo source imagery was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
