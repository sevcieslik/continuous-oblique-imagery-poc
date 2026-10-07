#!/usr/bin/env python
"""
Continuous Oblique Imagery POC v0.7

Purpose
-------
Build a geometry-first continuous-oblique QA model from an NM-style project root.
This version deliberately does NOT use DINOv3 and does NOT alter/copy/rename source images.

Implemented through v0.7
-------------------
- Auto-discovery of NM product folders and highest vN folders
- PLS-CADD staking XML -> physical structures/spans, alignment, chainage
- Reference-topology reconstruction from PLS .tow ordinals (separate product spans from intermediate PLS rows)
- DXF image footprints -> image polygons keyed by filename
- EO TXT -> camera positions/orientations keyed by filename
- Raw image index
- Existing 163 naming index for benchmark/reference only
- Flight-pass segmentation from EO trajectory
- FORWARD_PASS / BACKWARD_PASS / CROSSING_PASS classification
- Per-image confirmed/provisional alignment coverage and primary span / primary structure
- Approximate optical-axis target from EO Omega/Phi/Kappa + staking elevation
- Optical primary-span attribution with geometry fallback
- Recovery of missing footprint coverage from EO trajectory + neighbouring confirmed footprints
- Gap classification: CAPTURE_GAP / METADATA_GAP / SOURCE_DATA_GAP / SEQUENCE_GAP
- Per-image geometric FORWARD / BACKWARD / SIDE view classification
- Structure observations for every footprint that contains a structure
- Pairwise transition QA at raw flight-pass level
- Circuit-run stitching across interrupted flight-pass fragments
- Product-level transitions ordered along circuit chainage
- Internal gap detection per circuit run (no false start/end gaps per fragment)
- Full-circuit coverage and gaps independently for FORWARD and BACKWARD directions
- Aggregate span benchmark against an existing 163 deliverable
- Optional appearance-based raw-to-163 matching for image-level span benchmark
- Span-completeness QA: require evidence of two consecutive reference structure groups before final 163 inclusion
- Separate sequence-valid imagery from final-163-valid imagery; rejected span views remain available for continuity QA
- GeoPackage + CSV outputs

Not implemented yet
-------------------
- LiDAR structure-position refinement
- Exact camera projection using camera calibration (v0.4 uses an approximate OPK optical axis)
- Calibrated structure crops (v0.5/v0.6 crops remain approximate)
- FLI matching (DINOv3 continuity is implemented in v0.7)

Example
-------
python continuous_oblique_v07.py "D:\\Sev\\Obliques\\NYPA" --circuit DH1

Dependencies
------------
pandas, numpy, shapely>=2, geopandas, pyproj, ezdxf, pyogrio, Pillow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
        "  python -m pip install pandas numpy shapely geopandas pyproj ezdxf pyogrio Pillow"
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
            "optical_max_distance": 500.0,
        }
    return {
        "max_eo_distance": 6000.0,
        "gap_tolerance": 2.0,
        "min_overlap": 15.0,
        "structure_tolerance": 3.0,
        "optical_max_distance": 150.0,
    }


# -----------------------------------------------------------------------------
# Staking / alignment
# -----------------------------------------------------------------------------


def _extract_tow_ordinal(structure_name: str | None) -> float:
    """Return the last #N token from the PLS .tow path, if present.

    PLS projects can contain auxiliary/intermediate staking rows that reuse an
    ordinal already used by the main circuit.  The last #N token is a useful
    candidate for the product/reference structure sequence.  It is treated as
    metadata, not blindly trusted: the topology builder validates monotonicity
    before using it.
    """
    if not structure_name:
        return np.nan
    vals = re.findall(r"#(\d+)", str(structure_name))
    return float(vals[-1]) if vals else np.nan


def _build_reference_topology(
    structures: gpd.GeoDataFrame,
    alignment: LineString,
    crs: CRS,
    mode: str = "auto",
):
    """Build product/reference structures and spans separately from raw PLS rows.

    Modes
    -----
    identity:
        Every PLS staking row is a product/reference structure.
    tow-ordinal:
        Use the last #N token from structure_name as the reference ordinal.
        When the same ordinal occurs multiple times, the last occurrence along
        the PLS row order is retained as canonical.  Earlier duplicates remain
        in the physical topology but are marked as intermediate rows.
    auto:
        Use tow-ordinal only when the resulting canonical sequence is strongly
        supported: >=80% rows have an ordinal, canonical chainage is monotonic,
        and reference ordinals are contiguous.  Otherwise fall back to identity.
    """
    out = structures.copy()
    if "structure_name" not in out.columns:
        out["structure_name"] = None
    out["tow_ordinal_candidate"] = out["structure_name"].map(_extract_tow_ordinal)
    out["reference_idx"] = np.nan
    out["is_reference_structure"] = False
    out["reference_selection_reason"] = ""
    out["reference_duplicate_count"] = 0

    def identity_topology(status: str):
        x = out.copy()
        x["reference_idx"] = x["structure_idx"].astype(int)
        x["is_reference_structure"] = True
        x["reference_selection_reason"] = "IDENTITY"
        x["reference_duplicate_count"] = 1
        return x, status

    requested = mode.lower()
    if requested not in {"auto", "tow-ordinal", "identity"}:
        raise ValueError(f"Unsupported reference topology mode: {mode}")
    if requested == "identity":
        out, status = identity_topology("IDENTITY_FORCED")
    else:
        candidates = out[out["tow_ordinal_candidate"].notna()].copy()
        candidate_fraction = len(candidates) / max(len(out), 1)
        if candidates.empty:
            if requested == "tow-ordinal":
                raise ValueError("No #N ordinal tokens found in PLS structure_name fields")
            out, status = identity_topology("IDENTITY_FALLBACK_NO_TOW_ORDINALS")
        else:
            candidates["tow_ordinal_candidate"] = candidates["tow_ordinal_candidate"].astype(int)
            counts = candidates.groupby("tow_ordinal_candidate").size().to_dict()
            # Last occurrence is deliberate: in DH1 the auxiliary station rows
            # claim #116/#117 before the main circuit resumes and reuses those
            # ordinals.  Retaining the later occurrence reconstructs the
            # customer/product structure sequence without removing physical rows.
            canonical_rows = (
                candidates.sort_values("rownum")
                .groupby("tow_ordinal_candidate", as_index=False)
                .tail(1)
                .sort_values("tow_ordinal_candidate")
                .copy()
            )
            ords = canonical_rows["tow_ordinal_candidate"].astype(int).to_numpy()
            ch = canonical_rows["chainage"].astype(float).to_numpy()
            contiguous = bool(len(ords) >= 2 and np.all(np.diff(ords) == 1))
            monotonic = bool(len(ch) >= 2 and np.all(np.diff(ch) > 0))
            auto_ok = candidate_fraction >= 0.80 and contiguous and monotonic
            if requested == "auto" and not auto_ok:
                out, status = identity_topology(
                    f"IDENTITY_FALLBACK candidate_fraction={candidate_fraction:.3f} "
                    f"contiguous={contiguous} monotonic={monotonic}"
                )
            else:
                if requested == "tow-ordinal" and not (contiguous and monotonic):
                    raise ValueError(
                        "tow-ordinal reference topology is not a contiguous monotonic sequence; "
                        "use --reference-topology identity or inspect reference_topology.csv"
                    )
                # Mark candidate metadata on every physical staking row.
                for idx, r in out.iterrows():
                    cand = r.get("tow_ordinal_candidate")
                    if pd.notna(cand):
                        ci = int(cand)
                        out.at[idx, "reference_idx"] = ci
                        out.at[idx, "reference_duplicate_count"] = int(counts.get(ci, 1))
                canonical_rownums = set(canonical_rows["rownum"].astype(int))
                for idx, r in out.iterrows():
                    cand = r.get("tow_ordinal_candidate")
                    if pd.isna(cand):
                        out.at[idx, "reference_selection_reason"] = "NO_TOW_ORDINAL"
                    elif int(r["rownum"]) in canonical_rownums:
                        out.at[idx, "is_reference_structure"] = True
                        dup = int(out.at[idx, "reference_duplicate_count"])
                        out.at[idx, "reference_selection_reason"] = (
                            "UNIQUE_TOW_ORDINAL" if dup == 1 else "DUPLICATE_LAST_OCCURRENCE"
                        )
                    else:
                        out.at[idx, "reference_selection_reason"] = "INTERMEDIATE_DUPLICATE_TOW_ORDINAL"
                status = (
                    f"TOW_ORDINAL candidate_fraction={candidate_fraction:.3f} "
                    f"reference_structures={len(canonical_rows)} "
                    f"intermediate_rows={int((~out['is_reference_structure']).sum())}"
                )

    refs = out[out["is_reference_structure"].eq(True)].copy()
    refs = refs.sort_values(["reference_idx", "chainage"]).reset_index(drop=True)
    if len(refs) < 2:
        raise ValueError("Reference topology contains fewer than two structures")

    span_rows = []
    for i in range(len(refs) - 1):
        a = refs.iloc[i]
        b = refs.iloc[i + 1]
        ai = int(a.reference_idx)
        bi = int(b.reference_idx)
        ch0 = float(a.chainage)
        ch1 = float(b.chainage)
        geom = substring(alignment, ch0, ch1)
        if isinstance(geom, Point):
            geom = LineString([a.geometry, b.geometry])
        intermediate = out[
            (out["chainage"] > ch0)
            & (out["chainage"] < ch1)
            & (~out["is_reference_structure"].eq(True))
        ]
        span_rows.append(
            {
                "span_id": f"{ai}-{bi}",
                "from_idx": ai,
                "to_idx": bi,
                "from_structure": a.structure_id,
                "to_structure": b.structure_id,
                "from_pls_structure_idx": int(a.structure_idx),
                "to_pls_structure_idx": int(b.structure_idx),
                "chainage_from": ch0,
                "chainage_to": ch1,
                "geom_length": float(geom.length),
                "pls_station_from": float(a.station) if pd.notna(a.station) else np.nan,
                "pls_station_to": float(b.station) if pd.notna(b.station) else np.nan,
                "intermediate_pls_rows": int(len(intermediate)),
                "intermediate_structure_ids": ";".join(intermediate["structure_id"].astype(str)),
                "geometry": geom,
            }
        )
    reference_spans = gpd.GeoDataFrame(span_rows, geometry="geometry", crs=crs)
    return out, refs, reference_spans, status


def parse_staking_xml(
    path: Path,
    epsg_override: int | None = None,
    reference_topology_mode: str = "auto",
):
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
            "structure_name": get("structure_name"),
            "structure_description": get("structure_description"),
            "structure_type": get("structure_comment_2"),
        }
        if math.isfinite(rec["x"]) and math.isfinite(rec["y"]):
            rows.append(rec)

    if len(rows) < 2:
        raise ValueError("Fewer than two coordinate-bearing staking rows found")

    df = pd.DataFrame(rows).sort_values("rownum").reset_index(drop=True)
    if pd.isna(df.loc[0, "station"]):
        df.loc[0, "station"] = 0.0

    points = [Point(xy) for xy in zip(df["x"], df["y"])]
    alignment = LineString([(p.x, p.y) for p in points])
    df["chainage"] = [alignment.project(p) for p in points]
    structures = gpd.GeoDataFrame(df, geometry=points, crs=crs)

    # Physical PLS spans are retained as QA topology.  Product labels are built
    # separately below from canonical reference structures.
    pls_span_rows: list[dict] = []
    for i in range(len(structures) - 1):
        a = structures.iloc[i]
        b = structures.iloc[i + 1]
        geom = LineString([a.geometry, b.geometry])
        pls_span_rows.append(
            {
                "pls_span_id": f"{int(a.structure_idx)}-{int(b.structure_idx)}",
                "from_pls_idx": int(a.structure_idx),
                "to_pls_idx": int(b.structure_idx),
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
    pls_spans = gpd.GeoDataFrame(pls_span_rows, geometry="geometry", crs=crs)

    structures, reference_structures, reference_spans, topology_status = _build_reference_topology(
        structures, alignment, crs, reference_topology_mode
    )

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
        "reference_topology_mode_requested": reference_topology_mode,
        "reference_topology_status": topology_status,
        "reference_structure_count": int(len(reference_structures)),
        "reference_span_count": int(len(reference_spans)),
        "intermediate_pls_structure_count": int((~structures["is_reference_structure"].eq(True)).sum()),
    }
    return structures, pls_spans, reference_structures, reference_spans, alignment_gdf, metadata

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
# Approximate optical-axis projection from EO OPK
# -----------------------------------------------------------------------------


def elevation_at_chainage(structures: gpd.GeoDataFrame, chainage: float) -> float:
    """Linear interpolation of PLS centreline Z along the ordered circuit."""
    if not math.isfinite(chainage) or structures.empty:
        return float("nan")
    ch = pd.to_numeric(structures["chainage"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(structures["centerline_z"], errors="coerce").to_numpy(float)
    mask = np.isfinite(ch) & np.isfinite(z)
    if not mask.any():
        return float("nan")
    order = np.argsort(ch[mask])
    chv = ch[mask][order]
    zv = z[mask][order]
    return float(np.interp(float(chainage), chv, zv))


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def opk_optical_axis(
    omega_deg: float,
    phi_deg: float,
    kappa_deg: float,
    convention: str = "STANDARD_KPO",
) -> np.ndarray | None:
    """
    Approximate ground-frame camera optical axis.

    v0.3 deliberately keeps this separate from a calibrated camera model.  The
    default uses the common photogrammetric K-P-O composition and the camera -Z
    axis.  It is good enough to test whether optical targeting improves the
    span/reference attribution, while all diagnostics remain exposed for QA.
    """
    if not all(math.isfinite(v) for v in [omega_deg, phi_deg, kappa_deg]):
        return None
    o, p, k = map(math.radians, [omega_deg, phi_deg, kappa_deg])
    if convention == "STANDARD_KPO":
        r = _rz(k) @ _ry(p) @ _rx(o)
    elif convention == "FLIP_PHI_KPO":
        r = _rz(k) @ _ry(-p) @ _rx(o)
    else:
        raise ValueError(f"Unsupported OPK convention: {convention}")
    v = r @ np.array([0.0, 0.0, -1.0], dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def project_optical_target(
    camera_x: float,
    camera_y: float,
    camera_z: float,
    omega: float,
    phi: float,
    kappa: float,
    camera_chainage: float,
    structures: gpd.GeoDataFrame,
    alignment: LineString,
    convention: str = "STANDARD_KPO",
    iterations: int = 4,
) -> dict:
    """Project the approximate camera optical axis onto a locally varying ground-Z plane."""
    empty = {
        "optical_target_x": np.nan,
        "optical_target_y": np.nan,
        "optical_target_z": np.nan,
        "optical_target_chainage": np.nan,
        "optical_target_distance": np.nan,
        "optical_axis_x": np.nan,
        "optical_axis_y": np.nan,
        "optical_axis_z": np.nan,
        "optical_projection_status": "UNAVAILABLE",
    }
    vals = [camera_x, camera_y, camera_z, omega, phi, kappa, camera_chainage]
    if not all(pd.notna(v) and math.isfinite(float(v)) for v in vals):
        return empty

    axis = opk_optical_axis(float(omega), float(phi), float(kappa), convention)
    if axis is None:
        return empty
    out = dict(empty)
    out.update({
        "optical_axis_x": float(axis[0]),
        "optical_axis_y": float(axis[1]),
        "optical_axis_z": float(axis[2]),
    })
    if axis[2] >= -1e-8:
        out["optical_projection_status"] = "AXIS_NOT_DOWNWARD"
        return out

    ch = float(np.clip(float(camera_chainage), 0.0, float(alignment.length)))
    ground_z = elevation_at_chainage(structures, ch)
    if not math.isfinite(ground_z):
        out["optical_projection_status"] = "NO_GROUND_Z"
        return out

    target = None
    for _ in range(max(1, int(iterations))):
        t = (ground_z - float(camera_z)) / float(axis[2])
        if not math.isfinite(t) or t <= 0:
            out["optical_projection_status"] = "NO_GROUND_INTERSECTION"
            return out
        target = Point(
            float(camera_x) + t * float(axis[0]),
            float(camera_y) + t * float(axis[1]),
        )
        ch = float(alignment.project(target))
        ground_z = elevation_at_chainage(structures, ch)
        if not math.isfinite(ground_z):
            out["optical_projection_status"] = "NO_GROUND_Z"
            return out

    out.update({
        "optical_target_x": float(target.x),
        "optical_target_y": float(target.y),
        "optical_target_z": float(ground_z),
        "optical_target_chainage": ch,
        "optical_target_distance": float(target.distance(alignment)),
        "optical_projection_status": "OK",
    })
    return out


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


def intervals_gap_segment(
    a: Sequence[tuple[float, float]],
    b: Sequence[tuple[float, float]],
) -> tuple[float, float] | tuple[None, None]:
    """Return the closest uncovered chainage segment between two interval sets."""
    if not a or not b or intervals_overlap_length(a, b) > 0:
        return (None, None)
    best = None
    for a0, a1 in a:
        for b0, b1 in b:
            if a1 < b0:
                seg = (float(a1), float(b0))
            elif b1 < a0:
                seg = (float(b1), float(a0))
            else:
                return (None, None)
            d = seg[1] - seg[0]
            if best is None or d < best[0]:
                best = (d, seg[0], seg[1])
    return (best[1], best[2]) if best is not None else (None, None)


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
    candidates = structures
    if "is_reference_structure" in structures.columns and structures["is_reference_structure"].eq(True).any():
        candidates = structures[structures["is_reference_structure"].eq(True)]
    idx = (candidates["chainage"].astype(float) - chainage).abs().idxmin()
    s = candidates.loc[idx]
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
    optical_max_distance: float,
    opk_convention: str = "STANDARD_KPO",
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
        geometry_centre = (
            (min(a for a, _ in intervals) + max(b for _, b in intervals)) / 2.0
            if intervals else np.nan
        )
        geometry_span = primary_span_for_chainage(spans, geometry_centre) if math.isfinite(geometry_centre) else None

        optical = project_optical_target(
            r.get("camera_x"), r.get("camera_y"), r.get("camera_z"),
            r.get("omega"), r.get("phi"), r.get("kappa"), r.get("camera_chainage"),
            structures, alignment, convention=opk_convention,
        )
        optical_ch = optical.get("optical_target_chainage", np.nan)
        optical_dist = optical.get("optical_target_distance", np.nan)
        optical_accepted = (
            optical.get("optical_projection_status") == "OK"
            and pd.notna(optical_ch) and math.isfinite(float(optical_ch))
            and pd.notna(optical_dist) and math.isfinite(float(optical_dist))
            and float(optical_dist) <= float(optical_max_distance)
        )
        optical_span = primary_span_for_chainage(spans, float(optical_ch)) if optical_accepted else None
        if optical.get("optical_projection_status") != "OK":
            optical_confidence = "UNAVAILABLE"
        elif math.isfinite(float(optical_dist)) and float(optical_dist) <= 0.4 * float(optical_max_distance):
            optical_confidence = "HIGH"
        elif math.isfinite(float(optical_dist)) and float(optical_dist) <= float(optical_max_distance):
            optical_confidence = "MEDIUM"
        else:
            optical_confidence = "LOW"

        if optical_span is not None:
            primary_span = optical_span
            primary_span_method = "OPK_OPTICAL"
            reference_chainage = float(optical_ch)
        else:
            primary_span = geometry_span
            primary_span_method = "COVERAGE_MIDPOINT" if geometry_span is not None else "NONE"
            reference_chainage = float(geometry_centre) if math.isfinite(geometry_centre) else np.nan

        pstruct, pstruct_idx = (
            primary_structure_for_chainage(structures, reference_chainage)
            if math.isfinite(reference_chainage) else (None, None)
        )
        visible_spans = spans_for_intervals(spans, intervals) if intervals else []

        view = "UNKNOWN"
        view_angle = np.nan
        along_offset = np.nan
        cross_offset = np.nan
        if math.isfinite(reference_chainage) and pd.notna(r.get("camera_x")) and pd.notna(r.get("camera_y")):
            camera = Point(float(r.camera_x), float(r.camera_y))
            if reference_mode == "structure" and pstruct_idx is not None:
                target_row = structures[structures["structure_idx"] == pstruct_idx].iloc[0]
                target = target_row.geometry
                target_chainage = float(target_row.chainage)
            else:
                target = alignment.interpolate(float(reference_chainage))
                target_chainage = float(reference_chainage)
            tangent = local_tangent(alignment, target_chainage)
            view, view_angle, along_offset, cross_offset = classify_camera_view(camera, target, tangent)

        pass_direction = r.get("pass_direction")
        progression_consistent = None
        if optical_accepted and pd.notna(r.get("camera_chainage")):
            delta = float(optical_ch) - float(r.get("camera_chainage"))
            if pass_direction == "FORWARD_PASS":
                progression_consistent = bool(delta >= 0)
            elif pass_direction == "BACKWARD_PASS":
                progression_consistent = bool(delta <= 0)

        reason = None
        valid = True
        if not intervals:
            valid = False
            reason = "MISSING_FOOTPRINT" if pd.isna(r.get("dxf_source")) else "NO_ALIGNMENT_COVERAGE"
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
        rec.update(optical)
        rec.update(
            {
                "opk_convention": opk_convention,
                "optical_target_confidence": optical_confidence,
                "optical_target_accepted": bool(optical_accepted),
                "optical_progression_consistent": progression_consistent,
                "coverage_intervals": interval_string(intervals),
                "coverage_from": min((a for a, _ in intervals), default=np.nan),
                "coverage_to": max((b for _, b in intervals), default=np.nan),
                "coverage_length": cov_len,
                "coverage_midpoint_chainage": geometry_centre,
                "centre_chainage": geometry_centre,
                "coverage_source": "FOOTPRINT" if intervals else "NONE",
                "coverage_status": ("CONFIRMED" if intervals else ("MISSING_METADATA" if pd.isna(r.get("dxf_source")) else "NO_INTERSECTION")),
                "coverage_confidence": "HIGH" if intervals else "NONE",
                "visible_spans": ";".join(visible_spans),
                "primary_span_geometry": geometry_span,
                "primary_span_optical": optical_span,
                "primary_span": primary_span,
                "primary_span_method": primary_span_method,
                "reference_chainage": reference_chainage,
                "primary_structure": pstruct,
                "primary_structure_idx": pstruct_idx,
                "reference_mode": reference_mode,
                "reference_id": reference_id,
                "view_direction": view,
                "view_angle_deg": view_angle,
                "along_track_offset": along_offset,
                "cross_track_offset": cross_offset,
                "valid_for_product": bool(valid),
                "product_status": "CONFIRMED" if valid else "REJECTED",
                "rejection_reason": reason,
            }
        )
        records.append(rec)
        coverage_geoms.append(coverage_geom)

    images_df = pd.DataFrame(records)

    spatial_mask = (
        images_df["camera_x"].notna()
        & images_df["camera_y"].notna()
        & images_df.get("spatial_compatible", pd.Series(False, index=images_df.index)).eq(True)
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
                "primary_span_method": rec.get("primary_span_method"),
                "reference_id": rec.get("reference_id"),
                "coverage_source": rec.get("coverage_source"),
                "coverage_status": rec.get("coverage_status"),
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
                    "circuit_run_id": img.get("circuit_run_id") if img else None,
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
# Missing-footprint recovery
# -----------------------------------------------------------------------------


def recover_missing_footprint_coverage(
    images_df: pd.DataFrame,
    spans: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    alignment: LineString,
    reference_mode: str,
    allow_side: bool,
    recovery_max_neighbor_gap: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recover provisional 1D circuit coverage when a raw image has valid EO but its
    DXF footprint is missing.

    The recovery is deliberately conservative:
    - only FORWARD/BACKWARD passes are eligible;
    - at least two confirmed footprint observations must exist in the pass;
    - interpolation is preferred; short-range pass-offset extrapolation is fallback;
    - recovered coverage is always PROVISIONAL, never CONFIRMED.
    """
    out = images_df.copy()
    recovery_rows: list[dict] = []

    if out.empty or "pass_id" not in out.columns:
        return out, pd.DataFrame()

    for pass_id, grp0 in out[out["pass_id"].notna()].groupby("pass_id", sort=False):
        grp = grp0.sort_values("eo_order")
        directions = grp["pass_direction"].dropna()
        if directions.empty:
            continue
        direction = str(directions.iloc[0])
        if direction not in {"FORWARD_PASS", "BACKWARD_PASS"}:
            continue

        confirmed = grp[
            grp["coverage_status"].eq("CONFIRMED")
            & pd.to_numeric(grp["centre_chainage"], errors="coerce").notna()
            & pd.to_numeric(grp["coverage_length"], errors="coerce").gt(0)
        ].copy().sort_values("eo_order")
        if len(confirmed) < 2:
            continue

        missing = grp[
            grp["coverage_status"].eq("MISSING_METADATA")
            & grp["eo_source"].notna()
            & grp.get("spatial_compatible", pd.Series(False, index=grp.index)).eq(True)
        ].copy()

        for idx, r in missing.iterrows():
            if pd.isna(r.get("eo_order")) or pd.isna(r.get("camera_chainage")):
                continue
            order = float(r["eo_order"])
            prev = confirmed[pd.to_numeric(confirmed["eo_order"], errors="coerce") < order].tail(1)
            nxt = confirmed[pd.to_numeric(confirmed["eo_order"], errors="coerce") > order].head(1)

            centre = np.nan
            length = np.nan
            method = None
            confidence = "LOW"
            prev_name = None
            next_name = None
            neighbour_span = np.nan

            if len(prev) and len(nxt):
                p = prev.iloc[0]
                n = nxt.iloc[0]
                op = float(p["eo_order"])
                on = float(n["eo_order"])
                total_order_gap = on - op
                prev_name = p.get("filename")
                next_name = n.get("filename")
                neighbour_span = total_order_gap
                # Includes the endpoints.  max=10 therefore permits up to nine
                # consecutive missing frames between confirmed neighbours.
                if total_order_gap <= float(recovery_max_neighbor_gap + 1) and total_order_gap > 0:
                    t = (order - op) / total_order_gap
                    centre = float(p["centre_chainage"]) + t * (float(n["centre_chainage"]) - float(p["centre_chainage"]))
                    length = float(p["coverage_length"]) + t * (float(n["coverage_length"]) - float(p["coverage_length"]))
                    method = "NEIGHBOUR_INTERPOLATION"
                    confidence = "HIGH" if total_order_gap <= 5 else "MEDIUM"

            if not math.isfinite(centre) or not math.isfinite(length) or length <= 0:
                local = confirmed.copy()
                local["_order_dist"] = (pd.to_numeric(local["eo_order"], errors="coerce") - order).abs()
                local = local.sort_values("_order_dist").head(min(5, len(local)))
                if len(local):
                    nearest_dist = float(local["_order_dist"].min())
                    if nearest_dist <= float(recovery_max_neighbor_gap):
                        offsets = (
                            pd.to_numeric(local["centre_chainage"], errors="coerce")
                            - pd.to_numeric(local["camera_chainage"], errors="coerce")
                        ).dropna()
                        lengths = pd.to_numeric(local["coverage_length"], errors="coerce").dropna()
                        if len(offsets) and len(lengths):
                            centre = float(r["camera_chainage"]) + float(offsets.median())
                            length = float(lengths.median())
                            method = "PASS_OFFSET_EXTRAPOLATION"
                            confidence = "MEDIUM" if nearest_dist <= 4 else "LOW"
                            nearest = local.iloc[0]
                            prev_name = nearest.get("filename") if float(nearest["eo_order"]) < order else None
                            next_name = nearest.get("filename") if float(nearest["eo_order"]) > order else None
                            neighbour_span = nearest_dist

            if not math.isfinite(centre) or not math.isfinite(length) or length <= 0:
                # Strong evidence that metadata rather than capture may be the issue.
                if pd.isna(r.get("dxf_source")):
                    out.at[idx, "coverage_status"] = "MISSING_METADATA"
                    out.at[idx, "coverage_source"] = "NONE"
                    if out.at[idx, "rejection_reason"] == "NO_ALIGNMENT_COVERAGE":
                        out.at[idx, "rejection_reason"] = "MISSING_FOOTPRINT"
                continue

            a = max(0.0, float(centre) - float(length) / 2.0)
            b = min(float(alignment.length), float(centre) + float(length) / 2.0)
            if b <= a:
                continue
            centre = (a + b) / 2.0
            intervals = [(a, b)]
            geometry_span = primary_span_for_chainage(spans, centre)
            visible_spans = spans_for_intervals(spans, intervals)

            out.at[idx, "coverage_intervals"] = interval_string(intervals)
            out.at[idx, "coverage_from"] = a
            out.at[idx, "coverage_to"] = b
            out.at[idx, "coverage_length"] = b - a
            out.at[idx, "coverage_midpoint_chainage"] = centre
            out.at[idx, "centre_chainage"] = centre
            out.at[idx, "coverage_source"] = method
            out.at[idx, "coverage_status"] = "PROVISIONAL"
            out.at[idx, "coverage_confidence"] = confidence
            out.at[idx, "visible_spans"] = ";".join(visible_spans)
            out.at[idx, "primary_span_geometry"] = geometry_span

            # Optical target remains the preferred reference if it passed the
            # corridor-distance QA.  Recovery only supplies the missing coverage.
            if pd.notna(r.get("primary_span_optical")) and bool(r.get("optical_target_accepted")):
                primary_span = r.get("primary_span_optical")
                reference_chainage = float(r.get("optical_target_chainage"))
                method_span = "OPK_OPTICAL"
            else:
                primary_span = geometry_span
                reference_chainage = centre
                method_span = "PROVISIONAL_COVERAGE_MIDPOINT"
            out.at[idx, "primary_span"] = primary_span
            out.at[idx, "primary_span_method"] = method_span
            out.at[idx, "reference_chainage"] = reference_chainage

            pstruct, pstruct_idx = primary_structure_for_chainage(structures, reference_chainage)
            out.at[idx, "primary_structure"] = pstruct
            out.at[idx, "primary_structure_idx"] = pstruct_idx
            out.at[idx, "reference_id"] = primary_span if reference_mode == "span" else pstruct

            view = str(r.get("view_direction")) if pd.notna(r.get("view_direction")) else "UNKNOWN"
            if view == "UNKNOWN" and pd.notna(r.get("camera_x")) and pd.notna(r.get("camera_y")):
                camera = Point(float(r["camera_x"]), float(r["camera_y"]))
                target = alignment.interpolate(reference_chainage)
                tangent = local_tangent(alignment, reference_chainage)
                view, angle, along, cross = classify_camera_view(camera, target, tangent)
                out.at[idx, "view_direction"] = view
                out.at[idx, "view_angle_deg"] = angle
                out.at[idx, "along_track_offset"] = along
                out.at[idx, "cross_track_offset"] = cross

            reason = None
            valid = True
            if direction not in {"FORWARD_PASS", "BACKWARD_PASS"}:
                valid = False
                reason = direction
            elif view == "SIDE" and not allow_side:
                valid = False
                reason = "SIDE_VIEW"
            out.at[idx, "valid_for_product"] = bool(valid)
            out.at[idx, "product_status"] = "PROVISIONAL" if valid else "REJECTED"
            out.at[idx, "rejection_reason"] = reason

            recovery_rows.append(
                {
                    "filename": r.get("filename"),
                    "pass_id": pass_id,
                    "pass_direction": direction,
                    "recovery_method": method,
                    "recovery_confidence": confidence,
                    "eo_order": order,
                    "previous_confirmed_image": prev_name,
                    "next_confirmed_image": next_name,
                    "neighbour_order_span": neighbour_span,
                    "recovered_from": a,
                    "recovered_to": b,
                    "recovered_length": b - a,
                    "primary_span": primary_span,
                    "primary_span_method": method_span,
                }
            )

    recovery = pd.DataFrame(recovery_rows)
    return out, recovery


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
# Circuit runs / product-level continuity / benchmark
# -----------------------------------------------------------------------------


def intervals_to_geometry(alignment: LineString, intervals: Sequence[tuple[float, float]]):
    """Convert chainage intervals to an alignment geometry (LineString/MultiLineString)."""
    parts = []
    for a, b in intervals:
        a = max(0.0, float(a))
        b = min(float(alignment.length), float(b))
        if b - a <= 1e-9:
            continue
        g = substring(alignment, a, b)
        if g is not None and not g.is_empty:
            parts.append(g)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    lines = []
    for g in parts:
        if isinstance(g, LineString):
            lines.append(g)
        elif isinstance(g, MultiLineString):
            lines.extend(list(g.geoms))
    return MultiLineString(lines) if lines else None



def build_images_layer(images_df: pd.DataFrame, crs: CRS) -> gpd.GeoDataFrame:
    if images_df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    mask = (
        images_df.get("camera_x", pd.Series(np.nan, index=images_df.index)).notna()
        & images_df.get("camera_y", pd.Series(np.nan, index=images_df.index)).notna()
        & images_df.get("spatial_compatible", pd.Series(False, index=images_df.index)).eq(True)
    )
    out = images_df.loc[mask].copy()
    if out.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    geom = [Point(float(x), float(y)) for x, y in zip(out["camera_x"], out["camera_y"])]
    return gpd.GeoDataFrame(out, geometry=geom, crs=crs)


def build_coverage_layer(images_df: pd.DataFrame, alignment: LineString, crs: CRS) -> gpd.GeoDataFrame:
    rows = []
    for _, r in images_df.iterrows():
        intervals = parse_interval_string(r.get("coverage_intervals"))
        if not intervals:
            continue
        geom = intervals_to_geometry(alignment, intervals)
        if geom is None or geom.is_empty:
            continue
        rows.append(
            {
                "filename": r.get("filename"),
                "pass_id": r.get("pass_id"),
                "pass_direction": r.get("pass_direction"),
                "circuit_run_id": r.get("circuit_run_id"),
                "primary_span": r.get("primary_span"),
                "primary_span_method": r.get("primary_span_method"),
                "reference_id": r.get("reference_id"),
                "coverage_source": r.get("coverage_source"),
                "coverage_status": r.get("coverage_status"),
                "coverage_confidence": r.get("coverage_confidence"),
                "coverage_from": r.get("coverage_from"),
                "coverage_to": r.get("coverage_to"),
                "coverage_length": r.get("coverage_length"),
                "product_status": r.get("product_status"),
                "valid_for_product": r.get("valid_for_product"),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


def suggested_run_stitch_gap(images_df: pd.DataFrame) -> float:
    """
    Derive a conservative stitch distance from the actual image coverage length.

    The intent is not to hide gaps. It only decides whether two interrupted flight
    fragments belong to the same logical circuit run. Any uncovered distance inside
    the resulting run is still reported by build_run_internal_gaps().
    """
    valid = images_df[images_df.get("valid_for_product", False) == True]  # noqa: E712
    vals = pd.to_numeric(valid.get("coverage_length", pd.Series(dtype=float)), errors="coerce")
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals):
        return float(np.median(vals) * 3.0)
    return 5000.0


def _pass_segment_records(images_df: pd.DataFrame) -> pd.DataFrame:
    candidates = images_df[
        images_df["pass_id"].notna()
        & images_df["valid_for_product"].eq(True)
        & images_df["pass_direction"].isin(["FORWARD_PASS", "BACKWARD_PASS"])
    ].copy()
    rows = []
    for pass_id, grp in candidates.groupby("pass_id", sort=False):
        all_intervals = []
        for v in grp["coverage_intervals"]:
            all_intervals.extend(parse_interval_string(v))
        merged = merge_intervals(all_intervals, tolerance=0.0)
        if not merged:
            continue
        rows.append(
            {
                "pass_id": pass_id,
                "eo_source": grp["eo_source"].dropna().iloc[0] if grp["eo_source"].notna().any() else None,
                "pass_direction": grp["pass_direction"].dropna().iloc[0],
                "eo_order_from": float(pd.to_numeric(grp["eo_order"], errors="coerce").min()),
                "eo_order_to": float(pd.to_numeric(grp["eo_order"], errors="coerce").max()),
                "time_from": float(pd.to_numeric(grp["time"], errors="coerce").min()) if grp["time"].notna().any() else np.nan,
                "time_to": float(pd.to_numeric(grp["time"], errors="coerce").max()) if grp["time"].notna().any() else np.nan,
                "coverage_from": float(min(a for a, _ in merged)),
                "coverage_to": float(max(b for _, b in merged)),
                "coverage_mid": float((min(a for a, _ in merged) + max(b for _, b in merged)) / 2.0),
                "valid_images": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def build_circuit_runs(
    images_df: pd.DataFrame,
    alignment: LineString,
    crs: CRS,
    circuit: str,
    run_stitch_gap: float,
    gap_tolerance: float,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict[str, str]]:
    """
    Stitch geometry-valid flight-pass fragments into logical circuit runs.

    A run is intentionally different from a flight pass. A flight can briefly turn,
    cross another circuit, or contain a local opposite-direction fragment; when the
    next valid same-direction fragment resumes near the prior circuit chainage it can
    still belong to the same product run.
    """
    segs = _pass_segment_records(images_df)
    if segs.empty:
        out = images_df.copy()
        out["circuit_run_id"] = None
        return out, gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs), {}

    run_specs: list[dict] = []
    direction_counters = {"FORWARD_PASS": 0, "BACKWARD_PASS": 0}

    for (eo_source, direction), grp in segs.groupby(["eo_source", "pass_direction"], dropna=False, sort=False):
        grp = grp.sort_values(["eo_order_from", "eo_order_to"]).reset_index(drop=True)
        current: list[dict] = []

        def finalize_current():
            nonlocal current
            if not current:
                return
            direction_counters[direction] += 1
            label = "FORWARD" if direction == "FORWARD_PASS" else "BACKWARD"
            run_id = f"{circuit}_{label}_R{direction_counters[direction]:02d}"
            run_specs.append(
                {
                    "circuit_run_id": run_id,
                    "eo_source": eo_source,
                    "run_direction": direction,
                    "segments": list(current),
                }
            )
            current = []

        last_seg = None
        run_cov_from = None
        run_cov_to = None
        for _, row in grp.iterrows():
            seg = row.to_dict()
            if not current:
                current = [seg]
                last_seg = seg
                run_cov_from = float(seg["coverage_from"])
                run_cov_to = float(seg["coverage_to"])
                continue

            if direction == "FORWARD_PASS":
                boundary_gap = float(seg["coverage_from"]) - float(run_cov_to)
                progression_ok = float(seg["coverage_mid"]) >= float(last_seg["coverage_mid"]) - run_stitch_gap
            else:
                boundary_gap = float(run_cov_from) - float(seg["coverage_to"])
                progression_ok = float(seg["coverage_mid"]) <= float(last_seg["coverage_mid"]) + run_stitch_gap

            stitch = progression_ok and boundary_gap <= run_stitch_gap
            if stitch:
                current.append(seg)
                run_cov_from = min(float(run_cov_from), float(seg["coverage_from"]))
                run_cov_to = max(float(run_cov_to), float(seg["coverage_to"]))
                last_seg = seg
            else:
                finalize_current()
                current = [seg]
                last_seg = seg
                run_cov_from = float(seg["coverage_from"])
                run_cov_to = float(seg["coverage_to"])

        finalize_current()

    pass_to_run: dict[str, str] = {}
    run_rows = []
    for spec in run_specs:
        run_id = spec["circuit_run_id"]
        pass_ids = [str(x["pass_id"]) for x in spec["segments"]]
        for pid in pass_ids:
            pass_to_run[pid] = run_id

        valid = images_df[
            images_df["pass_id"].isin(pass_ids)
            & images_df["valid_for_product"].eq(True)
        ].copy()
        all_imgs = images_df[images_df["pass_id"].isin(pass_ids)].copy()
        all_intervals = []
        for v in valid["coverage_intervals"]:
            all_intervals.extend(parse_interval_string(v))
        merged = merge_intervals(all_intervals, tolerance=gap_tolerance)
        if merged:
            cov_from = float(merged[0][0])
            cov_to = float(merged[-1][1])
            covered = float(sum(b - a for a, b in merged))
            extent = max(0.0, cov_to - cov_from)
            internal_gap_count = max(0, len(merged) - 1)
            internal_gap_length = float(sum(merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)))
            geom = intervals_to_geometry(alignment, merged)
        else:
            cov_from = cov_to = covered = extent = internal_gap_length = np.nan
            internal_gap_count = 0
            geom = None

        run_rows.append(
            {
                "circuit_run_id": run_id,
                "eo_source": spec["eo_source"],
                "run_direction": spec["run_direction"],
                "member_passes": ";".join(pass_ids),
                "pass_count": len(pass_ids),
                "image_count": int(len(all_imgs)),
                "valid_image_count": int(len(valid)),
                "eo_order_from": float(pd.to_numeric(all_imgs["eo_order"], errors="coerce").min()) if len(all_imgs) else np.nan,
                "eo_order_to": float(pd.to_numeric(all_imgs["eo_order"], errors="coerce").max()) if len(all_imgs) else np.nan,
                "coverage_from": cov_from,
                "coverage_to": cov_to,
                "covered_length": covered,
                "run_extent_length": extent,
                "coverage_ratio_within_run": (covered / extent) if extent and math.isfinite(extent) else np.nan,
                "internal_gap_count": int(internal_gap_count),
                "internal_gap_length": internal_gap_length,
                "geometry": geom,
            }
        )

    out_images = images_df.copy()
    out_images["circuit_run_id"] = out_images["pass_id"].map(pass_to_run)
    runs = gpd.GeoDataFrame(run_rows, geometry="geometry", crs=crs)
    return out_images, runs, pass_to_run


def build_run_internal_gaps(
    images_df: pd.DataFrame,
    runs: gpd.GeoDataFrame,
    alignment: LineString,
    crs: CRS,
    gap_tolerance: float,
) -> gpd.GeoDataFrame:
    rows = []
    if runs.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)

    for _, run in runs.iterrows():
        run_id = run["circuit_run_id"]
        grp = images_df[
            images_df["circuit_run_id"].eq(run_id)
            & images_df["valid_for_product"].eq(True)
        ].copy()
        all_intervals = []
        for s in grp["coverage_intervals"]:
            all_intervals.extend(parse_interval_string(s))
        merged = merge_intervals(all_intervals, tolerance=gap_tolerance)
        if len(merged) < 2:
            continue
        for i in range(len(merged) - 1):
            a = float(merged[i][1])
            b = float(merged[i + 1][0])
            if b - a <= gap_tolerance:
                continue
            before = grp[pd.to_numeric(grp["coverage_to"], errors="coerce") <= a + gap_tolerance]
            after = grp[pd.to_numeric(grp["coverage_from"], errors="coerce") >= b - gap_tolerance]
            before_image = None
            after_image = None
            if len(before):
                idx = pd.to_numeric(before["coverage_to"], errors="coerce").idxmax()
                before_image = before.loc[idx, "filename"]
            if len(after):
                idx = pd.to_numeric(after["coverage_from"], errors="coerce").idxmin()
                after_image = after.loc[idx, "filename"]
            rows.append(
                {
                    "gap_id": f"{run_id}_IG{i + 1:03d}",
                    "circuit_run_id": run_id,
                    "run_direction": run["run_direction"],
                    "gap_from": a,
                    "gap_to": b,
                    "gap_length": b - a,
                    "gap_type": "INTERNAL_GEOMETRY_GAP",
                    "before_image": before_image,
                    "after_image": after_image,
                    "geometry": substring(alignment, a, b),
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


def build_run_transitions(
    images_df: pd.DataFrame,
    crs: CRS,
    gap_tolerance: float,
    min_overlap: float,
) -> gpd.GeoDataFrame:
    rows = []
    candidates = images_df[
        images_df["circuit_run_id"].notna()
        & images_df["valid_for_product"].eq(True)
    ].copy()

    for run_id, grp in candidates.groupby("circuit_run_id", sort=False):
        direction = grp["pass_direction"].dropna().iloc[0] if grp["pass_direction"].notna().any() else None
        ascending = direction == "FORWARD_PASS"
        grp = grp.sort_values(["centre_chainage", "eo_order"], ascending=[ascending, True], na_position="last")
        vals = list(grp.to_dict("records"))
        for a, b in zip(vals[:-1], vals[1:]):
            ia = parse_interval_string(a.get("coverage_intervals"))
            ib = parse_interval_string(b.get("coverage_intervals"))
            overlap = intervals_overlap_length(ia, ib)
            gap = intervals_gap(ia, ib)
            gap_from, gap_to = intervals_gap_segment(ia, ib)
            progression = (
                float(b.get("centre_chainage")) - float(a.get("centre_chainage"))
                if pd.notna(a.get("centre_chainage")) and pd.notna(b.get("centre_chainage"))
                else np.nan
            )

            if not ia or not ib:
                status = "MISSING_COVERAGE"
            elif math.isfinite(gap) and gap > gap_tolerance:
                status = "HARD_GAP"
            elif overlap < min_overlap:
                status = "WEAK_OVERLAP"
            else:
                status = "OK"

            geom = None
            if all(pd.notna(v) for v in [a.get("camera_x"), a.get("camera_y"), b.get("camera_x"), b.get("camera_y")]):
                geom = LineString(
                    [
                        (float(a["camera_x"]), float(a["camera_y"])),
                        (float(b["camera_x"]), float(b["camera_y"])),
                    ]
                )

            rows.append(
                {
                    "circuit_run_id": run_id,
                    "run_direction": direction,
                    "image_a": a.get("filename"),
                    "image_b": b.get("filename"),
                    "pass_a": a.get("pass_id"),
                    "pass_b": b.get("pass_id"),
                    "bridge_between_passes": bool(a.get("pass_id") != b.get("pass_id")),
                    "reference_a": a.get("reference_id"),
                    "reference_b": b.get("reference_id"),
                    "chainage_a": a.get("centre_chainage"),
                    "chainage_b": b.get("centre_chainage"),
                    "progression": progression,
                    "coverage_overlap": overlap,
                    "coverage_gap": gap,
                    "gap_from": gap_from,
                    "gap_to": gap_to,
                    "coverage_status_a": a.get("coverage_status"),
                    "coverage_status_b": b.get("coverage_status"),
                    "transition_status": status,
                    "gap_type": "SEQUENCE_GAP" if status == "SEQUENCE_JUMP" else ("UNCLASSIFIED_SPATIAL_GAP" if status == "HARD_GAP" else None),
                    "geometry": geom,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


def build_direction_coverage(
    images_df: pd.DataFrame,
    alignment: LineString,
    crs: CRS,
    gap_tolerance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Coverage by direction, keeping CONFIRMED and PROVISIONAL coverage distinct."""
    cov_rows = []
    confirmed_rows = []
    gap_rows = []
    directions = ["FORWARD_PASS", "BACKWARD_PASS"]

    for direction in directions:
        accepted = images_df[
            images_df["pass_direction"].eq(direction)
            & images_df["valid_for_product"].eq(True)
        ].copy()
        confirmed = accepted[accepted.get("coverage_status", "CONFIRMED").eq("CONFIRMED")].copy()
        provisional = accepted[accepted.get("coverage_status", "CONFIRMED").eq("PROVISIONAL")].copy()

        all_intervals = []
        for value in accepted.get("coverage_intervals", pd.Series(dtype=str)):
            all_intervals.extend(parse_interval_string(value))
        confirmed_intervals = []
        for value in confirmed.get("coverage_intervals", pd.Series(dtype=str)):
            confirmed_intervals.extend(parse_interval_string(value))

        merged = merge_intervals(all_intervals, tolerance=gap_tolerance)
        merged_confirmed = merge_intervals(confirmed_intervals, tolerance=gap_tolerance)
        covered = float(sum(b - a for a, b in merged)) if merged else 0.0
        confirmed_covered = float(sum(b - a for a, b in merged_confirmed)) if merged_confirmed else 0.0
        length = float(alignment.length)
        label = "FORWARD" if direction == "FORWARD_PASS" else "BACKWARD"

        cov_rows.append(
            {
                "direction": label,
                "pass_direction": direction,
                "valid_images": int(len(accepted)),
                "confirmed_images": int(len(confirmed)),
                "provisional_images": int(len(provisional)),
                "run_count": int(accepted["circuit_run_id"].nunique(dropna=True)) if "circuit_run_id" in accepted else 0,
                "coverage_from": float(merged[0][0]) if merged else np.nan,
                "coverage_to": float(merged[-1][1]) if merged else np.nan,
                "covered_length": covered,
                "confirmed_covered_length": confirmed_covered,
                "alignment_length": length,
                "coverage_percent": (100.0 * covered / length) if length > 0 else np.nan,
                "confirmed_coverage_percent": (100.0 * confirmed_covered / length) if length > 0 else np.nan,
                "provisional_gain_percent": (100.0 * max(0.0, covered - confirmed_covered) / length) if length > 0 else np.nan,
                "geometry": intervals_to_geometry(alignment, merged),
            }
        )
        confirmed_rows.append(
            {
                "direction": label,
                "pass_direction": direction,
                "confirmed_images": int(len(confirmed)),
                "covered_length": confirmed_covered,
                "alignment_length": length,
                "coverage_percent": (100.0 * confirmed_covered / length) if length > 0 else np.nan,
                "geometry": intervals_to_geometry(alignment, merged_confirmed),
            }
        )

        cursor = 0.0
        gap_no = 0
        for a, b in merged:
            a = max(0.0, float(a))
            b = min(length, float(b))
            if a - cursor > gap_tolerance:
                gap_no += 1
                gap_rows.append(
                    {
                        "gap_id": f"{label}_DG{gap_no:03d}",
                        "direction": label,
                        "pass_direction": direction,
                        "gap_from": cursor,
                        "gap_to": a,
                        "gap_length": a - cursor,
                        "gap_type": "UNCLASSIFIED_GAP",
                        "geometry": substring(alignment, cursor, a),
                    }
                )
            cursor = max(cursor, b)
        if length - cursor > gap_tolerance:
            gap_no += 1
            gap_rows.append(
                {
                    "gap_id": f"{label}_DG{gap_no:03d}",
                    "direction": label,
                    "pass_direction": direction,
                    "gap_from": cursor,
                    "gap_to": length,
                    "gap_length": length - cursor,
                    "gap_type": "UNCLASSIFIED_GAP",
                    "geometry": substring(alignment, cursor, length),
                }
            )

    cov = gpd.GeoDataFrame(cov_rows, geometry="geometry", crs=crs)
    confirmed_cov = gpd.GeoDataFrame(confirmed_rows, geometry="geometry", crs=crs)
    gaps = gpd.GeoDataFrame(gap_rows, geometry="geometry", crs=crs) if gap_rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    return cov, confirmed_cov, gaps


def classify_direction_gaps(
    gaps: gpd.GeoDataFrame,
    images_df: pd.DataFrame,
    existing: pd.DataFrame,
    spans: gpd.GeoDataFrame,
    gap_tolerance: float,
) -> gpd.GeoDataFrame:
    """
    Classify residual directional gaps after provisional footprint recovery.

    METADATA_GAP: current EO points to the gap but usable footprint coverage is missing.
    SOURCE_DATA_GAP: historical 163 contains references in the gap but current source set does not cover it.
    CAPTURE_GAP: no metadata/source evidence explains the missing current coverage.
    """
    if gaps.empty:
        return gaps
    out = gaps.copy()
    existing_counts = Counter()
    if not existing.empty and "span_id" in existing.columns:
        for sid in existing["span_id"].dropna().astype(str):
            existing_counts[sid] += 1

    confirmed_lengths = pd.to_numeric(
        images_df.loc[images_df.get("coverage_status", pd.Series(index=images_df.index)).eq("CONFIRMED"), "coverage_length"],
        errors="coerce",
    ).dropna()
    typical_image_coverage = float(confirmed_lengths.median()) if len(confirmed_lengths) else max(1.0, float(gap_tolerance) * 10.0)

    gap_types = []
    metadata_counts = []
    metadata_cover_fracs = []
    existing_image_counts = []
    existing_span_lists = []
    evidence = []

    for _, g in out.iterrows():
        a, b = float(g["gap_from"]), float(g["gap_to"])
        direction = g.get("pass_direction")

        meta = images_df[
            images_df["pass_direction"].eq(direction)
            & images_df.get("spatial_compatible", pd.Series(False, index=images_df.index)).eq(True)
            & images_df.get("coverage_status", pd.Series("", index=images_df.index)).eq("MISSING_METADATA")
        ].copy()
        if len(meta):
            och = pd.to_numeric(meta.get("optical_target_chainage"), errors="coerce")
            cch = pd.to_numeric(meta.get("camera_chainage"), errors="coerce")
            # Optical target is preferred. Camera chainage is only a loose fallback.
            hit = (och.between(a - gap_tolerance, b + gap_tolerance)) | (
                och.isna() & cch.between(a - 2.0 * gap_tolerance, b + 2.0 * gap_tolerance)
            )
            meta = meta[hit]

        span_hits = spans[
            pd.to_numeric(spans["chainage_to"], errors="coerce").gt(a)
            & pd.to_numeric(spans["chainage_from"], errors="coerce").lt(b)
        ]
        existing_spans = [str(x) for x in span_hits["span_id"] if existing_counts.get(str(x), 0) > 0]
        ex_images = int(sum(existing_counts.get(sid, 0) for sid in existing_spans))

        meta_intervals = []
        if len(meta):
            for ch in pd.to_numeric(meta.get("optical_target_chainage"), errors="coerce").dropna():
                meta_intervals.append((
                    max(a, float(ch) - typical_image_coverage / 2.0),
                    min(b, float(ch) + typical_image_coverage / 2.0),
                ))
        meta_merged = merge_intervals([(x, y) for x, y in meta_intervals if y > x], tolerance=gap_tolerance)
        meta_len = float(sum(y - x for x, y in meta_merged)) if meta_merged else 0.0
        meta_frac = meta_len / (b - a) if b > a else 0.0
        metadata_is_explanatory = bool(len(meta)) and (
            (b - a) <= 2.5 * typical_image_coverage or meta_frac >= 0.50
        )

        if metadata_is_explanatory:
            gt = "METADATA_GAP"
            ev = f"{len(meta)} current EO image(s) explain approximately {meta_frac:.0%} of the gap without confirmed footprint coverage"
        elif ex_images > 0:
            gt = "SOURCE_DATA_GAP"
            ev = f"existing 163 has {ex_images} image(s) across {len(existing_spans)} span(s) in gap"
        else:
            gt = "CAPTURE_GAP"
            ev = "no current metadata recovery or historical 163 evidence"

        gap_types.append(gt)
        metadata_counts.append(int(len(meta)))
        metadata_cover_fracs.append(meta_frac)
        existing_image_counts.append(ex_images)
        existing_span_lists.append(";".join(existing_spans))
        evidence.append(ev)

    out["gap_type"] = gap_types
    out["metadata_candidate_images"] = metadata_counts
    out["metadata_estimated_coverage_fraction"] = metadata_cover_fracs
    out["existing_163_images_in_gap"] = existing_image_counts
    out["existing_163_spans_in_gap"] = existing_span_lists
    out["gap_evidence"] = evidence
    return out


def classify_transition_gaps(
    transitions: gpd.GeoDataFrame,
    images_df: pd.DataFrame,
    gap_tolerance: float,
) -> gpd.GeoDataFrame:
    """Mark an ordered-image discontinuity as SEQUENCE_GAP when other accepted imagery covers it."""
    if transitions.empty:
        return transitions
    out = transitions.copy()
    gap_types = []
    for _, r in out.iterrows():
        status = r.get("transition_status")
        if status == "SEQUENCE_JUMP":
            gap_types.append("SEQUENCE_GAP")
            continue
        if status != "HARD_GAP" or pd.isna(r.get("gap_from")) or pd.isna(r.get("gap_to")):
            gap_types.append(r.get("gap_type"))
            continue
        a, b = float(r["gap_from"]), float(r["gap_to"])
        direction = r.get("run_direction")
        accepted = images_df[
            images_df["pass_direction"].eq(direction)
            & images_df["valid_for_product"].eq(True)
        ]
        intervals = []
        for s in accepted.get("coverage_intervals", pd.Series(dtype=str)):
            intervals.extend(parse_interval_string(s))
        merged = merge_intervals(intervals, tolerance=gap_tolerance)
        covered = intervals_overlap_length(merged, [(a, b)]) if b > a else 0.0
        frac = covered / (b - a) if b > a else 0.0
        gap_types.append("SEQUENCE_GAP" if frac >= 0.95 else "SPATIAL_GAP")
    out["gap_type"] = gap_types
    return out


def build_gap_events(
    direction_gaps: gpd.GeoDataFrame,
    transitions: gpd.GeoDataFrame,
    alignment: LineString,
    crs: CRS,
) -> gpd.GeoDataFrame:
    rows = []
    if direction_gaps is not None and not direction_gaps.empty:
        for _, r in direction_gaps.iterrows():
            d = r.drop(labels=["geometry"], errors="ignore").to_dict()
            d.update({"event_source": "DIRECTION_COVERAGE", "image_a": None, "image_b": None, "geometry": r.geometry})
            rows.append(d)
    if transitions is not None and not transitions.empty:
        seq = transitions[transitions.get("gap_type", pd.Series(index=transitions.index)).eq("SEQUENCE_GAP")]
        for i, r in seq.iterrows():
            a = r.get("gap_from")
            b = r.get("gap_to")
            geom = None
            if pd.notna(a) and pd.notna(b) and float(b) > float(a):
                geom = substring(alignment, float(a), float(b))
            d = {
                "gap_id": f"SEQ_{i + 1:04d}",
                "direction": "FORWARD" if r.get("run_direction") == "FORWARD_PASS" else "BACKWARD",
                "pass_direction": r.get("run_direction"),
                "gap_from": a,
                "gap_to": b,
                "gap_length": (float(b) - float(a)) if pd.notna(a) and pd.notna(b) else r.get("coverage_gap"),
                "gap_type": "SEQUENCE_GAP",
                "gap_evidence": "ordered transition jumps while other accepted imagery covers the interval",
                "event_source": "RUN_TRANSITION",
                "image_a": r.get("image_a"),
                "image_b": r.get("image_b"),
                "geometry": geom,
            }
            rows.append(d)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)


def build_span_benchmark(
    spans: gpd.GeoDataFrame,
    images_df: pd.DataFrame,
    existing: pd.DataFrame,
    circuit: str,
) -> gpd.GeoDataFrame:
    """Aggregate benchmark. It does not assume a 1:1 relationship between images and spans."""
    out = spans.copy()
    generated = (
        images_df[images_df["valid_for_product"].eq(True)]
        .groupby("primary_span")
        .size()
        .rename("generated_primary_count")
    )
    ex = existing.copy()
    if not ex.empty:
        ex = ex[ex.get("parse_status", "OK").eq("OK")]
        if "circuit" in ex.columns:
            ex = ex[ex["circuit"].astype(str).str.upper().eq(str(circuit).upper())]
        existing_counts = ex.groupby("span_id").size().rename("existing_163_count")
    else:
        existing_counts = pd.Series(dtype=int, name="existing_163_count")

    out = out.merge(generated, left_on="span_id", right_index=True, how="left")
    out = out.merge(existing_counts, left_on="span_id", right_index=True, how="left")
    out["generated_primary_count"] = out["generated_primary_count"].fillna(0).astype(int)
    out["existing_163_count"] = out["existing_163_count"].fillna(0).astype(int)
    out["count_delta_generated_minus_existing"] = out["generated_primary_count"] - out["existing_163_count"]
    out["benchmark_state"] = np.select(
        [
            out["generated_primary_count"].gt(0) & out["existing_163_count"].gt(0),
            out["generated_primary_count"].eq(0) & out["existing_163_count"].gt(0),
            out["generated_primary_count"].gt(0) & out["existing_163_count"].eq(0),
        ],
        ["BOTH", "EXISTING_ONLY", "GENERATED_ONLY"],
        default="NEITHER",
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=spans.crs)


def _image_fingerprint(path: Path, size: tuple[int, int] = (32, 24)) -> np.ndarray:
    """Low-resolution appearance descriptor, deliberately not a learned/DINO feature."""
    from PIL import Image

    # These are trusted local aerial source images and can legitimately exceed
    # Pillow's generic decompression-bomb pixel threshold.  We only downsample.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            # Existing 163 annotation is in the lower-right. Removing the lower 18%
            # makes source matching much less sensitive to the annotation/compression.
            crop_h = max(1, int(round(h * 0.82)))
            im = im.crop((0, 0, w, crop_h))
            resampling = getattr(Image, "Resampling", Image)
            im = im.resize(size, resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32).reshape(-1)
    arr -= float(arr.mean())
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.zeros_like(arr)
    return arr / norm


def build_existing_163_image_benchmark(
    raw: pd.DataFrame,
    existing: pd.DataFrame,
    images_df: pd.DataFrame,
    circuit: str,
) -> pd.DataFrame:
    """
    Match existing final 163 JPGs back to raw JPGs by low-res appearance.

    This is only a benchmark helper. It is not used to select product imagery and is
    intentionally independent from DINOv3. The result exposes best/second-best score
    and a confidence class so ambiguous matches remain visible.
    """
    if raw.empty or existing.empty:
        return pd.DataFrame()
    try:
        import PIL  # noqa: F401
    except ImportError:
        warnings.warn("Pillow is not installed; skipping image-level 163 benchmark")
        return pd.DataFrame()

    ex = existing.copy()
    if "parse_status" in ex.columns:
        ex = ex[ex["parse_status"].eq("OK")]
    if "circuit" in ex.columns:
        ex = ex[ex["circuit"].astype(str).str.upper().eq(str(circuit).upper())]
    if ex.empty:
        return pd.DataFrame()

    raw_names = []
    raw_vecs = []
    for _, r in raw.iterrows():
        path = Path(str(r["raw_path"]))
        if not path.exists():
            continue
        try:
            raw_vecs.append(_image_fingerprint(path))
            raw_names.append(str(r["filename"]))
        except Exception as exc:
            warnings.warn(f"Could not fingerprint raw image {path.name}: {exc}")
    if not raw_vecs:
        return pd.DataFrame()

    raw_mat = np.vstack(raw_vecs).astype(np.float32, copy=False)
    image_lookup = images_df.set_index("filename").to_dict("index")
    rows = []
    for _, r in ex.iterrows():
        path = Path(str(r["path"]))
        if not path.exists():
            continue
        try:
            v = _image_fingerprint(path).astype(np.float32, copy=False)
        except Exception as exc:
            warnings.warn(f"Could not fingerprint existing 163 image {path.name}: {exc}")
            continue
        scores = raw_mat @ v
        if len(scores) == 1:
            order = np.array([0], dtype=int)
        else:
            order = np.argpartition(scores, -2)[-2:]
            order = order[np.argsort(scores[order])[::-1]]
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx
        best_score = float(scores[best_idx])
        second_score = float(scores[second_idx]) if len(order) > 1 else np.nan
        margin = best_score - second_score if len(order) > 1 else np.nan
        if best_score >= 0.985 and (not math.isfinite(margin) or margin >= 0.010):
            confidence = "HIGH"
        elif best_score >= 0.950 and (not math.isfinite(margin) or margin >= 0.003):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        raw_name = raw_names[best_idx]
        img = image_lookup.get(raw_name, {})
        generated_span = img.get("primary_span")
        if confidence == "LOW":
            status = "LOW_SOURCE_MATCH_CONFIDENCE"
        elif not generated_span or pd.isna(generated_span):
            status = "NO_GENERATED_SPAN"
        elif str(generated_span) == str(r.get("span_id")):
            status = "SPAN_MATCH"
        else:
            status = "SPAN_MISMATCH"

        rows.append(
            {
                "existing_163_filename": r.get("filename"),
                "existing_span": r.get("span_id"),
                "existing_output_id": r.get("output_id"),
                "capture_date": r.get("capture_date"),
                "matched_raw_filename": raw_name,
                "second_raw_filename": raw_names[second_idx] if len(order) > 1 else None,
                "appearance_score": best_score,
                "appearance_second_score": second_score,
                "appearance_margin": margin,
                "source_match_confidence": confidence,
                "generated_primary_span": generated_span,
                "generated_primary_span_geometry": img.get("primary_span_geometry"),
                "generated_primary_span_optical": img.get("primary_span_optical"),
                "generated_primary_span_method": img.get("primary_span_method"),
                "optical_target_chainage": img.get("optical_target_chainage"),
                "optical_target_distance": img.get("optical_target_distance"),
                "optical_target_confidence": img.get("optical_target_confidence"),
                "coverage_status": img.get("coverage_status"),
                "product_status": img.get("product_status"),
                "generated_reference_id": img.get("reference_id"),
                "generated_pass_id": img.get("pass_id"),
                "generated_circuit_run_id": img.get("circuit_run_id"),
                "generated_view_direction": img.get("view_direction"),
                "generated_valid_for_product": img.get("valid_for_product"),
                "benchmark_status": status,
            }
        )
    return pd.DataFrame(rows)



def build_opk_benchmark_summary(image_benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compare coverage-midpoint vs approximate OPK optical targeting on high-confidence source matches."""
    if image_benchmark.empty:
        return pd.DataFrame()
    b = image_benchmark[image_benchmark["source_match_confidence"].isin(["HIGH", "MEDIUM"])].copy()
    rows = []
    for label, col in [
        ("COVERAGE_MIDPOINT", "generated_primary_span_geometry"),
        ("OPK_OPTICAL", "generated_primary_span_optical"),
        ("SELECTED_PRIMARY", "generated_primary_span"),
    ]:
        if col not in b.columns:
            continue
        c = b[b[col].notna()].copy()
        exact = int(c[col].astype(str).eq(c["existing_span"].astype(str)).sum()) if len(c) else 0
        rows.append(
            {
                "target_method": label,
                "comparable_images": int(len(c)),
                "exact_span_matches": exact,
                "exact_span_match_percent": (100.0 * exact / len(c)) if len(c) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def benchmark_points_layer(
    benchmark: pd.DataFrame,
    images_df: pd.DataFrame,
    crs: CRS,
) -> gpd.GeoDataFrame:
    if benchmark.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    coords = images_df[["filename", "camera_x", "camera_y"]].drop_duplicates("filename")
    out = benchmark.merge(coords, left_on="matched_raw_filename", right_on="filename", how="left")
    out = out[out["camera_x"].notna() & out["camera_y"].notna()].copy()
    if out.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    geom = [Point(float(x), float(y)) for x, y in zip(out["camera_x"], out["camera_y"])]
    out = out.drop(columns=["filename"], errors="ignore")
    return gpd.GeoDataFrame(out, geometry=geom, crs=crs)


def build_summary_v03(
    paths: ProjectPaths,
    metadata: dict,
    raw: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    eo: pd.DataFrame,
    eo_summary: pd.DataFrame,
    structures: gpd.GeoDataFrame,
    spans: gpd.GeoDataFrame,
    passes: pd.DataFrame,
    runs: gpd.GeoDataFrame,
    images: pd.DataFrame,
    recovery: pd.DataFrame,
    run_transitions: gpd.GeoDataFrame,
    run_gaps: gpd.GeoDataFrame,
    direction_coverage: gpd.GeoDataFrame,
    direction_gaps: gpd.GeoDataFrame,
    gap_events: gpd.GeoDataFrame,
    existing: pd.DataFrame,
    span_benchmark: gpd.GeoDataFrame,
    image_benchmark: pd.DataFrame,
    opk_benchmark: pd.DataFrame,
    circuit: str,
) -> dict:
    run_summary = []
    if not runs.empty:
        for _, r in runs.iterrows():
            rid = r["circuit_run_id"]
            trans = run_transitions[run_transitions["circuit_run_id"].eq(rid)] if not run_transitions.empty else pd.DataFrame()
            rgaps = run_gaps[run_gaps["circuit_run_id"].eq(rid)] if not run_gaps.empty else pd.DataFrame()
            run_summary.append(
                {
                    "circuit_run_id": rid,
                    "direction": r["run_direction"],
                    "member_passes": str(r["member_passes"]).split(";") if pd.notna(r["member_passes"]) else [],
                    "valid_images": int(r["valid_image_count"]),
                    "coverage_from": float(r["coverage_from"]) if pd.notna(r["coverage_from"]) else None,
                    "coverage_to": float(r["coverage_to"]) if pd.notna(r["coverage_to"]) else None,
                    "coverage_ratio_within_run": float(r["coverage_ratio_within_run"]) if pd.notna(r["coverage_ratio_within_run"]) else None,
                    "internal_gaps": int(len(rgaps)),
                    "transition_flags": dict(Counter(trans["transition_status"])) if not trans.empty else {},
                    "transition_gap_types": dict(Counter(trans["gap_type"].dropna())) if not trans.empty and "gap_type" in trans else {},
                }
            )

    direction_summary = []
    if not direction_coverage.empty:
        for _, r in direction_coverage.iterrows():
            dg = direction_gaps[direction_gaps["direction"].eq(r["direction"])] if not direction_gaps.empty else pd.DataFrame()
            direction_summary.append(
                {
                    "direction": r["direction"],
                    "valid_images": int(r["valid_images"]),
                    "confirmed_images": int(r.get("confirmed_images", 0)),
                    "provisional_images": int(r.get("provisional_images", 0)),
                    "runs": int(r["run_count"]),
                    "covered_length": float(r["covered_length"]),
                    "coverage_percent": float(r["coverage_percent"]),
                    "confirmed_coverage_percent": float(r.get("confirmed_coverage_percent", np.nan)),
                    "provisional_gain_percent": float(r.get("provisional_gain_percent", np.nan)),
                    "gaps": int(len(dg)),
                    "gap_types": dict(Counter(dg["gap_type"])) if len(dg) else {},
                }
            )

    bench = {}
    if not span_benchmark.empty:
        bench["aggregate_span_states"] = {
            str(k): int(v) for k, v in span_benchmark["benchmark_state"].value_counts().items()
        }
    if not image_benchmark.empty:
        confident = image_benchmark[image_benchmark["source_match_confidence"].isin(["HIGH", "MEDIUM"])]
        comparable = confident[confident["generated_primary_span"].notna()]
        exact = int(comparable["benchmark_status"].eq("SPAN_MATCH").sum()) if len(comparable) else 0
        bench["image_level"] = {
            "rows": int(len(image_benchmark)),
            "high_confidence_source_matches": int(image_benchmark["source_match_confidence"].eq("HIGH").sum()),
            "medium_confidence_source_matches": int(image_benchmark["source_match_confidence"].eq("MEDIUM").sum()),
            "low_confidence_source_matches": int(image_benchmark["source_match_confidence"].eq("LOW").sum()),
            "comparable_span_labels": int(len(comparable)),
            "exact_span_matches": exact,
            "exact_span_match_percent": (100.0 * exact / len(comparable)) if len(comparable) else None,
        }
    if not opk_benchmark.empty:
        bench["opk_targeting"] = opk_benchmark.to_dict("records")

    existing_dates = {}
    if not existing.empty and "capture_date" in existing:
        existing_dates = {str(k): int(v) for k, v in existing["capture_date"].value_counts(dropna=False).items()}

    valid = images[images["valid_for_product"].eq(True)] if len(images) else pd.DataFrame()
    confirmed_valid = valid[valid.get("product_status", pd.Series(index=valid.index)).eq("CONFIRMED")] if len(valid) else pd.DataFrame()
    provisional_valid = valid[valid.get("product_status", pd.Series(index=valid.index)).eq("PROVISIONAL")] if len(valid) else pd.DataFrame()

    return {
        "poc_version": "0.4",
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
            "pls_structures": int(len(structures)),
            "reference_structures": int(metadata.get("reference_structure_count", len(structures))),
            "intermediate_pls_structures": int(metadata.get("intermediate_pls_structure_count", 0)),
            "reference_spans": int(len(spans)),
            "flight_passes": int(len(passes)),
            "circuit_runs": int(len(runs)),
            "valid_product_images": int(len(valid)),
            "confirmed_product_images": int(len(confirmed_valid)),
            "provisional_product_images": int(len(provisional_valid)),
            "recovered_missing_footprints": int(len(recovery)),
            "existing_163_images": int(len(existing)),
        },
        "reference_topology": {
            "requested_mode": metadata.get("reference_topology_mode_requested"),
            "status": metadata.get("reference_topology_status"),
            "reference_structure_count": int(metadata.get("reference_structure_count", 0)),
            "reference_span_count": int(metadata.get("reference_span_count", 0)),
            "intermediate_pls_structure_count": int(metadata.get("intermediate_pls_structure_count", 0)),
        },
        "eo_sources": eo_summary.to_dict("records") if not eo_summary.empty else [],
        "circuit_runs": run_summary,
        "direction_coverage": direction_summary,
        "gap_types": dict(Counter(gap_events["gap_type"])) if not gap_events.empty and "gap_type" in gap_events else {},
        "benchmark": bench,
        "existing_163_capture_dates": existing_dates,
    }

# -----------------------------------------------------------------------------
# v0.5 product generation / reference confidence / structure crops
# -----------------------------------------------------------------------------


def _clean_token(value: object, fallback: str = "NA") -> str:
    s = str(value).strip() if value is not None and pd.notna(value) else ""
    if not s:
        s = fallback
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-_") or fallback


def _date_code_from_iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    try:
        return datetime.fromisoformat(s).strftime("%m%d%y")
    except ValueError:
        return None


def _date_display_from_code(code: object) -> str:
    if code is None or pd.isna(code):
        return "Unknown"
    s = str(code).strip()
    try:
        return datetime.strptime(s, "%m%d%y").strftime("%m-%d-%Y")
    except ValueError:
        return "Unknown"


def validate_capture_date_code(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%m%d%y")
    except ValueError as exc:
        raise ValueError("--capture-date must use MMDDYY, e.g. 042226") from exc
    return value


def augment_reference_confidence(
    images_df: pd.DataFrame,
    spans: gpd.GeoDataFrame,
    boundary_distance: float,
    boundary_fraction: float,
) -> pd.DataFrame:
    """Add explicit confidence and adjacent-span ambiguity fields.

    The product sequence is never filtered on these fields.  They only explain
    how reliable the visible reference label is for a retained image.
    """
    out = images_df.copy()
    spans_sorted = spans.sort_values("chainage_from").reset_index(drop=True)
    span_pos = {str(r.span_id): i for i, r in spans_sorted.iterrows()}

    cols = {
        "reference_assignment_status": [],
        "reference_confidence": [],
        "reference_boundary_distance": [],
        "reference_boundary_fraction": [],
        "reference_boundary_threshold": [],
        "reference_boundary_side": [],
        "alternative_reference": [],
    }

    for _, r in out.iterrows():
        sid = r.get("primary_span")
        ch = pd.to_numeric(pd.Series([r.get("reference_chainage")]), errors="coerce").iloc[0]
        method = str(r.get("primary_span_method") or "NONE")
        optical_conf = str(r.get("optical_target_confidence") or "NONE")

        status = "UNASSIGNED"
        conf = "NONE"
        dist = np.nan
        frac = np.nan
        threshold = np.nan
        side = None
        alt = None

        if sid is not None and pd.notna(sid) and math.isfinite(float(ch)) and str(sid) in span_pos:
            i = span_pos[str(sid)]
            s = spans_sorted.iloc[i]
            a = float(s.chainage_from)
            b = float(s.chainage_to)
            slen = max(b - a, 1e-9)
            d_from = abs(float(ch) - a)
            d_to = abs(b - float(ch))
            if d_from <= d_to:
                dist = d_from
                side = "FROM"
                if i > 0:
                    alt = str(spans_sorted.iloc[i - 1].span_id)
            else:
                dist = d_to
                side = "TO"
                if i + 1 < len(spans_sorted):
                    alt = str(spans_sorted.iloc[i + 1].span_id)
            frac = dist / slen
            threshold = min(float(boundary_distance), slen * float(boundary_fraction))

            if method == "OPK_OPTICAL":
                conf = "HIGH" if optical_conf == "HIGH" else ("MEDIUM" if optical_conf == "MEDIUM" else "LOW")
            elif method == "COVERAGE_MIDPOINT":
                conf = "LOW"
            else:
                conf = "LOW"

            if dist <= threshold and alt is not None:
                status = "BOUNDARY_AMBIGUOUS"
                conf = "AMBIGUOUS"
            else:
                status = "ASSIGNED"

        cols["reference_assignment_status"].append(status)
        cols["reference_confidence"].append(conf)
        cols["reference_boundary_distance"].append(dist)
        cols["reference_boundary_fraction"].append(frac)
        cols["reference_boundary_threshold"].append(threshold)
        cols["reference_boundary_side"].append(side)
        cols["alternative_reference"].append(alt)

    for name, vals in cols.items():
        out[name] = vals
    return out



def classify_span_completeness(
    images_df: pd.DataFrame,
    reference_structures: gpd.GeoDataFrame,
    reference_mode: str,
    outside_tolerance: float,
    edge_margin: float,
) -> pd.DataFrame:
    """Describe span visibility for every geometry-valid sequence image.

    IMPORTANT: visibility is QA metadata, not a product-membership filter. A legal
    continuous-oblique sequence may need frames that show only part of a span, one
    structure group, or no structure group at all. Removing those frames can create
    an artificial continuity gap.

    Membership remains controlled by valid_for_product / sequence_keep. This
    routine only records how much span evidence is visible and whether the label is
    supported by a full consecutive structure pair. Delivered DXF footprints are
    approximate, so near-edge/tolerated-outside states remain explicit QA metadata.
    """
    out = images_df.copy()
    defaults = {
        "span_view_status": "NOT_GEOMETRIC_CANDIDATE",
        "span_view_confidence": "NONE",
        "span_view_candidate_structure_count": 0,
        "span_view_valid_pair_count": 0,
        "span_view_pair": None,
        "span_view_from_reference": np.nan,
        "span_view_to_reference": np.nan,
        "span_view_from_structure": None,
        "span_view_to_structure": None,
        "span_view_pair_matches_primary": False,
        "span_view_uses_outside_tolerance": False,
        "span_view_evidence": None,
        "span_view_rejection_reason": None,
        "span_view_outside_tolerance": float(outside_tolerance),
        "span_view_edge_margin": float(edge_margin),
        "span_visibility_class": "NOT_APPLICABLE",
        "full_span_visible": False,
        "sequence_keep": False,
        # Backward-compatible alias. In v0.7 this means membership in the legal
        # continuous sequence, NOT full-span visibility.
        "valid_for_163": False,
        "sequence_only_image": False,
    }
    for col, val in defaults.items():
        out[col] = val

    if out.empty:
        return out

    if reference_mode != "span":
        geom = out.get("valid_for_product", pd.Series(False, index=out.index)).eq(True)
        out.loc[geom, "span_view_status"] = "NOT_APPLICABLE_STRUCTURE_MODE"
        out.loc[geom, "span_view_confidence"] = "NOT_APPLICABLE"
        out.loc[geom, "span_visibility_class"] = "NOT_APPLICABLE"
        out.loc[geom, "sequence_keep"] = True
        out.loc[geom, "valid_for_163"] = True
        return out

    refs = reference_structures.copy()
    if "is_reference_structure" in refs.columns:
        refs = refs[refs["is_reference_structure"].eq(True)].copy()
    refs = refs[
        pd.to_numeric(refs.get("chainage"), errors="coerce").notna()
        & pd.to_numeric(refs.get("reference_idx"), errors="coerce").notna()
    ].copy()
    refs["chainage"] = pd.to_numeric(refs["chainage"], errors="coerce")
    refs["reference_idx"] = pd.to_numeric(refs["reference_idx"], errors="coerce").astype(int)
    refs = refs.sort_values(["chainage", "reference_idx"]).reset_index(drop=True)
    refs["_ref_position"] = np.arange(len(refs), dtype=int)
    if len(refs) < 2:
        return out

    for idx, r in out.iterrows():
        if not bool(r.get("valid_for_product") is True or r.get("valid_for_product") == True):
            continue
        # Product membership is already decided by flight/circuit/AOI geometry.
        # Span visibility must never remove a legal frame from a continuous run.
        out.at[idx, "sequence_keep"] = True
        out.at[idx, "valid_for_163"] = True
        intervals = parse_interval_string(r.get("coverage_intervals"))
        if not intervals:
            out.at[idx, "span_view_status"] = "NO_COVERAGE"
            out.at[idx, "span_visibility_class"] = "NO_STRUCTURE_VISIBLE"
            out.at[idx, "span_view_rejection_reason"] = "VISIBILITY_QA:NO_COVERAGE"
            out.at[idx, "sequence_only_image"] = False
            continue

        lo = min(a for a, _ in intervals) - float(outside_tolerance)
        hi = max(b for _, b in intervals) + float(outside_tolerance)
        nearby = refs[(refs["chainage"] >= lo) & (refs["chainage"] <= hi)].copy()
        evidence = []
        by_position: dict[int, dict] = {}

        for _, s in nearby.iterrows():
            ch = float(s.chainage)
            signed_margins = []
            for a, b in intervals:
                if a <= ch <= b:
                    signed = min(ch - a, b - ch)
                elif ch < a:
                    signed = -(a - ch)
                else:
                    signed = -(ch - b)
                signed_margins.append(float(signed))
            if not signed_margins:
                continue
            signed = max(signed_margins)
            if signed >= float(edge_margin):
                state = "SAFE_INSIDE"
            elif signed >= 0.0:
                state = "NEAR_EDGE_INSIDE"
            elif signed >= -float(outside_tolerance):
                state = "TOLERATED_OUTSIDE"
            else:
                state = "OUTSIDE"
            if state == "OUTSIDE":
                continue
            rec = {
                "position": int(s._ref_position),
                "reference_idx": int(s.reference_idx),
                "structure_id": str(s.get("structure_id")),
                "chainage": ch,
                "state": state,
                "signed_margin": float(signed),
            }
            by_position[rec["position"]] = rec
            evidence.append(
                f"{rec['reference_idx']}:{state}:{rec['signed_margin']:+.1f}"
            )

        valid_pairs = []
        edge_pairs = []
        for pos in range(len(refs) - 1):
            a = by_position.get(pos)
            b = by_position.get(pos + 1)
            if a is None or b is None:
                continue
            states = {a["state"], b["state"]}
            pair_mid = (a["chainage"] + b["chainage"]) / 2.0
            target = pd.to_numeric(pd.Series([r.get("reference_chainage")]), errors="coerce").iloc[0]
            if pd.isna(target) or not math.isfinite(float(target)):
                target = pair_mid
            safe_pair = (
                a["state"] in {"SAFE_INSIDE", "TOLERATED_OUTSIDE"}
                and b["state"] in {"SAFE_INSIDE", "TOLERATED_OUTSIDE"}
                and "SAFE_INSIDE" in states
            )
            quality = 2 if states == {"SAFE_INSIDE"} else 1
            candidate = {
                "a": a,
                "b": b,
                "quality": quality,
                "target_distance": abs(pair_mid - float(target)),
            }
            if safe_pair:
                valid_pairs.append(candidate)
            elif "NEAR_EDGE_INSIDE" in states:
                edge_pairs.append(candidate)

        out.at[idx, "span_view_candidate_structure_count"] = int(len(by_position))
        out.at[idx, "span_view_valid_pair_count"] = int(len(valid_pairs))
        out.at[idx, "span_view_evidence"] = ";".join(evidence)

        if valid_pairs:
            best = sorted(valid_pairs, key=lambda p: (-p["quality"], p["target_distance"]))[0]
            a = best["a"]
            b = best["b"]
            pair_id = f"{a['reference_idx']}-{b['reference_idx']}"
            uses_tol = "TOLERATED_OUTSIDE" in {a["state"], b["state"]}
            coverage_status = str(r.get("coverage_status") or "")
            if uses_tol:
                conf = "MEDIUM"
            else:
                conf = "HIGH"
            if coverage_status == "PROVISIONAL":
                conf = "LOW" if conf == "MEDIUM" else "MEDIUM"

            out.at[idx, "span_view_status"] = "VALID_SPAN_VIEW"
            out.at[idx, "span_visibility_class"] = "FULL_SPAN"
            out.at[idx, "full_span_visible"] = True
            out.at[idx, "span_view_confidence"] = conf
            out.at[idx, "span_view_pair"] = pair_id
            out.at[idx, "span_view_from_reference"] = int(a["reference_idx"])
            out.at[idx, "span_view_to_reference"] = int(b["reference_idx"])
            out.at[idx, "span_view_from_structure"] = a["structure_id"]
            out.at[idx, "span_view_to_structure"] = b["structure_id"]
            out.at[idx, "span_view_pair_matches_primary"] = str(r.get("primary_span")) == pair_id
            out.at[idx, "span_view_uses_outside_tolerance"] = bool(uses_tol)
            out.at[idx, "valid_for_163"] = True
            out.at[idx, "sequence_keep"] = True
            out.at[idx, "sequence_only_image"] = False
        else:
            if edge_pairs:
                status = "STRUCTURE_NEAR_EDGE"
            elif len(by_position) == 1:
                status = "SINGLE_STRUCTURE_GROUP"
            elif len(by_position) == 0:
                status = "NO_STRUCTURE_GROUP"
            else:
                status = "NO_CONSECUTIVE_STRUCTURE_PAIR"
            out.at[idx, "span_view_status"] = status
            if status == "SINGLE_STRUCTURE_GROUP":
                visibility = "SINGLE_STRUCTURE"
            elif status == "NO_STRUCTURE_GROUP":
                visibility = "NO_STRUCTURE_VISIBLE"
            else:
                visibility = "PARTIAL_SPAN"
            out.at[idx, "span_visibility_class"] = visibility
            out.at[idx, "full_span_visible"] = False
            out.at[idx, "span_view_confidence"] = "NONE"
            out.at[idx, "span_view_rejection_reason"] = f"VISIBILITY_QA:{status}"
            # Keep the image in the continuous product. It is only a partial view.
            out.at[idx, "valid_for_163"] = True
            out.at[idx, "sequence_keep"] = True
            out.at[idx, "sequence_only_image"] = False

    return out


def span_completeness_table(images_df: pd.DataFrame) -> pd.DataFrame:
    if images_df is None or images_df.empty:
        return pd.DataFrame()
    mask = images_df.get("valid_for_product", pd.Series(False, index=images_df.index)).eq(True)
    cols = [
        "filename", "raw_path", "pass_id", "pass_direction", "circuit_run_id",
        "product_status", "coverage_status", "coverage_source", "coverage_intervals",
        "coverage_from", "coverage_to", "primary_span", "primary_span_method",
        "reference_chainage", "reference_assignment_status", "reference_confidence",
        "span_view_status", "span_view_confidence", "span_view_candidate_structure_count",
        "span_view_valid_pair_count", "span_view_pair", "span_view_from_reference",
        "span_view_to_reference", "span_view_from_structure", "span_view_to_structure",
        "span_view_pair_matches_primary", "span_view_uses_outside_tolerance",
        "span_view_evidence", "span_view_rejection_reason", "span_view_outside_tolerance",
        "span_view_edge_margin", "valid_for_product", "valid_for_163", "sequence_only_image",
    ]
    return images_df.loc[mask, [c for c in cols if c in images_df.columns]].copy().reset_index(drop=True)


def _infer_capture_dates_from_benchmark(
    manifest: pd.DataFrame,
    image_benchmark: pd.DataFrame,
    override_code: str | None,
) -> pd.DataFrame:
    out = manifest.copy()
    if override_code:
        out["capture_date_code"] = override_code
        out["capture_date"] = datetime.strptime(override_code, "%m%d%y").date().isoformat()
        out["capture_date_source"] = "USER_OVERRIDE"
        return out

    raw_map: dict[str, str] = {}
    source_modes: dict[str, str] = {}
    global_mode = None

    if image_benchmark is not None and not image_benchmark.empty:
        b = image_benchmark.copy()
        if "source_match_confidence" in b:
            b = b[b["source_match_confidence"].isin(["HIGH", "MEDIUM"])]
        if "capture_date" in b and "matched_raw_filename" in b:
            b["capture_date_code"] = b["capture_date"].map(_date_code_from_iso)
            b = b[b["capture_date_code"].notna()]
            if len(b):
                # A raw image may be used more than once historically. Mode keeps the
                # inference deterministic without forcing a 1:1 relationship.
                for fn, grp in b.groupby("matched_raw_filename"):
                    mode = grp["capture_date_code"].mode()
                    if len(mode):
                        raw_map[str(fn)] = str(mode.iloc[0])
                gm = b["capture_date_code"].mode()
                if len(gm):
                    global_mode = str(gm.iloc[0])

                if "eo_source" in out.columns:
                    raw_source = out[["filename", "eo_source"]].dropna().drop_duplicates("filename")
                    bb = b.merge(raw_source, left_on="matched_raw_filename", right_on="filename", how="left")
                    for src, grp in bb.dropna(subset=["eo_source"]).groupby("eo_source"):
                        mode = grp["capture_date_code"].mode()
                        if len(mode):
                            source_modes[str(src)] = str(mode.iloc[0])

    codes = []
    sources = []
    dates = []
    for _, r in out.iterrows():
        fn = str(r.get("filename"))
        src = str(r.get("eo_source")) if pd.notna(r.get("eo_source")) else None
        if fn in raw_map:
            code = raw_map[fn]
            source = "MATCHED_EXISTING_163"
        elif src and src in source_modes:
            code = source_modes[src]
            source = "EO_SOURCE_MODE_EXISTING_163"
        elif global_mode:
            code = global_mode
            source = "GLOBAL_MODE_EXISTING_163"
        else:
            code = "000000"
            source = "UNKNOWN"
        codes.append(code)
        sources.append(source)
        try:
            dates.append(datetime.strptime(code, "%m%d%y").date().isoformat())
        except ValueError:
            dates.append(None)
    out["capture_date_code"] = codes
    out["capture_date"] = dates
    out["capture_date_source"] = sources
    return out



def _parse_exif_datetime(value: object) -> tuple[str | None, str | None]:
    """Return (MMDDYY, ISO date) from a common EXIF datetime value."""
    if value is None:
        return None, None
    text = str(value).strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text[:19] if "%H" in fmt else text[:10], fmt)
            return dt.strftime("%m%d%y"), dt.date().isoformat()
        except ValueError:
            continue
    return None, None


def _raw_exif_capture_date(path_value: object) -> tuple[str | None, str | None, str | None]:
    """Read capture date from raw-image metadata without decoding the full raster."""
    try:
        from PIL import Image
    except ImportError:
        return None, None, None
    try:
        path = Path(str(path_value))
        if not path.exists():
            return None, None, None
        with Image.open(path) as im:
            exif = im.getexif()
            # DateTimeOriginal, DateTimeDigitized, DateTime.
            for tag, source in ((36867, "RAW_EXIF_DATETIME_ORIGINAL"), (36868, "RAW_EXIF_DATETIME_DIGITIZED"), (306, "RAW_EXIF_DATETIME")):
                code, iso = _parse_exif_datetime(exif.get(tag))
                if code:
                    return code, iso, source
    except Exception:
        return None, None, None
    return None, None, None


def _infer_capture_dates_from_raw(manifest: pd.DataFrame, override_code: str | None) -> pd.DataFrame:
    """Infer output dates from the raw imagery only.

    Existing 163 is deliberately not consulted.  A user override has precedence,
    then per-image raw EXIF, then an EO-source mode derived from raw EXIF.  If no
    independent date is available the code remains 000000 rather than borrowing a
    historical product label.
    """
    out = manifest.copy()
    if override_code:
        out["capture_date_code"] = override_code
        out["capture_date"] = datetime.strptime(override_code, "%m%d%y").date().isoformat()
        out["capture_date_source"] = "USER_OVERRIDE"
        return out

    by_file: dict[str, tuple[str, str, str]] = {}
    for _, r in out[[c for c in ["filename", "raw_path"] if c in out.columns]].drop_duplicates("filename").iterrows():
        code, iso, source = _raw_exif_capture_date(r.get("raw_path"))
        if code and iso and source:
            by_file[str(r.get("filename"))] = (code, iso, source)

    source_modes: dict[str, str] = {}
    if by_file and "eo_source" in out.columns:
        tmp = out[["filename", "eo_source"]].copy()
        tmp["_code"] = tmp["filename"].astype(str).map(lambda x: by_file.get(x, (None, None, None))[0])
        for src, grp in tmp.dropna(subset=["eo_source", "_code"]).groupby("eo_source"):
            mode = grp["_code"].mode()
            if len(mode):
                source_modes[str(src)] = str(mode.iloc[0])

    codes: list[str] = []
    dates: list[str | None] = []
    sources: list[str] = []
    for _, r in out.iterrows():
        fn = str(r.get("filename"))
        src = str(r.get("eo_source")) if pd.notna(r.get("eo_source")) else None
        if fn in by_file:
            code, iso, source = by_file[fn]
        elif src and src in source_modes:
            code = source_modes[src]
            try:
                iso = datetime.strptime(code, "%m%d%y").date().isoformat()
            except ValueError:
                iso = None
            source = "EO_SOURCE_MODE_RAW_EXIF"
        else:
            code, iso, source = "000000", None, "UNKNOWN_INDEPENDENT_SOURCE"
        codes.append(code)
        dates.append(iso)
        sources.append(source)
    out["capture_date_code"] = codes
    out["capture_date"] = dates
    out["capture_date_source"] = sources
    return out

def build_product_manifest(
    images_df: pd.DataFrame,
    image_benchmark: pd.DataFrame,
    circuit: str,
    reference_mode: str,
    capture_date_code: str | None,
    output_version: str,
    output_order: str,
    output_id_start: int,
    require_span_complete: bool = False,
) -> pd.DataFrame:
    """Build a deterministic manifest for the legal continuous-oblique sequence.

    v0.7 deliberately does not filter on full-span visibility. Partial-span and
    no-structure frames remain when they belong to the valid circuit run, because
    they may be required to preserve visual continuity.
    """
    eligibility = "sequence_keep" if "sequence_keep" in images_df.columns else "valid_for_product"
    valid = images_df[images_df[eligibility].eq(True)].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["product_direction"] = valid["pass_direction"].map(
        {"FORWARD_PASS": "FORWARD", "BACKWARD_PASS": "BACKWARD"}
    ).fillna("OTHER")

    # Product sequence is circuit-aware. Acquisition sequence is retained separately.
    valid["acquisition_index"] = pd.to_numeric(valid.get("eo_order"), errors="coerce")
    valid["chainage_sort"] = pd.to_numeric(valid.get("reference_chainage"), errors="coerce")
    valid["sequence_in_run"] = np.nan
    valid["sequence_in_direction"] = np.nan

    for rid, grp in valid.groupby("circuit_run_id", dropna=False):
        direction = str(grp["pass_direction"].dropna().iloc[0]) if grp["pass_direction"].notna().any() else ""
        asc = direction != "BACKWARD_PASS"
        order = grp.sort_values(["chainage_sort", "acquisition_index"], ascending=[asc, True]).index
        valid.loc[order, "sequence_in_run"] = np.arange(1, len(order) + 1)

    for direction, grp in valid.groupby("product_direction", dropna=False):
        asc = direction != "BACKWARD"
        order = grp.sort_values(["chainage_sort", "acquisition_index"], ascending=[asc, True]).index
        valid.loc[order, "sequence_in_direction"] = np.arange(1, len(order) + 1)

    valid = _infer_capture_dates_from_raw(valid, capture_date_code)

    if output_order == "acquisition":
        sort_cols = ["eo_source", "acquisition_index", "filename"]
        valid = valid.sort_values(sort_cols, na_position="last").copy()
    elif output_order == "chainage":
        # FORWARD then BACKWARD, each in the expected travel direction.
        parts = []
        for d in ["FORWARD", "BACKWARD", "OTHER"]:
            g = valid[valid["product_direction"].eq(d)].copy()
            if g.empty:
                continue
            asc = d != "BACKWARD"
            parts.append(g.sort_values(["chainage_sort", "acquisition_index"], ascending=[asc, True]))
        valid = pd.concat(parts, ignore_index=False) if parts else valid
    else:
        raise ValueError(f"Unsupported output order: {output_order}")

    valid["generated_output_id"] = np.arange(int(output_id_start), int(output_id_start) + len(valid))
    valid["output_version"] = output_version

    refs = valid["reference_id"].where(valid["reference_id"].notna(), valid.get("primary_span"))
    generated_names = []
    for ref, oid, code in zip(refs, valid["generated_output_id"], valid["capture_date_code"]):
        ref_token = _clean_token(ref, "UNASSIGNED")
        generated_names.append(
            f"{_clean_token(circuit)}_{ref_token}_{int(oid)}_{code}_{_clean_token(output_version, 'v1')}.jpg"
        )
    valid["generated_filename"] = generated_names
    valid["render_status"] = "NOT_RUN"
    valid["rendered_path"] = None
    valid["render_width"] = np.nan
    valid["render_height"] = np.nan
    return valid.reset_index(drop=True)


def build_product_candidates_layer(manifest: pd.DataFrame, crs: CRS) -> gpd.GeoDataFrame:
    if manifest is None or manifest.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    out = manifest.copy()
    mask = pd.to_numeric(out.get("camera_x"), errors="coerce").notna() & pd.to_numeric(out.get("camera_y"), errors="coerce").notna()
    out = out[mask].copy()
    geom = [Point(float(x), float(y)) for x, y in zip(out["camera_x"], out["camera_y"])]
    return gpd.GeoDataFrame(out, geometry=geom, crs=crs)


def build_structure_crop_candidates(
    structure_obs: gpd.GeoDataFrame,
    structures: gpd.GeoDataFrame,
    images_df: pd.DataFrame,
) -> pd.DataFrame:
    if structure_obs is None or structure_obs.empty:
        return pd.DataFrame()
    refs = structures[structures["is_reference_structure"].eq(True)].copy()
    refcols = [
        "structure_idx", "reference_idx", "structure_id", "x", "y", "centerline_z",
        "structure_height", "chainage", "structure_type", "structure_description",
    ]
    refmeta = refs[[c for c in refcols if c in refs.columns]].drop_duplicates("structure_idx")

    obs = pd.DataFrame(structure_obs.drop(columns="geometry", errors="ignore")).copy()
    obs = obs.merge(refmeta, on=["structure_idx", "structure_id"], how="inner", suffixes=("", "_ref"))
    imgcols = [
        "filename", "raw_path", "dxf_source", "camera_x", "camera_y", "camera_z",
        "omega", "phi", "kappa", "optical_target_x", "optical_target_y", "optical_target_z",
        "optical_target_chainage", "primary_span", "primary_structure", "product_status",
        "valid_for_product", "valid_for_163", "sequence_only_image", "rejection_reason",
        "reference_confidence", "reference_assignment_status", "alternative_reference",
        "span_view_status", "span_view_confidence", "span_view_pair",
        "circuit_run_id", "pass_direction", "view_direction",
    ]
    imeta = images_df[[c for c in imgcols if c in images_df.columns]].drop_duplicates("filename")
    # structure_obs already contains some image fields. Prefer fresh image model values.
    obs = obs.drop(columns=[c for c in imeta.columns if c != "filename" and c in obs.columns], errors="ignore")
    obs = obs.merge(imeta, on="filename", how="left")

    roles = []
    buckets = []
    render_default = []
    for _, r in obs.iterrows():
        if bool(r.get("sequence_keep", r.get("valid_for_product"))):
            role = "PRODUCT_VIEW"
            pdirection = str(r.get("pass_direction") or "")
            if pdirection == "FORWARD_PASS":
                bucket = "forward"
            elif pdirection == "BACKWARD_PASS":
                bucket = "backward"
            else:
                bucket = "other_views"
            default = True
        elif bool(r.get("valid_for_product")):
            role = "PRODUCT_VIEW_LEGACY"
            pdirection = str(r.get("pass_direction") or "")
            if pdirection == "FORWARD_PASS":
                bucket = "forward"
            elif pdirection == "BACKWARD_PASS":
                bucket = "backward"
            else:
                bucket = "other_views"
            default = True
        else:
            if str(r.get("pass_direction")) == "CROSSING_PASS":
                role = "QA_CROSSING_VIEW"
            elif str(r.get("view_direction")) == "SIDE":
                role = "QA_SIDE_VIEW"
            else:
                role = "QA_OTHER_VIEW"
            bucket = "other_views"
            default = False
        roles.append(role)
        buckets.append(bucket)
        render_default.append(default)
    obs["crop_role"] = roles
    obs["crop_bucket"] = buckets
    obs["render_by_default"] = render_default
    obs["crop_projection_status"] = "NOT_RUN"
    obs["crop_projection_confidence"] = "NONE"
    obs["source_width"] = np.nan
    obs["source_height"] = np.nan
    for c in ["base_x_px", "base_y_px", "top_x_px", "top_y_px", "crop_xmin", "crop_ymin", "crop_xmax", "crop_ymax", "footprint_corner_residual"]:
        obs[c] = np.nan
    obs["crop_render_status"] = "NOT_RUN"
    obs["crop_path"] = None
    return obs.drop_duplicates(subset=["structure_idx", "filename"]).reset_index(drop=True)


def _opk_rotation_matrix_v05(omega: float, phi: float, kappa: float, convention: str) -> np.ndarray | None:
    if not all(pd.notna(v) and math.isfinite(float(v)) for v in [omega, phi, kappa]):
        return None
    o, p, k = map(math.radians, [float(omega), float(phi), float(kappa)])
    if convention == "STANDARD_KPO":
        return _rz(k) @ _ry(p) @ _rx(o)
    if convention == "FLIP_PHI_KPO":
        return _rz(k) @ _ry(-p) @ _rx(o)
    return None


def _footprint_vertices(poly) -> list[tuple[float, float]]:
    if poly is None or getattr(poly, "is_empty", True):
        return []
    try:
        g = poly.convex_hull
        coords = list(g.exterior.coords)[:-1]
    except Exception:
        return []
    if len(coords) == 4:
        return [(float(x), float(y)) for x, y in coords]
    # Fallback for non-quadrilateral footprint geometry.
    try:
        rr = g.minimum_rotated_rectangle
        coords = list(rr.exterior.coords)[:-1]
        return [(float(x), float(y)) for x, y in coords[:4]]
    except Exception:
        return []


def _normalized_camera_xy(
    point_xyz: tuple[float, float, float],
    camera_xyz: tuple[float, float, float],
    rotation: np.ndarray,
) -> tuple[float, float] | None:
    d = np.array(point_xyz, dtype=float) - np.array(camera_xyz, dtype=float)
    q = rotation.T @ d
    denom = -float(q[2])
    if denom <= 1e-9 or not math.isfinite(denom):
        return None
    return float(q[0] / denom), float(q[1] / denom)


def estimate_footprint_pixel_model(
    footprint,
    width: int,
    height: int,
    camera_xyz: tuple[float, float, float],
    omega: float,
    phi: float,
    kappa: float,
    ground_z: float,
    convention: str,
) -> dict | None:
    """Estimate image-normalized intrinsics from EO + the delivered ground footprint.

    This is not camera calibration. It converts the footprint FOV extents in camera
    coordinates into an approximate pixel model so structure crops can be generated
    for review. Exact calibration can replace it later without changing the crop schema.
    """
    verts = _footprint_vertices(footprint)
    if len(verts) < 4 or not math.isfinite(float(ground_z)):
        return None
    rot = _opk_rotation_matrix_v05(omega, phi, kappa, convention)
    if rot is None:
        return None
    norms = []
    for x, y in verts:
        uv = _normalized_camera_xy((x, y, float(ground_z)), camera_xyz, rot)
        if uv is None:
            return None
        norms.append(uv)
    xs = np.array([u for u, _ in norms], dtype=float)
    ys = np.array([v for _, v in norms], dtype=float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax - xmin <= 1e-9 or ymax - ymin <= 1e-9:
        return None

    # Pixel y increases downward while conventional photogrammetric camera y is up.
    pix = []
    for nx, ny in norms:
        u = (nx - xmin) / (xmax - xmin) * float(width)
        v = (ymax - ny) / (ymax - ymin) * float(height)
        pix.append((u, v))
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=float)
    diag = max(math.hypot(width, height), 1.0)
    residuals = []
    for u, v in pix:
        residuals.append(float(np.min(np.linalg.norm(corners - np.array([u, v]), axis=1)) / diag))
    residual = float(np.mean(residuals))
    confidence = "HIGH" if residual <= 0.08 else ("MEDIUM" if residual <= 0.18 else "LOW")
    return {
        "rotation": rot,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": int(width),
        "height": int(height),
        "corner_residual": residual,
        "confidence": confidence,
    }


def project_xyz_to_pixel(
    point_xyz: tuple[float, float, float],
    camera_xyz: tuple[float, float, float],
    model: dict,
) -> tuple[float, float] | None:
    norm = _normalized_camera_xy(point_xyz, camera_xyz, model["rotation"])
    if norm is None:
        return None
    nx, ny = norm
    u = (nx - model["xmin"]) / (model["xmax"] - model["xmin"]) * float(model["width"])
    v = (model["ymax"] - ny) / (model["ymax"] - model["ymin"]) * float(model["height"])
    return float(u), float(v)


def _structure_crop_box(
    base_px: tuple[float, float],
    top_px: tuple[float, float],
    width: int,
    height: int,
    padding_factor: float,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    bx, by = base_px
    tx, ty = top_px
    if not all(math.isfinite(v) for v in [bx, by, tx, ty]):
        return None
    projected_h = max(math.hypot(tx - bx, ty - by), 1.0)
    pad = max(float(min_size) * 0.5, projected_h * float(padding_factor))
    xmin = min(bx, tx) - pad
    xmax = max(bx, tx) + pad
    ymin = min(by, ty) - pad
    ymax = max(by, ty) + pad
    # Enforce a useful minimum square around the projected structure centre.
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = max(xmax - xmin, float(min_size))
    bh = max(ymax - ymin, float(min_size))
    xmin, xmax = cx - bw / 2.0, cx + bw / 2.0
    ymin, ymax = cy - bh / 2.0, cy + bh / 2.0
    x0 = max(0, int(math.floor(xmin)))
    y0 = max(0, int(math.floor(ymin)))
    x1 = min(int(width), int(math.ceil(xmax)))
    y1 = min(int(height), int(math.ceil(ymax)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def _load_annotation_font(size: int):
    from PIL import ImageFont
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate_product_image(im, reference_mode: str, reference_id: object, circuit: str, date_code: object):
    from PIL import ImageDraw
    w, h = im.size
    font_size = max(14, int(round(min(w, h) * 0.012)))
    font = _load_annotation_font(font_size)
    label = "Span ID" if reference_mode == "span" else "Structure ID"
    lines = [
        f"{label}: {reference_id if reference_id is not None and pd.notna(reference_id) else 'UNASSIGNED'}",
        f"Circuit: {circuit}",
        f"Date: {_date_display_from_code(date_code)}",
    ]
    draw = ImageDraw.Draw(im)
    line_boxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
    widths = [b[2] - b[0] for b in line_boxes]
    heights = [b[3] - b[1] for b in line_boxes]
    pad = max(5, int(round(font_size * 0.45)))
    spacing = max(2, int(round(font_size * 0.15)))
    box_w = max(widths) + pad * 2
    box_h = sum(heights) + spacing * (len(lines) - 1) + pad * 2
    x0 = max(0, w - box_w)
    y0 = max(0, h - box_h)
    draw.rectangle([x0, y0, w, h], fill=(255, 255, 255))
    y = y0 + pad
    for t, th in zip(lines, heights):
        draw.text((x0 + pad, y), t, fill=(0, 0, 0), font=font)
        y += th + spacing
    return im



def render_span_completeness_review(
    review_manifest: pd.DataFrame,
    out_dir: Path,
    circuit: str,
    reference_mode: str,
    preview_max_dimension: int,
    jpeg_quality: int,
    review_limit: int | None,
) -> pd.DataFrame:
    """Render a compact FULL/PARTIAL span-visibility review set.

    Both classes remain in the product sequence. This is visibility QA only.
    """
    if review_manifest is None or review_manifest.empty or review_limit is None or int(review_limit) <= 0:
        return pd.DataFrame()

    from PIL import Image, ImageDraw, ImageOps
    Image.MAX_IMAGE_PIXELS = None

    review = review_manifest.head(int(review_limit)).copy().reset_index(drop=True)
    review["span_review_render_status"] = "NOT_RUN"
    review["span_review_path"] = None
    root = out_dir / "qa" / "span_completeness"
    full_dir = root / "full_span"
    partial_dir = root / "partial_span"
    full_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)

    for i, r in review.iterrows():
        src = Path(str(r.get("raw_path")))
        if not src.exists():
            review.at[i, "span_review_render_status"] = "SOURCE_MISSING"
            continue
        try:
            with Image.open(src) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
                if preview_max_dimension and max(im.size) > int(preview_max_dimension):
                    im.thumbnail((int(preview_max_dimension), int(preview_max_dimension)), Image.Resampling.LANCZOS)
                annotate_product_image(
                    im,
                    reference_mode,
                    r.get("reference_id") if pd.notna(r.get("reference_id")) else r.get("primary_span"),
                    circuit,
                    r.get("capture_date_code"),
                )
                passed = bool(r.get("full_span_visible"))
                status = str(r.get("span_view_status") or "UNKNOWN")
                pair = str(r.get("span_view_pair")) if pd.notna(r.get("span_view_pair")) else "-"
                label = "FULL" if passed else "PARTIAL"
                visibility = str(r.get("span_visibility_class") or "UNKNOWN")
                lines = [f"Span visibility: {label}", f"Class: {visibility}", f"Status: {status}", f"Evidence pair: {pair}"]
                draw = ImageDraw.Draw(im)
                font = _load_annotation_font(max(18, int(round(im.size[0] / 140))))
                pad = max(8, int(round(im.size[0] / 500)))
                spacing = max(2, int(round(im.size[0] / 1500)))
                boxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
                w = max(b[2] - b[0] for b in boxes) + 2 * pad
                h = sum(b[3] - b[1] for b in boxes) + spacing * (len(lines) - 1) + 2 * pad
                draw.rectangle((0, 0, w, h), fill="white")
                y = pad
                for t, b in zip(lines, boxes):
                    draw.text((pad, y), t, fill="black", font=font)
                    y += (b[3] - b[1]) + spacing
                dest_dir = full_dir if passed else partial_dir
                name = f"{label}_{Path(str(r.get('generated_filename'))).name}"
                dest = dest_dir / name
                im.save(dest, "JPEG", quality=int(jpeg_quality), optimize=True)
                review.at[i, "span_review_render_status"] = "RENDERED"
                review.at[i, "span_review_path"] = str(dest)
        except Exception as exc:
            review.at[i, "span_review_render_status"] = f"ERROR:{type(exc).__name__}"
    return review


def render_candidate_products_and_crops(
    manifest: pd.DataFrame,
    crop_candidates: pd.DataFrame,
    footprints: gpd.GeoDataFrame,
    out_dir: Path,
    circuit: str,
    reference_mode: str,
    opk_convention: str,
    render_mode: str,
    preview_max_dimension: int,
    jpeg_quality: int,
    crop_padding_factor: float,
    crop_min_size: int,
    crop_max_dimension: int,
    render_other_views: bool,
    skip_structure_crops: bool,
    render_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render review-ready candidate 163 JPGs and per-structure crops.

    Source imagery is opened read-only. All outputs are new files under out_dir.
    """
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None

    man = manifest.copy()
    crops = crop_candidates.copy()
    if man.empty:
        return man, crops
    if render_mode == "none" and skip_structure_crops:
        man["render_status"] = "NOT_RENDERED"
        if not crops.empty:
            crops["crop_render_status"] = "NOT_RENDERED"
        return man, crops

    fp_lookup = {str(r.filename): r.geometry for _, r in footprints.iterrows() if pd.notna(r.filename)}
    product_root = out_dir / "163_candidate"
    crop_root = out_dir / "structure_crops"
    product_root.mkdir(parents=True, exist_ok=True)
    if not skip_structure_crops:
        crop_root.mkdir(parents=True, exist_ok=True)

    renderable = man.copy()
    if render_limit is not None:
        renderable = renderable.head(max(0, int(render_limit)))
    render_names = set(renderable["filename"].astype(str))

    # Render product images and crops in one source-image pass to avoid decoding a
    # very large aerial JPG twice.
    for filename, mg in man.groupby("filename", sort=False):
        if str(filename) not in render_names and render_mode != "none":
            continue
        src = Path(str(mg.iloc[0].get("raw_path")))
        midx = mg.index.tolist()
        if not src.exists():
            man.loc[midx, "render_status"] = "SOURCE_MISSING"
            if not crops.empty:
                crops.loc[crops["filename"].astype(str).eq(str(filename)), "crop_render_status"] = "SOURCE_MISSING"
            continue

        try:
            with Image.open(src) as im0:
                im = ImageOps.exif_transpose(im0).convert("RGB")
                sw, sh = im.size
                man.loc[midx, "source_width"] = sw
                man.loc[midx, "source_height"] = sh

                # Structure crops first, while the image remains full-resolution.
                if not skip_structure_crops and not crops.empty:
                    cg = crops[crops["filename"].astype(str).eq(str(filename))]
                    for ci, cr in cg.iterrows():
                        if not bool(cr.get("render_by_default")) and not render_other_views:
                            crops.at[ci, "crop_render_status"] = "NOT_RENDERED_OTHER_VIEW"
                            continue
                        footprint = fp_lookup.get(str(filename))
                        if footprint is None:
                            crops.at[ci, "crop_projection_status"] = "NO_FOOTPRINT"
                            crops.at[ci, "crop_render_status"] = "NO_FOOTPRINT"
                            continue
                        vals = [cr.get("camera_x"), cr.get("camera_y"), cr.get("camera_z"), cr.get("omega"), cr.get("phi"), cr.get("kappa")]
                        if not all(pd.notna(v) and math.isfinite(float(v)) for v in vals):
                            crops.at[ci, "crop_projection_status"] = "NO_EO"
                            crops.at[ci, "crop_render_status"] = "NO_EO"
                            continue
                        base_z = float(cr.get("centerline_z")) if pd.notna(cr.get("centerline_z")) else np.nan
                        if not math.isfinite(base_z):
                            base_z = float(cr.get("optical_target_z")) if pd.notna(cr.get("optical_target_z")) else np.nan
                        ground_z = float(cr.get("optical_target_z")) if pd.notna(cr.get("optical_target_z")) else base_z
                        camera_xyz = (float(cr.camera_x), float(cr.camera_y), float(cr.camera_z))
                        model = estimate_footprint_pixel_model(
                            footprint, sw, sh, camera_xyz,
                            float(cr.omega), float(cr.phi), float(cr.kappa),
                            ground_z, opk_convention,
                        )
                        if model is None:
                            crops.at[ci, "crop_projection_status"] = "PIXEL_MODEL_FAILED"
                            crops.at[ci, "crop_render_status"] = "PIXEL_MODEL_FAILED"
                            continue
                        height_proxy = float(cr.get("structure_height")) if pd.notna(cr.get("structure_height")) else 0.0
                        if not math.isfinite(height_proxy) or height_proxy <= 0:
                            height_proxy = 60.0 if sw > 0 else 20.0
                        sx = float(cr.get("x"))
                        sy = float(cr.get("y"))
                        base = project_xyz_to_pixel((sx, sy, base_z), camera_xyz, model)
                        top = project_xyz_to_pixel((sx, sy, base_z + height_proxy), camera_xyz, model)
                        crops.at[ci, "source_width"] = sw
                        crops.at[ci, "source_height"] = sh
                        crops.at[ci, "footprint_corner_residual"] = model["corner_residual"]
                        crops.at[ci, "crop_projection_confidence"] = model["confidence"]
                        if base is None or top is None:
                            crops.at[ci, "crop_projection_status"] = "POINT_BEHIND_CAMERA"
                            crops.at[ci, "crop_render_status"] = "POINT_BEHIND_CAMERA"
                            continue
                        crops.at[ci, "base_x_px"], crops.at[ci, "base_y_px"] = base
                        crops.at[ci, "top_x_px"], crops.at[ci, "top_y_px"] = top
                        box = _structure_crop_box(base, top, sw, sh, crop_padding_factor, crop_min_size)
                        if box is None:
                            crops.at[ci, "crop_projection_status"] = "OUTSIDE_IMAGE"
                            crops.at[ci, "crop_render_status"] = "OUTSIDE_IMAGE"
                            continue
                        x0, y0, x1, y1 = box
                        crops.at[ci, "crop_xmin"] = x0
                        crops.at[ci, "crop_ymin"] = y0
                        crops.at[ci, "crop_xmax"] = x1
                        crops.at[ci, "crop_ymax"] = y1
                        crops.at[ci, "crop_projection_status"] = "OK_APPROX"

                        crop_im = im.crop(box)
                        if crop_max_dimension and max(crop_im.size) > int(crop_max_dimension):
                            crop_im.thumbnail((int(crop_max_dimension), int(crop_max_dimension)), Image.Resampling.LANCZOS)
                        refidx = int(cr.reference_idx) if pd.notna(cr.get("reference_idx")) else int(cr.structure_idx)
                        structure_folder = f"{refidx:03d}_{_clean_token(cr.get('structure_id'), 'structure')}"
                        bucket = _clean_token(cr.get("crop_bucket"), "other_views")
                        dest_dir = crop_root / structure_folder / bucket
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / f"{Path(str(filename)).stem}.jpg"
                        crop_im.save(dest, "JPEG", quality=int(jpeg_quality), optimize=True)
                        crops.at[ci, "crop_path"] = str(dest)
                        crops.at[ci, "crop_render_status"] = "RENDERED"

                if render_mode == "none":
                    man.loc[midx, "render_status"] = "NOT_RENDERED"
                    continue

                # There should be one manifest row per raw filename, but loop keeps
                # behavior deterministic if future products deliberately duplicate it.
                for mi in midx:
                    row = man.loc[mi]
                    direction = str(row.get("product_direction") or "other").lower()
                    dest_dir = product_root / direction
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / str(row["generated_filename"])
                    if render_mode == "preview":
                        out_im = im.copy()
                        maxdim = max(256, int(preview_max_dimension))
                        if max(out_im.size) > maxdim:
                            out_im.thumbnail((maxdim, maxdim), Image.Resampling.LANCZOS)
                    else:
                        out_im = im.copy()
                    annotate_product_image(
                        out_im, reference_mode, row.get("reference_id"), circuit, row.get("capture_date_code")
                    )
                    out_im.save(dest, "JPEG", quality=int(jpeg_quality), optimize=True)
                    man.at[mi, "render_status"] = "RENDERED_PREVIEW" if render_mode == "preview" else "RENDERED_FULL"
                    man.at[mi, "rendered_path"] = str(dest)
                    man.at[mi, "render_width"] = out_im.size[0]
                    man.at[mi, "render_height"] = out_im.size[1]
        except Exception as exc:
            man.loc[midx, "render_status"] = f"ERROR:{type(exc).__name__}"
            if not crops.empty:
                sel = crops["filename"].astype(str).eq(str(filename)) & crops["crop_render_status"].eq("NOT_RUN")
                crops.loc[sel, "crop_render_status"] = f"ERROR:{type(exc).__name__}"

    # Rows intentionally excluded by render-limit are explicit in the manifest.
    if render_limit is not None:
        sel = ~man["filename"].astype(str).isin(render_names) & man["render_status"].eq("NOT_RUN")
        man.loc[sel, "render_status"] = "NOT_RENDERED_LIMIT"
    return man, crops


def build_crop_candidates_layer(crops: pd.DataFrame, crs: CRS) -> gpd.GeoDataFrame:
    if crops is None or crops.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    mask = pd.to_numeric(crops.get("x"), errors="coerce").notna() & pd.to_numeric(crops.get("y"), errors="coerce").notna()
    out = crops[mask].copy()
    geom = [Point(float(x), float(y)) for x, y in zip(out["x"], out["y"])]
    return gpd.GeoDataFrame(out, geometry=geom, crs=crs)


def load_reusable_image_benchmark(root: Path) -> pd.DataFrame:
    for version in ["v04", "v03", "v02"]:
        p = root / f"_continuous_oblique_poc_{version}" / "existing_163_benchmark.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                required = {"matched_raw_filename", "source_match_confidence", "capture_date"}
                if required.issubset(df.columns):
                    return df
            except Exception:
                pass
    return pd.DataFrame()

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
    parser = argparse.ArgumentParser(description="Geometry-first continuous oblique imagery POC v0.4")
    parser.add_argument("project_root", type=Path, help="Project root, e.g. D:\\Sev\\Obliques\\NYPA")
    parser.add_argument("--circuit", default=None, help="Circuit label. Default: infer from existing 163/staking filename")
    parser.add_argument("--epsg", type=int, default=None, help="Override working EPSG if PLS zone cannot be mapped")
    parser.add_argument("--reference-mode", choices=["span", "structure"], default="span")
    parser.add_argument(
        "--reference-topology",
        choices=["auto", "tow-ordinal", "identity"],
        default="auto",
        help=(
            "How to convert physical PLS staking rows into product/reference structures. "
            "auto uses the last #N token from structure_name when it forms a contiguous monotonic sequence; "
            "identity treats every PLS row as a reference structure. Default: auto"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <root>\\_continuous_oblique_poc_v04")
    parser.add_argument("--max-eo-distance", type=float, default=None, help="Max camera-to-alignment distance in working units")
    parser.add_argument("--gap-tolerance", type=float, default=None, help="Ignore gaps smaller than this, working units")
    parser.add_argument("--min-overlap", type=float, default=None, help="Transition overlap below this is WEAK_OVERLAP")
    parser.add_argument("--structure-tolerance", type=float, default=None, help="Buffer around structure when testing footprint visibility")
    parser.add_argument("--optical-max-distance", type=float, default=None, help="Max projected optical-target distance from alignment before falling back to coverage midpoint")
    parser.add_argument(
        "--opk-convention",
        choices=["STANDARD_KPO", "FLIP_PHI_KPO"],
        default="STANDARD_KPO",
        help="Approximate OPK optical-axis convention used by v0.4. Default: STANDARD_KPO",
    )
    parser.add_argument(
        "--recovery-max-neighbor-gap",
        type=int,
        default=10,
        help="Maximum EO-order distance used to recover a missing footprint from neighbouring confirmed images. Default: 10",
    )
    parser.add_argument("--disable-footprint-recovery", action="store_true", help="Do not infer provisional coverage for images missing DXF footprints")
    parser.add_argument(
        "--run-stitch-gap",
        type=float,
        default=None,
        help="Max directional chainage gap used to stitch interrupted flight-pass fragments into one circuit run. Default: 3x median image coverage length.",
    )
    parser.add_argument("--allow-side", action="store_true", help="Allow SIDE images as product candidates")
    parser.add_argument(
        "--skip-image-benchmark",
        action="store_true",
        help="Skip appearance-based matching between existing 163 JPGs and raw JPGs",
    )
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    if not root.exists():
        parser.error(f"Project root does not exist: {root}")

    paths = discover_project(root)
    validate_required_paths(paths)

    out_dir = args.output.resolve() if args.output else root / "_continuous_oblique_poc_v04"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "continuous_oblique_poc_v04.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    print("[1/11] Parsing staking and building alignment...")
    structures, pls_spans, reference_structures, spans, alignment_gdf, metadata = parse_staking_xml(
        paths.staking_xml, args.epsg, args.reference_topology
    )
    alignment = alignment_gdf.iloc[0].geometry
    crs = metadata["crs"]
    defaults = unit_defaults(crs)
    max_eo_distance = args.max_eo_distance if args.max_eo_distance is not None else defaults["max_eo_distance"]
    gap_tolerance = args.gap_tolerance if args.gap_tolerance is not None else defaults["gap_tolerance"]
    min_overlap = args.min_overlap if args.min_overlap is not None else defaults["min_overlap"]
    structure_tolerance = args.structure_tolerance if args.structure_tolerance is not None else defaults["structure_tolerance"]
    optical_max_distance = args.optical_max_distance if args.optical_max_distance is not None else defaults["optical_max_distance"]

    print("[2/11] Indexing raw images and existing 163...")
    raw = index_raw_images(paths.raw_dir)
    existing = index_existing_163(paths.final_163_dir)
    circuit = args.circuit or infer_circuit(existing, paths.staking_xml)

    print("[3/11] Parsing image footprints...")
    footprints = load_footprints(paths.footprint_dxfs, crs)

    print("[4/11] Parsing EO and detecting flight passes...")
    eo = load_all_eo(paths.eo_dir)
    if eo.empty:
        raise ValueError("No EO records parsed")
    eo = project_eo_to_alignment(eo, alignment)
    eo, eo_summary = eo_file_compatibility(eo, max_eo_distance)
    eo, passes_df = classify_passes(eo)

    print("[5/11] Building confirmed coverage and OPK optical targeting...")
    images_df, _, _ = build_image_model(
        raw, eo, footprints, structures, spans, alignment, crs,
        args.reference_mode, args.allow_side, optical_max_distance, args.opk_convention,
    )

    print("[6/11] Recovering missing-footprint coverage...")
    if args.disable_footprint_recovery:
        recovery = pd.DataFrame()
    else:
        images_df, recovery = recover_missing_footprint_coverage(
            images_df, spans, structures, alignment,
            args.reference_mode, args.allow_side, args.recovery_max_neighbor_gap,
        )

    run_stitch_gap = args.run_stitch_gap if args.run_stitch_gap is not None else suggested_run_stitch_gap(images_df)

    print("[7/11] Stitching logical circuit runs...")
    images_df, runs_gdf, pass_to_run = build_circuit_runs(
        images_df, alignment, crs, circuit, run_stitch_gap, gap_tolerance
    )
    if not passes_df.empty:
        passes_df = passes_df.copy()
        passes_df["circuit_run_id"] = passes_df["pass_id"].map(pass_to_run)
    passes_gdf = (
        gpd.GeoDataFrame(passes_df, geometry="geometry", crs=crs)
        if not passes_df.empty
        else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    )

    # Build fresh spatial layers after recovery/run assignment so every v0.4 field is current.
    images_gdf = build_images_layer(images_df, crs)
    coverage_gdf = build_coverage_layer(images_df, alignment, crs)

    print("[8/11] Building structure observations...")
    structure_obs = build_structure_observations(
        images_df, footprints, structures, alignment, crs, structure_tolerance
    )

    print("[9/11] Checking transitions, directional continuity and gap types...")
    flight_transitions = build_transitions(images_df, crs, gap_tolerance, min_overlap)
    run_transitions = build_run_transitions(images_df, crs, gap_tolerance, min_overlap)
    run_internal_gaps = build_run_internal_gaps(images_df, runs_gdf, alignment, crs, gap_tolerance)
    direction_coverage, direction_coverage_confirmed, direction_gaps = build_direction_coverage(
        images_df, alignment, crs, gap_tolerance
    )
    direction_gaps = classify_direction_gaps(direction_gaps, images_df, existing, spans, gap_tolerance)
    run_transitions = classify_transition_gaps(run_transitions, images_df, gap_tolerance)
    gap_events = build_gap_events(direction_gaps, run_transitions, alignment, crs)

    print("[10/11] Benchmarking span attribution against existing 163...")
    span_benchmark = build_span_benchmark(spans, images_df, existing, circuit)
    if args.skip_image_benchmark:
        image_benchmark = pd.DataFrame()
    else:
        image_benchmark = build_existing_163_image_benchmark(raw, existing, images_df, circuit)
    opk_benchmark = build_opk_benchmark_summary(image_benchmark)
    benchmark_points = benchmark_points_layer(image_benchmark, images_df, crs)

    print("[11/11] Writing GeoPackage and CSV QA outputs...")
    layers = [
        (structures, "structures"),
        (reference_structures, "reference_structures"),
        (pls_spans, "pls_spans"),
        (spans, "spans"),
        (alignment_gdf, "alignment"),
        (footprints, "image_footprints"),
        (passes_gdf, "flight_passes"),
        (runs_gdf, "circuit_runs"),
        (images_gdf, "images"),
        (coverage_gdf, "image_coverage"),
        (structure_obs, "structure_observations"),
        (flight_transitions, "flight_transitions"),
        (run_transitions, "run_transitions"),
        (visual_transitions, "visual_transitions"),
        (run_internal_gaps, "run_internal_gaps"),
        (direction_coverage, "direction_coverage"),
        (direction_coverage_confirmed, "direction_coverage_confirmed"),
        (direction_gaps, "direction_gaps"),
        (gap_events, "gap_events"),
        (span_benchmark, "span_benchmark"),
        (benchmark_points, "benchmark_163"),
    ]
    first = True
    for gdf, layer in layers:
        if gdf is not None and not gdf.empty:
            write_gpkg_layer(gdf, gpkg, layer, first=first)
            first = False

    safe_csv(raw, out_dir / "raw_images.csv")
    safe_csv(structures, out_dir / "reference_topology.csv")
    safe_csv(pls_spans, out_dir / "pls_spans.csv")
    safe_csv(spans, out_dir / "reference_spans.csv")
    safe_csv(eo, out_dir / "eo_index.csv")
    safe_csv(eo_summary, out_dir / "eo_sources.csv")
    safe_csv(images_df, out_dir / "image_index.csv")
    safe_csv(recovery, out_dir / "footprint_recovery.csv")
    safe_csv(passes_df, out_dir / "flight_passes.csv")
    safe_csv(runs_gdf, out_dir / "circuit_runs.csv")
    safe_csv(flight_transitions, out_dir / "flight_transitions.csv")
    safe_csv(run_transitions, out_dir / "transitions.csv")
    safe_csv(visual_transitions, out_dir / "visual_transitions.csv")
    safe_csv(visual_transitions, out_dir / "product_visual_transitions.csv")
    safe_csv(dino_feature_index, out_dir / "dino_feature_index.csv")
    safe_csv(visual_review, out_dir / "visual_review_manifest.csv")
    safe_csv(run_internal_gaps, out_dir / "run_internal_gaps.csv")
    safe_csv(direction_coverage, out_dir / "direction_coverage.csv")
    safe_csv(direction_coverage_confirmed, out_dir / "direction_coverage_confirmed.csv")
    safe_csv(direction_gaps, out_dir / "direction_gaps.csv")
    safe_csv(gap_events, out_dir / "gap_events.csv")
    safe_csv(structure_obs, out_dir / "structure_observations.csv")
    safe_csv(span_benchmark, out_dir / "span_benchmark.csv")
    safe_csv(opk_benchmark, out_dir / "opk_benchmark.csv")
    if not existing.empty:
        safe_csv(existing, out_dir / "existing_163_index.csv")
    if not image_benchmark.empty:
        safe_csv(image_benchmark, out_dir / "existing_163_benchmark.csv")

    summary = build_summary_v03(
        paths, metadata, raw, footprints, eo, eo_summary, structures, spans,
        passes_df, runs_gdf, images_df, recovery, run_transitions, run_internal_gaps,
        direction_coverage, direction_gaps, gap_events, existing, span_benchmark,
        image_benchmark, opk_benchmark, circuit,
    )
    summary["settings"] = {
        "reference_mode": args.reference_mode,
        "reference_topology": args.reference_topology,
        "reference_topology_status": metadata.get("reference_topology_status"),
        "allow_side": args.allow_side,
        "max_eo_distance": max_eo_distance,
        "gap_tolerance": gap_tolerance,
        "min_overlap": min_overlap,
        "structure_tolerance": structure_tolerance,
        "optical_max_distance": optical_max_distance,
        "opk_convention": args.opk_convention,
        "footprint_recovery_enabled": not args.disable_footprint_recovery,
        "recovery_max_neighbor_gap": args.recovery_max_neighbor_gap,
        "run_stitch_gap": run_stitch_gap,
        "image_benchmark_enabled": not args.skip_image_benchmark,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    valid = images_df[images_df["valid_for_product"].eq(True)] if len(images_df) else pd.DataFrame()
    confirmed_n = int(valid.get("product_status", pd.Series(index=valid.index)).eq("CONFIRMED").sum()) if len(valid) else 0
    provisional_n = int(valid.get("product_status", pd.Series(index=valid.index)).eq("PROVISIONAL").sum()) if len(valid) else 0

    print()
    print(f"Circuit:              {circuit}")
    print(f"CRS:                  EPSG:{metadata['epsg']}")
    print(f"PLS structures:       {len(structures)}")
    print(f"Reference structures: {len(reference_structures)}")
    print(f"PLS spans:            {len(pls_spans)}")
    print(f"Reference spans:      {len(spans)}")
    print(f"Intermediate PLS:     {metadata.get('intermediate_pls_structure_count', 0)}")
    print(f"Reference topology:   {metadata.get('reference_topology_status')}")
    print(f"Raw images:           {len(raw)}")
    print(f"Footprints:           {len(footprints)}")
    print(f"EO records:           {len(eo)}")
    print(f"Flight passes:        {len(passes_df)}")
    print(f"Circuit runs:         {len(runs_gdf)}")
    print(f"Valid candidates:     {len(valid)} (confirmed={confirmed_n}, provisional={provisional_n})")
    print(f"Recovered footprints: {len(recovery)}")
    print(f"Existing 163 images:  {len(existing)}")
    print(f"Run stitch gap:       {run_stitch_gap:.1f}")
    print(f"OPK convention:       {args.opk_convention}")
    print(f"Output:               {out_dir}")

    if not eo_summary.empty:
        print("\nEO source compatibility:")
        for _, r in eo_summary.iterrows():
            print(
                f"  {r['eo_source']}: records={int(r['records'])}, "
                f"min_distance={r['min_distance_to_alignment']:.1f}, "
                f"compatible={bool(r['spatial_compatible'])}"
            )

    if not runs_gdf.empty:
        print("\nCircuit runs:")
        for _, r in runs_gdf.iterrows():
            pct = 100.0 * float(r["coverage_ratio_within_run"]) if pd.notna(r["coverage_ratio_within_run"]) else float("nan")
            print(
                f"  {r['circuit_run_id']}: {r['run_direction']}, passes={int(r['pass_count'])}, "
                f"valid={int(r['valid_image_count'])}, coverage={r['coverage_from']:.1f}-{r['coverage_to']:.1f}, "
                f"internal_gaps={int(r['internal_gap_count'])}, within_run={pct:.2f}%"
            )

    if not direction_coverage.empty:
        print("\nDirection coverage:")
        for _, r in direction_coverage.iterrows():
            dg = direction_gaps[direction_gaps["direction"].eq(r["direction"])] if not direction_gaps.empty else pd.DataFrame()
            gap_counts = dict(Counter(dg["gap_type"])) if len(dg) else {}
            print(
                f"  {r['direction']}: images={int(r['valid_images'])} "
                f"(confirmed={int(r['confirmed_images'])}, provisional={int(r['provisional_images'])}), "
                f"coverage={r['coverage_percent']:.2f}% "
                f"[confirmed={r['confirmed_coverage_percent']:.2f}%, +provisional={r['provisional_gain_percent']:.2f}%], "
                f"gaps={len(dg)} {gap_counts}"
            )

    if not image_benchmark.empty:
        counts = image_benchmark["source_match_confidence"].value_counts()
        comparable = image_benchmark[
            image_benchmark["source_match_confidence"].isin(["HIGH", "MEDIUM"])
            & image_benchmark["generated_primary_span"].notna()
        ]
        matches = int(comparable["benchmark_status"].eq("SPAN_MATCH").sum()) if len(comparable) else 0
        pct = 100.0 * matches / len(comparable) if len(comparable) else float("nan")
        print("\nExisting 163 image benchmark:")
        print(
            f"  source matches: HIGH={int(counts.get('HIGH', 0))}, "
            f"MEDIUM={int(counts.get('MEDIUM', 0))}, LOW={int(counts.get('LOW', 0))}"
        )
        print(f"  selected primary span: comparable={len(comparable)}, exact={matches}, exact_match={pct:.2f}%")
        if not opk_benchmark.empty:
            for _, r in opk_benchmark.iterrows():
                print(
                    f"  {r['target_method']}: comparable={int(r['comparable_images'])}, "
                    f"exact={int(r['exact_span_matches'])}, exact_match={r['exact_span_match_percent']:.2f}%"
                )

    print("\nNo source imagery was modified.")
    return 0


# -----------------------------------------------------------------------------
# Main v0.6
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# v0.7 DINOv3 visual-continuity QA
# -----------------------------------------------------------------------------


def _safe_feature_key(filename: str, raw_path: str, model_id: str, image_size: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)[:80]
    digest = hashlib.sha1(f"{raw_path}|{model_id}|{image_size}".encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{stem}_{digest}.npz"


def _load_dino_input_image(path: Path, image_size: int):
    """Decode a very large JPEG cheaply, then resize to the DINO working grid."""
    from PIL import Image

    with Image.open(path) as src:
        try:
            # JPEG draft asks libjpeg for a lower-resolution decode where possible.
            src.draft("RGB", (max(image_size * 2, 512), max(image_size * 2, 512)))
        except Exception:
            pass
        im = src.convert("RGB")
        # DINOv3 ViT accepts arbitrary patch-aligned spatial sizes.  For a stable
        # pairwise QA grid we use a fixed square working image; source imagery is
        # close to square and the metric is comparative, not a mapping product.
        im = im.resize((int(image_size), int(image_size)), Image.Resampling.LANCZOS)
        return im


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _dino_pair_metrics(a: dict, b: dict, match_threshold: float) -> dict:
    ga = _normalise_rows(np.asarray(a["global"])[None, :])[0]
    gb = _normalise_rows(np.asarray(b["global"])[None, :])[0]
    pa = _normalise_rows(np.asarray(a["patches"]))
    pb = _normalise_rows(np.asarray(b["patches"]))

    global_similarity = float(np.dot(ga, gb))
    if len(pa) == 0 or len(pb) == 0:
        return {
            "global_similarity": global_similarity,
            "dense_best_mean": np.nan,
            "mutual_match_count": 0,
            "mutual_match_ratio": 0.0,
            "mutual_similarity_mean": np.nan,
            "mutual_similarity_median": np.nan,
            "mutual_similarity_p10": np.nan,
            "spatial_coherence": np.nan,
            "median_patch_dx": np.nan,
            "median_patch_dy": np.nan,
        }

    sim = pa @ pb.T
    a_to_b = np.argmax(sim, axis=1)
    b_to_a = np.argmax(sim, axis=0)
    a_best = sim[np.arange(len(pa)), a_to_b]
    b_best = sim[b_to_a, np.arange(len(pb))]
    dense_best_mean = float((float(np.mean(a_best)) + float(np.mean(b_best))) / 2.0)

    ai = np.arange(len(pa), dtype=np.int64)
    reciprocal = b_to_a[a_to_b] == ai
    mutual_i = ai[reciprocal]
    mutual_j = a_to_b[reciprocal]
    mutual_scores = sim[mutual_i, mutual_j] if len(mutual_i) else np.asarray([], dtype=np.float32)
    strong = mutual_scores >= float(match_threshold)
    mutual_i = mutual_i[strong]
    mutual_j = mutual_j[strong]
    mutual_scores = mutual_scores[strong]
    mutual_count = int(len(mutual_scores))
    mutual_ratio = float(mutual_count / max(1, min(len(pa), len(pb))))

    coherence = np.nan
    med_dx = np.nan
    med_dy = np.nan
    if mutual_count >= 3:
        ha, wa = a["grid"]
        hb, wb = b["grid"]
        ya, xa = np.divmod(mutual_i, int(wa))
        yb, xb = np.divmod(mutual_j, int(wb))
        # Express target coordinates in source-grid units so this remains valid
        # if a future run uses rectangular or slightly different patch grids.
        xbn = xb.astype(np.float32) * (float(wa) / max(float(wb), 1.0))
        ybn = yb.astype(np.float32) * (float(ha) / max(float(hb), 1.0))
        dx = xbn - xa.astype(np.float32)
        dy = ybn - ya.astype(np.float32)
        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        residual = np.hypot(dx - med_dx, dy - med_dy)
        # 2.5 patch cells is intentionally permissive for oblique perspective.
        coherence = float(np.mean(residual <= 2.5))

    return {
        "global_similarity": global_similarity,
        "dense_best_mean": dense_best_mean,
        "mutual_match_count": mutual_count,
        "mutual_match_ratio": mutual_ratio,
        "mutual_similarity_mean": float(np.mean(mutual_scores)) if mutual_count else np.nan,
        "mutual_similarity_median": float(np.median(mutual_scores)) if mutual_count else np.nan,
        "mutual_similarity_p10": float(np.quantile(mutual_scores, 0.10)) if mutual_count else np.nan,
        "spatial_coherence": coherence,
        "median_patch_dx": med_dx,
        "median_patch_dy": med_dy,
    }


def _load_dinov3_model(model_id: str, device_arg: str):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 requires torch and transformers. Install requirements_continuous_oblique_v07.txt first."
        ) from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available to PyTorch")

    token = os.environ.get("HF_TOKEN") or None
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the DINOv3 model. The official Meta DINOv3 Hugging Face checkpoints are gated. "
            "Accept the model licence, authenticate with `hf auth login` (or set HF_TOKEN), or pass --dino-model to a local checkpoint path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    model.to(device)
    return torch, processor, model, device


def extract_dinov3_features(
    manifest: pd.DataFrame,
    needed_filenames: set[str],
    out_dir: Path,
    model_id: str,
    image_size: int,
    batch_size: int,
    device_arg: str,
    use_cache: bool,
) -> tuple[dict[str, dict], pd.DataFrame, dict]:
    if image_size < 112:
        raise ValueError("--dino-image-size must be >=112")

    rows = manifest[manifest["filename"].astype(str).isin(needed_filenames)].drop_duplicates("filename").copy()
    raw_map = {str(r["filename"]): Path(str(r["raw_path"])) for _, r in rows.iterrows()}
    cache_dir = out_dir / "qa" / "visual_continuity" / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    features: dict[str, dict] = {}
    feature_rows: list[dict] = []
    pending: list[tuple[str, Path, Path]] = []

    for fn, path in raw_map.items():
        cache = cache_dir / _safe_feature_key(fn, str(path), model_id, int(image_size))
        if use_cache and cache.exists() and path.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                if int(z["source_size"]) == int(path.stat().st_size) and abs(float(z["source_mtime"]) - float(path.stat().st_mtime)) < 1.0:
                    feat = {
                        "global": z["global_embedding"].astype(np.float32),
                        "patches": z["patch_features"].astype(np.float32),
                        "grid": tuple(int(x) for x in z["grid"]),
                    }
                    features[fn] = feat
                    feature_rows.append({"filename": fn, "feature_status": "CACHE", "patch_count": len(feat["patches"]), "feature_cache": str(cache)})
                    continue
            except Exception:
                pass
        pending.append((fn, path, cache))

    model_meta = {"model_id": model_id, "device": None, "image_size": int(image_size), "cached": len(features), "computed": 0}
    if not pending:
        return features, pd.DataFrame(feature_rows), model_meta

    torch, processor, model, device = _load_dinov3_model(model_id, device_arg)
    model_meta["device"] = device
    patch_size = getattr(model.config, "patch_size", 16)
    if isinstance(patch_size, (list, tuple)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    num_register = int(getattr(model.config, "num_register_tokens", 0) or 0)
    model_meta["patch_size"] = [patch_h, patch_w]
    model_meta["num_register_tokens"] = num_register

    for start in range(0, len(pending), max(1, int(batch_size))):
        chunk = pending[start:start + max(1, int(batch_size))]
        images = []
        valid_chunk = []
        for fn, path, cache in chunk:
            if not path.exists():
                feature_rows.append({"filename": fn, "feature_status": "SOURCE_MISSING", "patch_count": 0, "feature_cache": str(cache)})
                continue
            try:
                images.append(_load_dino_input_image(path, int(image_size)))
                valid_chunk.append((fn, path, cache))
            except Exception as exc:
                feature_rows.append({"filename": fn, "feature_status": f"IMAGE_ERROR:{type(exc).__name__}", "patch_count": 0, "feature_cache": str(cache)})
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state.detach().float().cpu().numpy()
        pixel_values = inputs["pixel_values"]
        gh = int(pixel_values.shape[-2]) // patch_h
        gw = int(pixel_values.shape[-1]) // patch_w
        patch_count = gh * gw

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            pooled_np = pooled.detach().float().cpu().numpy()
        else:
            pooled_np = hidden[:, 0, :]

        for bi, (fn, path, cache) in enumerate(valid_chunk):
            patches = hidden[bi, 1 + num_register:1 + num_register + patch_count, :]
            feat = {
                "global": pooled_np[bi].astype(np.float32),
                "patches": patches.astype(np.float32),
                "grid": (gh, gw),
            }
            features[fn] = feat
            if use_cache:
                np.savez_compressed(
                    cache,
                    global_embedding=feat["global"].astype(np.float16),
                    patch_features=feat["patches"].astype(np.float16),
                    grid=np.asarray(feat["grid"], dtype=np.int16),
                    source_size=np.asarray(path.stat().st_size, dtype=np.int64),
                    source_mtime=np.asarray(path.stat().st_mtime, dtype=np.float64),
                )
            feature_rows.append({"filename": fn, "feature_status": "COMPUTED", "patch_count": len(patches), "feature_cache": str(cache)})
            model_meta["computed"] += 1

    return features, pd.DataFrame(feature_rows), model_meta


def _robust_visual_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = float(x.median()) if x.notna().any() else np.nan
    mad = float((x - med).abs().median()) if x.notna().any() else np.nan
    if not math.isfinite(mad) or mad < 1e-6:
        return pd.Series(0.0, index=values.index)
    return (x - med) / (1.4826 * mad)


def build_dinov3_visual_transitions(
    product_manifest: pd.DataFrame,
    run_transitions: gpd.GeoDataFrame,
    out_dir: Path,
    model_id: str,
    image_size: int,
    batch_size: int,
    device_arg: str,
    match_threshold: float,
    max_transitions: int | None,
    use_cache: bool,
    duplicate_global: float,
    duplicate_dense: float,
    absolute_weak: float,
    robust_weak_z: float,
    robust_review_z: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    if run_transitions is None or run_transitions.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(run_transitions, "crs", None)), pd.DataFrame(), {"analyzed": 0}

    transitions = run_transitions.copy().reset_index(drop=True)
    transitions["visual_analysis_status"] = "NOT_ANALYZED"
    selected_idx = list(transitions.index)
    if max_transitions is not None:
        selected_idx = selected_idx[:max(0, int(max_transitions))]
        transitions.loc[~transitions.index.isin(selected_idx), "visual_analysis_status"] = "NOT_ANALYZED_LIMIT"

    needed: set[str] = set()
    for i in selected_idx:
        needed.add(str(transitions.at[i, "image_a"]))
        needed.add(str(transitions.at[i, "image_b"]))

    features, feature_index, model_meta = extract_dinov3_features(
        product_manifest, needed, out_dir, model_id, image_size, batch_size, device_arg, use_cache,
    )

    metric_cols = [
        "global_similarity", "dense_best_mean", "mutual_match_count", "mutual_match_ratio",
        "mutual_similarity_mean", "mutual_similarity_median", "mutual_similarity_p10",
        "spatial_coherence", "median_patch_dx", "median_patch_dy", "visual_score",
    ]
    for c in metric_cols:
        transitions[c] = np.nan

    for i in selected_idx:
        a = str(transitions.at[i, "image_a"])
        b = str(transitions.at[i, "image_b"])
        if a not in features or b not in features:
            transitions.at[i, "visual_analysis_status"] = "FEATURE_MISSING"
            continue
        metrics = _dino_pair_metrics(features[a], features[b], match_threshold)
        for k, v in metrics.items():
            transitions.at[i, k] = v
        g = float(metrics.get("global_similarity", np.nan))
        d = float(metrics.get("dense_best_mean", np.nan))
        coh = float(metrics.get("spatial_coherence", np.nan))
        if not math.isfinite(coh):
            coh = 0.0
        if math.isfinite(g) and math.isfinite(d):
            transitions.at[i, "visual_score"] = 0.45 * g + 0.35 * d + 0.20 * coh
            transitions.at[i, "visual_analysis_status"] = "ANALYZED"
        else:
            transitions.at[i, "visual_analysis_status"] = "METRIC_FAILED"

    transitions["visual_robust_z"] = np.nan
    analyzed_mask = transitions["visual_analysis_status"].eq("ANALYZED")
    for run_id, grp in transitions[analyzed_mask].groupby("circuit_run_id", sort=False):
        transitions.loc[grp.index, "visual_robust_z"] = _robust_visual_z(grp["visual_score"])

    transitions["visual_status"] = transitions["visual_analysis_status"]
    transitions["review_reason"] = None
    for i in transitions.index:
        if transitions.at[i, "visual_analysis_status"] != "ANALYZED":
            continue
        score = float(transitions.at[i, "visual_score"])
        z = float(transitions.at[i, "visual_robust_z"]) if pd.notna(transitions.at[i, "visual_robust_z"]) else 0.0
        glob = float(transitions.at[i, "global_similarity"])
        dense = float(transitions.at[i, "dense_best_mean"])
        geometry_status = str(transitions.at[i, "transition_status"])
        bridge = bool(transitions.at[i, "bridge_between_passes"]) if pd.notna(transitions.at[i, "bridge_between_passes"]) else False

        possible_duplicate = glob >= duplicate_global and dense >= duplicate_dense
        visual_weak = score < absolute_weak or z <= robust_weak_z
        visual_review = visual_weak or z <= robust_review_z
        geometry_hard = geometry_status in {"HARD_GAP", "MISSING_COVERAGE", "SEQUENCE_JUMP"}
        geometry_weak = geometry_status == "WEAK_OVERLAP"

        if possible_duplicate:
            status, reason = "POSSIBLE_DUPLICATE", "Very high DINO global and dense similarity"
        elif geometry_hard and visual_weak:
            status, reason = "POSSIBLE_GAP", f"Geometry={geometry_status} and DINO continuity is weak"
        elif geometry_hard and not visual_review:
            status, reason = "GEOMETRY_WEAK_VISUAL_OK", f"Geometry={geometry_status}, but DINO continuity is normal"
        elif visual_weak:
            status, reason = "GEOMETRY_OK_VISUAL_WEAK", "Geometry is acceptable but DINO pair is a strong visual outlier"
        elif bridge and visual_review:
            status, reason = "VIEW_CHANGE", "Transition bridges flight-pass fragments and DINO score is below normal"
        elif geometry_weak and not visual_review:
            status, reason = "GEOMETRY_WEAK_VISUAL_OK", "Footprint overlap is weak, but DINO continuity is normal"
        elif visual_review:
            status, reason = "WEAK_VISUAL_OVERLAP", "DINO score is below the run distribution"
        else:
            status, reason = "CONTINUOUS", "Geometry and DINO continuity are consistent"
        transitions.at[i, "visual_status"] = status
        transitions.at[i, "review_reason"] = reason

    transitions["qa_bucket"] = transitions["visual_status"].map({
        "CONTINUOUS": "OK",
        "POSSIBLE_DUPLICATE": "DUPLICATE",
        "POSSIBLE_GAP": "GAP",
        "GEOMETRY_OK_VISUAL_WEAK": "GAP",
        "GEOMETRY_WEAK_VISUAL_OK": "REVIEW",
        "WEAK_VISUAL_OVERLAP": "REVIEW",
        "VIEW_CHANGE": "REVIEW",
    }).fillna("REVIEW")

    meta = dict(model_meta)
    meta.update({
        "analyzed": int(transitions["visual_analysis_status"].eq("ANALYZED").sum()),
        "requested_transitions": int(len(selected_idx)),
        "total_transitions": int(len(transitions)),
        "status_counts": dict(Counter(transitions["visual_status"])),
        "qa_bucket_counts": dict(Counter(transitions["qa_bucket"])),
        "match_threshold": float(match_threshold),
        "duplicate_global": float(duplicate_global),
        "duplicate_dense": float(duplicate_dense),
        "absolute_weak": float(absolute_weak),
        "robust_weak_z": float(robust_weak_z),
        "robust_review_z": float(robust_review_z),
        "membership_effect": "QA_ONLY_NEVER_REMOVES_IMAGES",
    })
    return gpd.GeoDataFrame(transitions, geometry="geometry", crs=run_transitions.crs), feature_index, meta


def _contact_sheet_image(path: Path, max_w: int = 900, max_h: int = 700):
    from PIL import Image
    with Image.open(path) as src:
        try:
            src.draft("RGB", (max_w * 2, max_h * 2))
        except Exception:
            pass
        im = src.convert("RGB")
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return im.copy()


def render_visual_transition_review(
    visual_transitions: pd.DataFrame,
    product_manifest: pd.DataFrame,
    out_dir: Path,
    limit: int,
    jpeg_quality: int,
) -> pd.DataFrame:
    from PIL import Image, ImageDraw

    if visual_transitions is None or visual_transitions.empty or limit == 0:
        return pd.DataFrame()
    raw_map = dict(zip(product_manifest["filename"].astype(str), product_manifest["raw_path"].astype(str)))
    todo = visual_transitions[~visual_transitions["qa_bucket"].eq("OK")].copy()
    todo = todo.head(max(0, int(limit)))
    rows = []
    root = out_dir / "qa" / "visual_continuity"
    for _, r in todo.iterrows():
        a, b = str(r.get("image_a")), str(r.get("image_b"))
        pa, pb = Path(raw_map.get(a, "")), Path(raw_map.get(b, ""))
        status = str(r.get("visual_status"))
        bucket = str(r.get("qa_bucket"))
        dest_dir = root / bucket.lower()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+','_', status)}_{Path(a).stem}__{Path(b).stem}.jpg"
        rec = {"image_a": a, "image_b": b, "visual_status": status, "qa_bucket": bucket, "review_path": str(dest), "review_render_status": "NOT_RUN"}
        try:
            if not pa.exists() or not pb.exists():
                rec["review_render_status"] = "SOURCE_MISSING"
                rows.append(rec)
                continue
            ia = _contact_sheet_image(pa)
            ib = _contact_sheet_image(pb)
            header_h = 150
            gap = 12
            w = ia.width + ib.width + gap
            h = header_h + max(ia.height, ib.height)
            canvas = Image.new("RGB", (w, h), "white")
            canvas.paste(ia, (0, header_h))
            canvas.paste(ib, (ia.width + gap, header_h))
            draw = ImageDraw.Draw(canvas)
            score = r.get("visual_score")
            gsim = r.get("global_similarity")
            dense = r.get("dense_best_mean")
            overlap = r.get("coverage_overlap")
            gapv = r.get("coverage_gap")
            text_lines = [
                f"{status} | {r.get('circuit_run_id')}",
                f"A: {a}   [{r.get('reference_a')}]",
                f"B: {b}   [{r.get('reference_b')}]",
                f"visual={score:.3f} global={gsim:.3f} dense={dense:.3f}  geom_overlap={overlap:.1f} geom_gap={gapv:.1f}",
                str(r.get("review_reason") or ""),
            ]
            y = 8
            for line in text_lines:
                draw.text((10, y), line, fill="black")
                y += 26
            canvas.save(dest, "JPEG", quality=int(jpeg_quality), optimize=True)
            rec["review_render_status"] = "RENDERED"
        except Exception as exc:
            rec["review_render_status"] = f"ERROR:{type(exc).__name__}"
        rows.append(rec)
    return pd.DataFrame(rows)

def main_v07(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous oblique candidate-product POC v0.7")
    parser.add_argument("project_root", type=Path, help="Project root, e.g. D:\\Sev\\Obliques\\NYPA")
    parser.add_argument("--circuit", default=None, help="Circuit label. Default: infer from existing 163/staking filename")
    parser.add_argument("--epsg", type=int, default=None, help="Override working EPSG if PLS zone cannot be mapped")
    parser.add_argument("--reference-mode", choices=["span", "structure"], default="span")
    parser.add_argument(
        "--reference-topology", choices=["auto", "tow-ordinal", "identity"], default="auto",
        help="Physical PLS rows -> product/reference topology. Default: auto",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Default: <root>\\_continuous_oblique_poc_v07")
    parser.add_argument("--max-eo-distance", type=float, default=None)
    parser.add_argument("--gap-tolerance", type=float, default=None)
    parser.add_argument("--min-overlap", type=float, default=None)
    parser.add_argument("--structure-tolerance", type=float, default=None)
    parser.add_argument("--optical-max-distance", type=float, default=None)
    parser.add_argument(
        "--opk-convention", choices=["STANDARD_KPO", "FLIP_PHI_KPO"], default="STANDARD_KPO",
        help="Approximate OPK convention. DH1 benchmark currently supports STANDARD_KPO.",
    )
    parser.add_argument("--recovery-max-neighbor-gap", type=int, default=10)
    parser.add_argument("--disable-footprint-recovery", action="store_true")
    parser.add_argument("--run-stitch-gap", type=float, default=None)
    parser.add_argument("--allow-side", action="store_true")
    parser.add_argument("--skip-image-benchmark", action="store_true")
    parser.add_argument(
        "--rebuild-image-benchmark", action="store_true",
        help="Ignore a compatible previous benchmark and reread all raw/existing JPGs.",
    )

    parser.add_argument(
        "--boundary-distance", type=float, default=None,
        help="Maximum absolute distance from a span boundary to flag BOUNDARY_AMBIGUOUS. Default: 100 ft or 30 m.",
    )
    parser.add_argument(
        "--boundary-fraction", type=float, default=0.15,
        help="Boundary ambiguity threshold as fraction of span length; combined with --boundary-distance using the smaller threshold. Default: 0.15.",
    )

    # v0.7 span-visibility QA settings.  Defaults were chosen from the DH1 smoke
    # review and remain explicit/configurable rather than hidden product rules.
    parser.add_argument(
        "--span-visibility-outside-tolerance", type=float, default=None,
        help="Allow a reference structure this far just outside a DXF coverage edge. Default: 100 ft or 30 m.",
    )
    parser.add_argument(
        "--span-visibility-edge-margin", type=float, default=None,
        help="A structure inside the footprint but closer than this to an along-circuit edge is treated as clipped/near-edge. Default: 100 ft or 30 m.",
    )
    parser.add_argument(
        "--disable-span-completeness-filter", action="store_true",
        help="Deprecated compatibility flag. v0.7 never removes a legal sequence image because a full span is not visible.",
    )
    parser.add_argument(
        "--span-review-limit", type=int, default=None,
        help="Render first N pre-filter sequence frames into qa\\span_completeness full/partial folders. If omitted, --render-limit is reused.",
    )

    parser.add_argument(
        "--capture-date", default=None,
        help="Force acquisition date in MMDDYY, e.g. 042226. Otherwise infer from raw EXIF only; existing 163 is benchmark-only.",
    )
    parser.add_argument("--output-version", default="v1")
    parser.add_argument("--output-id-start", type=int, default=1)
    parser.add_argument("--output-order", choices=["acquisition", "chainage"], default="acquisition")
    parser.add_argument(
        "--render-mode", choices=["preview", "full", "none"], default="preview",
        help="preview writes review-size 163 JPGs; full preserves source pixel dimensions; none writes manifests only. Default: preview.",
    )
    parser.add_argument("--preview-max-dimension", type=int, default=4096)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--render-limit", type=int, default=None, help="Render only first N legal continuous-sequence 163 images for a quick smoke test.")

    parser.add_argument("--skip-structure-crops", action="store_true")
    parser.add_argument("--render-other-views", action="store_true", help="Also render SIDE/CROSSING/non-product structure crops into other_views.")
    parser.add_argument("--crop-padding-factor", type=float, default=1.0)
    parser.add_argument("--crop-min-size", type=int, default=800, help="Minimum crop size in source pixels. Default: 800.")
    parser.add_argument("--crop-max-dimension", type=int, default=1600, help="Downsample saved crops to this maximum dimension. Default: 1600.")

    # v0.7 DINOv3 visual-continuity QA. DINO never changes sequence membership.
    parser.add_argument("--skip-dino", action="store_true", help="Skip DINOv3 visual continuity QA and run geometry/product generation only.")
    parser.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Hugging Face DINOv3 model ID or local checkpoint directory.")
    parser.add_argument("--dino-device", default="auto", help="auto, cpu, cuda or cuda:N. Default: auto.")
    parser.add_argument("--dino-image-size", type=int, default=224, help="Square DINO working image size. Default: 224.")
    parser.add_argument("--dino-batch-size", type=int, default=4, help="DINO inference batch size. Default: 4.")
    parser.add_argument("--dino-max-transitions", type=int, default=None, help="Analyze only first N logical transitions for a smoke test. Default: all.")
    parser.add_argument("--dino-match-threshold", type=float, default=0.55, help="Minimum reciprocal patch cosine similarity counted as a dense match. Default: 0.55.")
    parser.add_argument("--dino-duplicate-global", type=float, default=0.995, help="Global cosine threshold for POSSIBLE_DUPLICATE. Default: 0.995.")
    parser.add_argument("--dino-duplicate-dense", type=float, default=0.95, help="Dense best-match threshold for POSSIBLE_DUPLICATE. Default: 0.95.")
    parser.add_argument("--dino-absolute-weak", type=float, default=0.58, help="Absolute combined visual score below which a pair is weak. Default: 0.58.")
    parser.add_argument("--dino-robust-weak-z", type=float, default=-3.0, help="Per-run robust z threshold for strong visual outlier. Default: -3.0.")
    parser.add_argument("--dino-robust-review-z", type=float, default=-2.0, help="Per-run robust z threshold for review. Default: -2.0.")
    parser.add_argument("--disable-dino-cache", action="store_true", help="Do not reuse/save DINO feature cache under qa\\visual_continuity\\feature_cache.")
    parser.add_argument("--visual-review-limit", type=int, default=50, help="Render up to N non-OK DINO transition contact sheets. Default: 50; 0 disables.")

    args = parser.parse_args(argv)
    if not (0.0 < float(args.boundary_fraction) <= 0.5):
        parser.error("--boundary-fraction must be >0 and <=0.5")
    if not (1 <= int(args.jpeg_quality) <= 100):
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.span_visibility_outside_tolerance is not None and args.span_visibility_outside_tolerance < 0:
        parser.error("--span-visibility-outside-tolerance must be >=0")
    if args.span_visibility_edge_margin is not None and args.span_visibility_edge_margin < 0:
        parser.error("--span-visibility-edge-margin must be >=0")
    if args.dino_image_size < 112:
        parser.error("--dino-image-size must be >=112")
    if args.dino_batch_size < 1:
        parser.error("--dino-batch-size must be >=1")
    if args.dino_max_transitions is not None and args.dino_max_transitions < 0:
        parser.error("--dino-max-transitions must be >=0")
    if not (-1.0 <= args.dino_match_threshold <= 1.0):
        parser.error("--dino-match-threshold must be between -1 and 1")
    capture_date_code = validate_capture_date_code(args.capture_date)

    root = args.project_root.resolve()
    if not root.exists():
        parser.error(f"Project root does not exist: {root}")

    paths = discover_project(root)
    validate_required_paths(paths)

    out_dir = args.output.resolve() if args.output else root / "_continuous_oblique_poc_v07"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "continuous_oblique_poc_v07.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    print("[1/16] Parsing staking and building alignment...")
    structures, pls_spans, reference_structures, spans, alignment_gdf, metadata = parse_staking_xml(
        paths.staking_xml, args.epsg, args.reference_topology
    )
    alignment = alignment_gdf.iloc[0].geometry
    crs = metadata["crs"]
    defaults = unit_defaults(crs)
    max_eo_distance = args.max_eo_distance if args.max_eo_distance is not None else defaults["max_eo_distance"]
    gap_tolerance = args.gap_tolerance if args.gap_tolerance is not None else defaults["gap_tolerance"]
    min_overlap = args.min_overlap if args.min_overlap is not None else defaults["min_overlap"]
    structure_tolerance = args.structure_tolerance if args.structure_tolerance is not None else defaults["structure_tolerance"]
    optical_max_distance = args.optical_max_distance if args.optical_max_distance is not None else defaults["optical_max_distance"]
    unit_name = ""
    try:
        unit_name = (crs.axis_info[0].unit_name or "").lower()
    except Exception:
        pass
    linear_100 = 100.0 if "foot" in unit_name else 30.0
    boundary_distance = args.boundary_distance if args.boundary_distance is not None else linear_100
    span_outside_tolerance = (
        args.span_visibility_outside_tolerance
        if args.span_visibility_outside_tolerance is not None else linear_100
    )
    span_edge_margin = (
        args.span_visibility_edge_margin
        if args.span_visibility_edge_margin is not None else linear_100
    )

    print("[2/16] Indexing raw images and existing 163...")
    raw = index_raw_images(paths.raw_dir)
    existing = index_existing_163(paths.final_163_dir)
    circuit = args.circuit or infer_circuit(existing, paths.staking_xml)

    print("[3/16] Parsing image footprints...")
    footprints = load_footprints(paths.footprint_dxfs, crs)

    print("[4/16] Parsing EO and detecting flight passes...")
    eo = load_all_eo(paths.eo_dir)
    if eo.empty:
        raise ValueError("No EO records parsed")
    eo = project_eo_to_alignment(eo, alignment)
    eo, eo_summary = eo_file_compatibility(eo, max_eo_distance)
    eo, passes_df = classify_passes(eo)

    print("[5/16] Building confirmed coverage and OPK optical targeting...")
    images_df, _, _ = build_image_model(
        raw, eo, footprints, structures, spans, alignment, crs,
        args.reference_mode, args.allow_side, optical_max_distance, args.opk_convention,
    )

    print("[6/16] Recovering missing-footprint coverage...")
    if args.disable_footprint_recovery:
        recovery = pd.DataFrame()
    else:
        images_df, recovery = recover_missing_footprint_coverage(
            images_df, spans, structures, alignment,
            args.reference_mode, args.allow_side, args.recovery_max_neighbor_gap,
        )

    run_stitch_gap = args.run_stitch_gap if args.run_stitch_gap is not None else suggested_run_stitch_gap(images_df)

    print("[7/16] Stitching logical circuit runs...")
    images_df, runs_gdf, pass_to_run = build_circuit_runs(
        images_df, alignment, crs, circuit, run_stitch_gap, gap_tolerance
    )
    if not passes_df.empty:
        passes_df = passes_df.copy()
        passes_df["circuit_run_id"] = passes_df["pass_id"].map(pass_to_run)
    passes_gdf = (
        gpd.GeoDataFrame(passes_df, geometry="geometry", crs=crs)
        if not passes_df.empty
        else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    )

    print("[8/16] Building structure observations...")
    structure_obs = build_structure_observations(
        images_df, footprints, structures, alignment, crs, structure_tolerance
    )

    print("[9/16] Checking transitions, directional continuity and gap types...")
    flight_transitions = build_transitions(images_df, crs, gap_tolerance, min_overlap)
    run_transitions = build_run_transitions(images_df, crs, gap_tolerance, min_overlap)
    run_internal_gaps = build_run_internal_gaps(images_df, runs_gdf, alignment, crs, gap_tolerance)
    direction_coverage, direction_coverage_confirmed, direction_gaps = build_direction_coverage(
        images_df, alignment, crs, gap_tolerance
    )
    direction_gaps = classify_direction_gaps(direction_gaps, images_df, existing, spans, gap_tolerance)
    run_transitions = classify_transition_gaps(run_transitions, images_df, gap_tolerance)
    gap_events = build_gap_events(direction_gaps, run_transitions, alignment, crs)

    print("[10/16] Benchmarking span attribution against existing 163...")
    span_benchmark = build_span_benchmark(spans, images_df, existing, circuit)
    image_benchmark = pd.DataFrame()
    benchmark_source = "SKIPPED"
    if not args.skip_image_benchmark:
        if not args.rebuild_image_benchmark:
            image_benchmark = load_reusable_image_benchmark(root)
            if not image_benchmark.empty:
                benchmark_source = "REUSED_PREVIOUS_POC"
        if image_benchmark.empty:
            image_benchmark = build_existing_163_image_benchmark(raw, existing, images_df, circuit)
            benchmark_source = "REBUILT"
    opk_benchmark = build_opk_benchmark_summary(image_benchmark)
    benchmark_points = benchmark_points_layer(image_benchmark, images_df, crs)

    print("[11/16] Building reference confidence...")
    images_df = augment_reference_confidence(images_df, spans, boundary_distance, args.boundary_fraction)

    print("[12/16] Classifying span visibility and building continuous 163 manifest...")
    images_df = classify_span_completeness(
        images_df, reference_structures, args.reference_mode,
        span_outside_tolerance, span_edge_margin,
    )
    if args.disable_span_completeness_filter:
        # Compatibility only: filtering is disabled by design in v0.7.
        pass

    # Review and product manifests use the same legal continuous sequence. Span
    # visibility is metadata only and never changes product membership.
    review_manifest = build_product_manifest(
        images_df, image_benchmark, circuit, args.reference_mode, capture_date_code,
        args.output_version, args.output_order, args.output_id_start,
        require_span_complete=False,
    )
    product_manifest = build_product_manifest(
        images_df, image_benchmark, circuit, args.reference_mode, capture_date_code,
        args.output_version, args.output_order, args.output_id_start,
        require_span_complete=False,
    )
    crop_candidates = build_structure_crop_candidates(structure_obs, structures, images_df)
    completeness = span_completeness_table(images_df)

    # Product coverage is the coverage of all legal sequence frames. Span visibility
    # cannot reduce it. Keep the legacy *_163 outputs as explicit aliases so existing
    # QA tooling does not break.
    product_images_for_coverage = images_df.copy()
    product_images_for_coverage["valid_for_product"] = product_images_for_coverage.get(
        "sequence_keep", product_images_for_coverage["valid_for_product"]
    ).eq(True)
    direction_coverage_163, _, direction_gaps_163 = build_direction_coverage(
        product_images_for_coverage, alignment, crs, gap_tolerance
    )

    print("[13/16] Running DINOv3 visual-continuity QA...")
    if args.skip_dino:
        visual_transitions = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
        dino_feature_index = pd.DataFrame()
        visual_meta = {"enabled": False, "membership_effect": "QA_ONLY_NEVER_REMOVES_IMAGES"}
        visual_review = pd.DataFrame()
        print("  DINOv3 skipped by --skip-dino")
    else:
        visual_transitions, dino_feature_index, visual_meta = build_dinov3_visual_transitions(
            product_manifest, run_transitions, out_dir,
            args.dino_model, args.dino_image_size, args.dino_batch_size, args.dino_device,
            args.dino_match_threshold, args.dino_max_transitions, not args.disable_dino_cache,
            args.dino_duplicate_global, args.dino_duplicate_dense, args.dino_absolute_weak,
            args.dino_robust_weak_z, args.dino_robust_review_z,
        )
        visual_meta["enabled"] = True
        visual_review = render_visual_transition_review(
            visual_transitions, product_manifest, out_dir, args.visual_review_limit, args.jpeg_quality
        )
        print(
            f"  analyzed={visual_meta.get('analyzed', 0)}/{visual_meta.get('total_transitions', 0)} "
            f"device={visual_meta.get('device')} statuses={visual_meta.get('status_counts', {})}"
        )

    print("[14/16] Rendering candidate 163, span-view QA and structure crops...")
    if args.render_mode == "none" and args.skip_structure_crops:
        print("  Product/crop rendering disabled; writing manifests only.")
    product_manifest, crop_candidates = render_candidate_products_and_crops(
        product_manifest, crop_candidates, footprints, out_dir, circuit,
        args.reference_mode, args.opk_convention, args.render_mode,
        args.preview_max_dimension, args.jpeg_quality, args.crop_padding_factor,
        args.crop_min_size, args.crop_max_dimension, args.render_other_views,
        args.skip_structure_crops, args.render_limit,
    )
    span_review_limit = args.span_review_limit if args.span_review_limit is not None else args.render_limit
    span_review = render_span_completeness_review(
        review_manifest, out_dir, circuit, args.reference_mode,
        args.preview_max_dimension, args.jpeg_quality, span_review_limit,
    )

    images_gdf = build_images_layer(images_df, crs)
    coverage_gdf = build_coverage_layer(images_df, alignment, crs)
    product_candidates_gdf = build_product_candidates_layer(product_manifest, crs)
    crop_candidates_gdf = build_crop_candidates_layer(crop_candidates, crs)
    span_candidates_gdf = build_product_candidates_layer(review_manifest, crs)

    print("[16/16] Writing GeoPackage and CSV QA outputs...")
    layers = [
        (structures, "structures"),
        (reference_structures, "reference_structures"),
        (pls_spans, "pls_spans"),
        (spans, "spans"),
        (alignment_gdf, "alignment"),
        (footprints, "image_footprints"),
        (passes_gdf, "flight_passes"),
        (runs_gdf, "circuit_runs"),
        (images_gdf, "images"),
        (coverage_gdf, "image_coverage"),
        (structure_obs, "structure_observations"),
        (span_candidates_gdf, "span_completeness"),
        (product_candidates_gdf, "product_candidates"),
        (crop_candidates_gdf, "structure_crop_candidates"),
        (flight_transitions, "flight_transitions"),
        (run_transitions, "run_transitions"),
        (run_internal_gaps, "run_internal_gaps"),
        (direction_coverage, "direction_coverage"),
        (direction_coverage_confirmed, "direction_coverage_confirmed"),
        (direction_gaps, "direction_gaps"),
        (direction_coverage_163, "direction_coverage_163"),
        (direction_gaps_163, "direction_gaps_163"),
        (gap_events, "gap_events"),
        (span_benchmark, "span_benchmark"),
        (benchmark_points, "benchmark_163"),
    ]
    first = True
    for gdf, layer in layers:
        if gdf is not None and not gdf.empty:
            write_gpkg_layer(gdf, gpkg, layer, first=first)
            first = False

    safe_csv(raw, out_dir / "raw_images.csv")
    safe_csv(structures, out_dir / "reference_topology.csv")
    safe_csv(pls_spans, out_dir / "pls_spans.csv")
    safe_csv(spans, out_dir / "reference_spans.csv")
    safe_csv(eo, out_dir / "eo_index.csv")
    safe_csv(eo_summary, out_dir / "eo_sources.csv")
    safe_csv(images_df, out_dir / "image_index.csv")
    safe_csv(recovery, out_dir / "footprint_recovery.csv")
    safe_csv(passes_df, out_dir / "flight_passes.csv")
    safe_csv(runs_gdf, out_dir / "circuit_runs.csv")
    safe_csv(flight_transitions, out_dir / "flight_transitions.csv")
    safe_csv(run_transitions, out_dir / "transitions.csv")
    safe_csv(run_internal_gaps, out_dir / "run_internal_gaps.csv")
    safe_csv(direction_coverage, out_dir / "direction_coverage.csv")
    safe_csv(direction_coverage_confirmed, out_dir / "direction_coverage_confirmed.csv")
    safe_csv(direction_gaps, out_dir / "direction_gaps.csv")
    safe_csv(direction_coverage_163, out_dir / "direction_coverage_163.csv")
    safe_csv(direction_gaps_163, out_dir / "direction_gaps_163.csv")
    safe_csv(gap_events, out_dir / "gap_events.csv")
    safe_csv(structure_obs, out_dir / "structure_observations.csv")
    safe_csv(completeness, out_dir / "span_completeness.csv")
    safe_csv(product_manifest, out_dir / "product_manifest.csv")
    safe_csv(crop_candidates, out_dir / "structure_crop_candidates.csv")
    safe_csv(span_review, out_dir / "span_review_manifest.csv")
    safe_csv(span_benchmark, out_dir / "span_benchmark.csv")
    safe_csv(opk_benchmark, out_dir / "opk_benchmark.csv")
    if not existing.empty:
        safe_csv(existing, out_dir / "existing_163_index.csv")
    if not image_benchmark.empty:
        safe_csv(image_benchmark, out_dir / "existing_163_benchmark.csv")

    summary = build_summary_v03(
        paths, metadata, raw, footprints, eo, eo_summary, structures, spans,
        passes_df, runs_gdf, images_df, recovery, run_transitions, run_internal_gaps,
        direction_coverage, direction_gaps, gap_events, existing, span_benchmark,
        image_benchmark, opk_benchmark, circuit,
    )
    summary["poc_version"] = "0.7"
    rendered_product = int(product_manifest["render_status"].astype(str).str.startswith("RENDERED").sum()) if not product_manifest.empty else 0
    rendered_crops = int(crop_candidates["crop_render_status"].eq("RENDERED").sum()) if not crop_candidates.empty else 0
    geom_candidates = images_df[images_df["valid_for_product"].eq(True)] if len(images_df) else pd.DataFrame()
    final_candidates = images_df[images_df.get("sequence_keep", images_df["valid_for_product"]).eq(True)] if len(images_df) else pd.DataFrame()
    partial_span = geom_candidates[~geom_candidates.get("full_span_visible", pd.Series(False, index=geom_candidates.index)).eq(True)] if len(geom_candidates) else pd.DataFrame()
    summary["counts"].update({
        "geometry_sequence_candidates": int(len(geom_candidates)),
        "span_complete_163_candidates": int(len(final_candidates)),
        "span_view_partial_but_kept": int(len(partial_span)),
        "candidate_product_images": int(len(product_manifest)),
        "rendered_product_images": rendered_product,
        "structure_crop_candidates": int(len(crop_candidates)),
        "rendered_structure_crops": rendered_crops,
    })
    summary["span_completeness"] = {
        "membership_filter_enabled": False,
        "outside_tolerance": float(span_outside_tolerance),
        "edge_margin": float(span_edge_margin),
        "status_counts": dict(Counter(geom_candidates["span_view_status"])) if len(geom_candidates) else {},
        "confidence_counts": dict(Counter(final_candidates["span_view_confidence"])) if len(final_candidates) else {},
        "visibility_counts": dict(Counter(final_candidates["span_visibility_class"])) if len(final_candidates) else {},
        "review_limit": span_review_limit,
        "review_rendered": int(span_review["span_review_render_status"].eq("RENDERED").sum()) if not span_review.empty else 0,
        "model": "REFERENCE_CHAINAGE_PAIR_WITH_FOOTPRINT_EDGE_TOLERANCE",
        "is_pixel_object_detection": False,
    }
    summary["visual_continuity"] = visual_meta
    if visual_transitions is not None and not visual_transitions.empty:
        summary["visual_continuity"]["status_counts"] = dict(Counter(visual_transitions["visual_status"]))
        summary["visual_continuity"]["qa_bucket_counts"] = dict(Counter(visual_transitions["qa_bucket"]))
        summary["visual_continuity"]["review_contact_sheets"] = int(visual_review.get("review_render_status", pd.Series(dtype=str)).eq("RENDERED").sum()) if visual_review is not None and not visual_review.empty else 0
    summary["product_generation"] = {
        "render_mode": args.render_mode,
        "preview_max_dimension": int(args.preview_max_dimension),
        "jpeg_quality": int(args.jpeg_quality),
        "output_order": args.output_order,
        "output_version": args.output_version,
        "capture_date_override": capture_date_code,
        "capture_date_sources": dict(Counter(product_manifest["capture_date_source"])) if not product_manifest.empty else {},
        "reference_confidence": dict(Counter(product_manifest["reference_confidence"])) if not product_manifest.empty else {},
        "reference_assignment_status": dict(Counter(product_manifest["reference_assignment_status"])) if not product_manifest.empty else {},
        "benchmark_source": benchmark_source,
        "structure_crop_model": "APPROX_OPK_PLUS_FOOTPRINT_FOV",
        "structure_crops_are_calibrated": False,
    }
    summary["settings"] = {
        "reference_mode": args.reference_mode,
        "reference_topology": args.reference_topology,
        "reference_topology_status": metadata.get("reference_topology_status"),
        "allow_side": args.allow_side,
        "max_eo_distance": max_eo_distance,
        "gap_tolerance": gap_tolerance,
        "min_overlap": min_overlap,
        "structure_tolerance": structure_tolerance,
        "optical_max_distance": optical_max_distance,
        "opk_convention": args.opk_convention,
        "footprint_recovery_enabled": not args.disable_footprint_recovery,
        "recovery_max_neighbor_gap": args.recovery_max_neighbor_gap,
        "run_stitch_gap": run_stitch_gap,
        "image_benchmark_enabled": not args.skip_image_benchmark,
        "reference_boundary_distance": boundary_distance,
        "reference_boundary_fraction": args.boundary_fraction,
        "span_visibility_outside_tolerance": span_outside_tolerance,
        "span_visibility_edge_margin": span_edge_margin,
        "span_completeness_filter_enabled": False,
        "render_other_views": args.render_other_views,
        "skip_structure_crops": args.skip_structure_crops,
        "crop_padding_factor": args.crop_padding_factor,
        "crop_min_size": args.crop_min_size,
        "crop_max_dimension": args.crop_max_dimension,
        "dino_enabled": not args.skip_dino,
        "dino_model": args.dino_model,
        "dino_device": args.dino_device,
        "dino_image_size": args.dino_image_size,
        "dino_batch_size": args.dino_batch_size,
        "dino_max_transitions": args.dino_max_transitions,
        "dino_match_threshold": args.dino_match_threshold,
        "dino_membership_effect": "QA_ONLY_NEVER_REMOVES_IMAGES",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    confirmed_n = int(geom_candidates.get("product_status", pd.Series(index=geom_candidates.index)).eq("CONFIRMED").sum()) if len(geom_candidates) else 0
    provisional_n = int(geom_candidates.get("product_status", pd.Series(index=geom_candidates.index)).eq("PROVISIONAL").sum()) if len(geom_candidates) else 0

    print()
    print(f"Circuit:              {circuit}")
    print(f"CRS:                  EPSG:{metadata['epsg']}")
    print(f"PLS structures:       {len(structures)}")
    print(f"Reference structures: {len(reference_structures)}")
    print(f"PLS spans:            {len(pls_spans)}")
    print(f"Reference spans:      {len(spans)}")
    print(f"Intermediate PLS:     {metadata.get('intermediate_pls_structure_count', 0)}")
    print(f"Reference topology:   {metadata.get('reference_topology_status')}")
    print(f"Raw images:           {len(raw)}")
    print(f"Footprints:           {len(footprints)}")
    print(f"EO records:           {len(eo)}")
    print(f"Flight passes:        {len(passes_df)}")
    print(f"Circuit runs:         {len(runs_gdf)}")
    print(f"Sequence candidates:  {len(geom_candidates)} (confirmed={confirmed_n}, provisional={provisional_n})")
    print(f"Continuous 163:       {len(final_candidates)} legal sequence images")
    print(f"Partial-span views:   {len(partial_span)} (KEPT in product)")
    print(f"Candidate 163 images: {len(product_manifest)} (rendered={rendered_product}, mode={args.render_mode})")
    print(f"Structure crops:      {len(crop_candidates)} candidates (rendered={rendered_crops})")
    print(f"Existing 163 images:  {len(existing)}")
    print(f"Run stitch gap:       {run_stitch_gap:.1f}")
    print(f"OPK convention:       {args.opk_convention}")
    print(f"Benchmark source:     {benchmark_source}")
    print(f"Output:               {out_dir}")

    if len(geom_candidates):
        print("\nSpan completeness:")
        for k, v in Counter(geom_candidates["span_view_status"]).items():
            print(f"  {k}: {v}")
        print(f"  outside tolerance: {span_outside_tolerance:.1f}")
        print(f"  edge margin:       {span_edge_margin:.1f}")
        if not span_review.empty:
            rv = Counter(span_review["span_view_status"])
            full_n = int(span_review["full_span_visible"].eq(True).sum())
            partial_n = len(span_review) - full_n
            print(f"  smoke review: FULL={full_n}, PARTIAL={partial_n}, statuses={dict(rv)}")

    if not product_manifest.empty:
        print("\nReference confidence (continuous 163):")
        for k, v in Counter(product_manifest["reference_confidence"]).items():
            print(f"  {k}: {v}")
        print("Capture date inference:")
        for k, v in Counter(product_manifest["capture_date_source"]).items():
            print(f"  {k}: {v}")

    if not eo_summary.empty:
        print("\nEO source compatibility:")
        for _, r in eo_summary.iterrows():
            print(
                f"  {r['eo_source']}: records={int(r['records'])}, "
                f"min_distance={r['min_distance_to_alignment']:.1f}, compatible={bool(r['spatial_compatible'])}"
            )

    if not runs_gdf.empty:
        print("\nCircuit runs:")
        for _, r in runs_gdf.iterrows():
            pct = 100.0 * float(r["coverage_ratio_within_run"]) if pd.notna(r["coverage_ratio_within_run"]) else float("nan")
            print(
                f"  {r['circuit_run_id']}: {r['run_direction']}, passes={int(r['pass_count'])}, "
                f"valid={int(r['valid_image_count'])}, coverage={r['coverage_from']:.1f}-{r['coverage_to']:.1f}, "
                f"internal_gaps={int(r['internal_gap_count'])}, within_run={pct:.2f}%"
            )

    if not direction_coverage.empty:
        print("\nSequence/source direction coverage:")
        for _, r in direction_coverage.iterrows():
            dg = direction_gaps[direction_gaps["direction"].eq(r["direction"])] if not direction_gaps.empty else pd.DataFrame()
            gap_counts = dict(Counter(dg["gap_type"])) if len(dg) else {}
            print(
                f"  {r['direction']}: images={int(r['valid_images'])} "
                f"(confirmed={int(r['confirmed_images'])}, provisional={int(r['provisional_images'])}), "
                f"coverage={r['coverage_percent']:.2f}% "
                f"[confirmed={r['confirmed_coverage_percent']:.2f}%, +provisional={r['provisional_gain_percent']:.2f}%], "
                f"gaps={len(dg)} {gap_counts}"
            )
    if not direction_coverage_163.empty:
        print("\nContinuous 163 coverage (all legal sequence images):")
        for _, r in direction_coverage_163.iterrows():
            print(
                f"  {r['direction']}: images={int(r['valid_images'])}, "
                f"coverage={r['coverage_percent']:.2f}%, gaps={int(r['gaps']) if 'gaps' in r else 'n/a'}"
            )

    if not args.skip_dino:
        print("\nDINOv3 visual continuity:")
        print(f"  model:    {args.dino_model}")
        print(f"  device:   {visual_meta.get('device')}")
        print(f"  analyzed: {visual_meta.get('analyzed', 0)}/{visual_meta.get('total_transitions', 0)} transitions")
        for k, v in visual_meta.get("status_counts", {}).items():
            print(f"  {k}: {v}")
        print("  DINO is QA-only: no image is removed from the legal continuous sequence.")

    if not image_benchmark.empty:
        counts = image_benchmark["source_match_confidence"].value_counts()
        comparable = image_benchmark[
            image_benchmark["source_match_confidence"].isin(["HIGH", "MEDIUM"])
            & image_benchmark["generated_primary_span"].notna()
        ]
        matches = int(comparable["benchmark_status"].eq("SPAN_MATCH").sum()) if len(comparable) else 0
        pct = 100.0 * matches / len(comparable) if len(comparable) else float("nan")
        print("\nExisting 163 image benchmark:")
        print(
            f"  source matches: HIGH={int(counts.get('HIGH', 0))}, "
            f"MEDIUM={int(counts.get('MEDIUM', 0))}, LOW={int(counts.get('LOW', 0))}"
        )
        print(f"  selected primary span: comparable={len(comparable)}, exact={matches}, exact_match={pct:.2f}%")
        if not opk_benchmark.empty:
            for _, r in opk_benchmark.iterrows():
                print(
                    f"  {r['target_method']}: comparable={int(r['comparable_images'])}, "
                    f"exact={int(r['exact_span_matches'])}, exact_match={r['exact_span_match_percent']:.2f}%"
                )

    if args.render_mode == "preview":
        print("\nCandidate 163 JPGs are review previews, not full-resolution deliverables.")
        print("Use --render-mode full after visual QA if full-resolution copies are required.")
    if not args.skip_structure_crops:
        print("Structure crop pixel projection is approximate (EO + footprint FOV), not calibrated camera IO.")
    print("Span visibility is QA metadata only. Partial-span/no-structure frames are kept when they belong to the legal continuous sequence.")
    print("Capture date inference uses raw EXIF or --capture-date; existing 163 is benchmark-only.")

    print("\nNo source imagery was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_v07())
