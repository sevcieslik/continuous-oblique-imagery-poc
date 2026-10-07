#!/usr/bin/env python
r"""
Audit a manually produced EPIC 163 Continuous Oblique Imagery product for internal continuity.

v0.3 fixes raw-match collision handling and refines high-similarity classification so normal continuous-image overlap is not treated
as a duplicate error, validates apparent numerical span gaps against the project reference
topology, and grades calibration trust as A / B / REVIEW.

This script deliberately does NOT treat the manual product as perfect ground truth.
It uses DINOv3 as QA-only visual evidence, combines that with trusted raw-image/EO
mapping where available, and flags suspicious manual images/transitions for review.
Nothing is deleted or relabelled automatically.

Designed for the DH1 project layout used by continuous_oblique_v014.py.

Outputs
-------
<output>/audit_summary.json
<output>/audit_summary.csv
<output>/manual_image_audit.csv
<output>/geometry_run_transitions.csv
<output>/span_continuity.csv
<output>/high_similarity_pairs.csv
<output>/duplicate_candidates.csv
<output>/redundant_frame_candidates.csv
<output>/expected_high_overlap_pairs.csv
<output>/raw_mapping_collisions.csv
<output>/label_suspects.csv
<output>/trusted_manual_grade_A.csv
<output>/calibration_pool_A_B.csv
<output>/trusted_manual_subset.csv  (backward-compatible alias for Grade A)
<output>/dino_feature_index.csv
<output>/review_manifest.csv
<output>/report.html
<output>/review/<CATEGORY>/*.jpg
<output>/qa/visual_continuity/feature_cache/*.npz

Example
-------
python audit_manual_163_v03.py "D:\\Sev\\Obliques\\DH1"

Recommended interpreter
-----------------------
Use the same environment that successfully runs DINOv3 for continuous_oblique_v014.py.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FINAL_163_RE = re.compile(
    r"^(?P<circuit>[^_]+)_"
    r"(?P<span_from>\d+)-(?P<span_to>\d+)_"
    r"(?P<output_id>\d+)_"
    r"(?P<date>\d{6})_"
    r"(?P<version>v\d+)\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)
IMAGE_EXTS = {".jpg", ".jpeg"}


def _safe_str(v) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return s


def _boolish(v) -> bool | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    exact = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in exact:
            return exact[c.lower()]
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): str(c) for c in df.columns}
    for c in candidates:
        k = re.sub(r"[^a-z0-9]+", "", c.lower())
        if k in norm:
            return norm[k]
    if required:
        raise KeyError(f"Required column not found. Tried {candidates}. Available={list(df.columns)}")
    return None


def _read_csv(path: Path | None, required: bool = False) -> pd.DataFrame:
    if path is None or not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def parse_final_name(name: str) -> dict:
    m = FINAL_163_RE.match(Path(name).name)
    if not m:
        return {
            "filename": Path(name).name,
            "parse_status": "UNPARSED",
            "circuit": None,
            "span_id": None,
            "span_from": np.nan,
            "span_to": np.nan,
            "output_id": np.nan,
            "capture_date_code": None,
            "version": None,
        }
    d = m.groupdict()
    sf, st = int(d["span_from"]), int(d["span_to"])
    return {
        "filename": Path(name).name,
        "parse_status": "OK",
        "circuit": d["circuit"],
        "span_id": f"{sf}-{st}",
        "span_from": sf,
        "span_to": st,
        "output_id": int(d["output_id"]),
        "capture_date_code": d["date"],
        "version": d["version"],
    }


def index_manual_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            r = parse_final_name(p.name)
            r["manual_path"] = str(p)
            rows.append(r)
    return pd.DataFrame(rows)


def discover_paths(root: Path, ai_run: str | None, continuous_script: str | None) -> dict:
    manual_dir = _first_existing([
        root / "163_continuous_oblique_imagery" / "v1",
        root / "163_continuous_oblique_imagery",
    ])
    if manual_dir is None:
        raise FileNotFoundError("Could not find manual 163_continuous_oblique_imagery")

    if ai_run:
        ai_root = root / ai_run
    else:
        candidates = [p for p in root.glob("_continuous_oblique_*v014*") if p.is_dir()]
        if not candidates:
            candidates = [p for p in root.glob("_continuous_oblique_*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError("Could not find _continuous_oblique_* AI run folder")
        ai_root = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    benchmark_csv = _first_existing([
        ai_root / "existing_163_benchmark.csv",
        ai_root / "deliverable" / "qa_metadata" / "existing_163_benchmark.csv",
    ])
    image_index_csv = _first_existing([
        ai_root / "image_index.csv",
        ai_root / "deliverable" / "qa_metadata" / "image_index.csv",
    ])
    if benchmark_csv is None:
        raise FileNotFoundError("existing_163_benchmark.csv not found")
    if image_index_csv is None:
        raise FileNotFoundError("image_index.csv not found")

    reference_spans_csv = _first_existing([
        ai_root / "reference_spans.csv",
        ai_root / "deliverable" / "qa_metadata" / "reference_spans.csv",
    ])

    raw_dir = _first_existing([
        root / "130_raw_overlapping_oblique_imagery" / "v1",
        root / "130_raw_overlapping_oblique_imagery",
    ])

    if continuous_script:
        cont = Path(continuous_script)
        if not cont.is_absolute():
            cont = root / cont
    else:
        cont = root / "continuous_oblique_v014.py"
    if not cont.exists():
        raise FileNotFoundError(
            f"continuous_oblique_v014.py not found at {cont}. "
            "Pass --continuous-script if it is elsewhere."
        )

    return {
        "manual_dir": manual_dir,
        "ai_root": ai_root,
        "benchmark_csv": benchmark_csv,
        "image_index_csv": image_index_csv,
        "reference_spans_csv": reference_spans_csv,
        "raw_dir": raw_dir,
        "continuous_script": cont,
    }


def load_continuous_module(path: Path):
    spec = importlib.util.spec_from_file_location("continuous_oblique_v014_for_manual_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = ["extract_dinov3_features", "_dino_pair_metrics", "_robust_visual_z"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"{path.name} is missing required DINO helpers: {missing}")
    return mod


def normalize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    mf = _find_col(df, ["existing_163_filename", "manual_filename", "filename"], required=True)
    ms = _find_col(df, ["existing_span", "manual_span", "span_id"], required=True)
    raw = _find_col(df, ["matched_raw_filename", "raw_filename", "source_filename"], required=True)
    conf = _find_col(df, ["source_match_confidence", "match_confidence", "confidence"], required=True)
    out = pd.DataFrame({
        "filename": df[mf].map(lambda x: Path(str(x)).name if _safe_str(x) else None),
        "benchmark_span": df[ms].map(_safe_str),
        "matched_raw_filename": df[raw].map(lambda x: Path(str(x)).name if _safe_str(x) else None),
        "source_match_confidence": df[conf].astype(str).str.upper().str.strip(),
    })
    for out_name, candidates in {
        "appearance_score": ["appearance_score", "match_score"],
        "appearance_margin": ["appearance_margin", "match_margin"],
        "benchmark_status": ["benchmark_status"],
        "benchmark_generated_primary_span": ["generated_primary_span", "primary_span"],
    }.items():
        c = _find_col(df, candidates)
        out[out_name] = df[c] if c else None
    return out.drop_duplicates("filename", keep="first")


def normalize_image_index(df: pd.DataFrame) -> pd.DataFrame:
    fn = _find_col(df, ["filename", "raw_filename"], required=True)
    out = pd.DataFrame({"matched_raw_filename": df[fn].map(lambda x: Path(str(x)).name if _safe_str(x) else None)})
    fields = {
        "idx_pass_id": ["pass_id"],
        "idx_pass_direction": ["pass_direction"],
        "idx_circuit_run_id": ["circuit_run_id", "run_id"],
        "idx_view_direction": ["view_direction"],
        "idx_product_status": ["product_status"],
        "idx_rejection_reason": ["rejection_reason"],
        "idx_valid_for_product": ["valid_for_product"],
        "idx_sequence_keep": ["sequence_keep", "valid_for_163"],
        "idx_primary_span": ["primary_span"],
        "idx_reference_chainage": ["reference_chainage", "centre_chainage", "camera_chainage"],
        "idx_camera_chainage": ["camera_chainage"],
        "idx_eo_order": ["eo_order", "acquisition_index"],
        "idx_eo_source": ["eo_source"],
        "idx_coverage_status": ["coverage_status"],
        "idx_reference_confidence": ["reference_confidence"],
        "idx_optical_target_confidence": ["optical_target_confidence"],
    }
    for out_name, candidates in fields.items():
        c = _find_col(df, candidates)
        out[out_name] = df[c] if c else None
    return out.drop_duplicates("matched_raw_filename", keep="first")


def load_reference_span_ids(df: pd.DataFrame) -> set[str]:
    """Return canonical span IDs present in the project reference topology."""
    if df is None or df.empty:
        return set()
    sid = _find_col(df, ["span_id", "reference_id", "span"])
    if sid:
        vals = {_safe_str(v) for v in df[sid]}
        return {v for v in vals if v}
    sf = _find_col(df, ["span_from", "from_idx", "from_structure_idx"])
    st = _find_col(df, ["span_to", "to_idx", "to_structure_idx"])
    if sf and st:
        out = set()
        for a, b in zip(df[sf], df[st]):
            try:
                out.add(f"{int(float(a))}-{int(float(b))}")
            except Exception:
                pass
        return out
    return set()


def build_manual_audit_index(
    manual: pd.DataFrame,
    benchmark: pd.DataFrame,
    image_idx: pd.DataFrame,
    raw_dir: Path | None,
    feature_source_mode: str,
) -> pd.DataFrame:
    out = manual.merge(benchmark, on="filename", how="left")
    out = out.merge(image_idx, on="matched_raw_filename", how="left")
    out["source_match_confidence"] = out["source_match_confidence"].fillna("UNRESOLVED").astype(str).str.upper()
    out["source_match_trusted"] = out["source_match_confidence"].isin(["HIGH", "MEDIUM"])

    feature_paths = []
    feature_sources = []
    raw_paths = []
    for _, r in out.iterrows():
        rp = None
        raw_name = _safe_str(r.get("matched_raw_filename"))
        if raw_dir is not None and raw_name:
            candidate = raw_dir / raw_name
            if candidate.exists():
                rp = candidate
        raw_paths.append(str(rp) if rp else None)
        mp = Path(str(r["manual_path"]))
        use_raw = feature_source_mode == "raw-when-trusted" and bool(r.get("source_match_trusted")) and rp is not None
        feature_paths.append(str(rp if use_raw else mp))
        feature_sources.append("RAW_TRUSTED_MATCH" if use_raw else "MANUAL_FINAL")
    out["matched_raw_path"] = raw_paths
    out["dino_source_path"] = feature_paths
    out["dino_source_type"] = feature_sources

    out["span_from"] = pd.to_numeric(out["span_from"], errors="coerce")
    out["span_to"] = pd.to_numeric(out["span_to"], errors="coerce")
    out["idx_reference_chainage"] = pd.to_numeric(out.get("idx_reference_chainage"), errors="coerce")
    out["idx_eo_order"] = pd.to_numeric(out.get("idx_eo_order"), errors="coerce")
    out["geometry_auditable"] = (
        out["source_match_trusted"]
        & out["idx_circuit_run_id"].notna()
        & out["idx_reference_chainage"].notna()
    )
    return out


def build_geometry_transitions(images: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gdf = images[images["geometry_auditable"].eq(True)].copy()
    if gdf.empty:
        return pd.DataFrame()

    for run_id, grp in gdf.groupby("idx_circuit_run_id", sort=False):
        dirs = grp["idx_pass_direction"].dropna().astype(str)
        direction = dirs.mode().iloc[0] if not dirs.empty else "UNKNOWN_PASS"
        asc = direction != "BACKWARD_PASS"
        grp = grp.sort_values(
            ["idx_reference_chainage", "idx_eo_order", "output_id"],
            ascending=[asc, True, True],
            na_position="last",
        ).reset_index(drop=True)
        vals = grp.to_dict("records")
        for i, (a, b) in enumerate(zip(vals[:-1], vals[1:]), start=1):
            cha = float(a["idx_reference_chainage"])
            chb = float(b["idx_reference_chainage"])
            chainage_delta = chb - cha
            eo_a = pd.to_numeric(pd.Series([a.get("idx_eo_order")]), errors="coerce").iloc[0]
            eo_b = pd.to_numeric(pd.Series([b.get("idx_eo_order")]), errors="coerce").iloc[0]
            eo_delta = float(eo_b - eo_a) if pd.notna(eo_a) and pd.notna(eo_b) else np.nan
            sf_a = pd.to_numeric(pd.Series([a.get("span_from")]), errors="coerce").iloc[0]
            sf_b = pd.to_numeric(pd.Series([b.get("span_from")]), errors="coerce").iloc[0]
            span_step = float(sf_b - sf_a) if pd.notna(sf_a) and pd.notna(sf_b) else np.nan
            expected_span_step = -1 if direction == "BACKWARD_PASS" else 1
            pass_problem = any(
                str(x) not in {"FORWARD_PASS", "BACKWARD_PASS"}
                for x in [a.get("idx_pass_direction"), b.get("idx_pass_direction")]
            )
            rows.append({
                "transition_type": "GEOMETRY_RUN",
                "circuit_run_id": run_id,
                "pass_direction": direction,
                "transition_index": i,
                "image_a": a["filename"],
                "image_b": b["filename"],
                "span_a": a.get("span_id"),
                "span_b": b.get("span_id"),
                "raw_a": a.get("matched_raw_filename"),
                "raw_b": b.get("matched_raw_filename"),
                "chainage_a": cha,
                "chainage_b": chb,
                "chainage_delta": chainage_delta,
                "chainage_gap_abs": abs(chainage_delta),
                "eo_order_a": eo_a,
                "eo_order_b": eo_b,
                "eo_order_delta": eo_delta,
                "span_step": span_step,
                "expected_span_step": expected_span_step,
                "same_raw": _safe_str(a.get("matched_raw_filename")) == _safe_str(b.get("matched_raw_filename")),
                "pass_problem": pass_problem,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["chainage_gap_robust_z"] = np.nan
    out["eo_gap_robust_z"] = np.nan
    for rid, grp in out.groupby("circuit_run_id", sort=False):
        for col, zcol in [("chainage_gap_abs", "chainage_gap_robust_z"), ("eo_order_delta", "eo_gap_robust_z")]:
            x = pd.to_numeric(grp[col], errors="coerce").abs()
            med = float(x.median()) if x.notna().any() else np.nan
            mad = float((x - med).abs().median()) if x.notna().any() else np.nan
            if math.isfinite(mad) and mad > 1e-9:
                out.loc[grp.index, zcol] = (x - med) / (1.4826 * mad)
            else:
                out.loc[grp.index, zcol] = 0.0
    return out


def pair_metrics(mod, fa: dict, fb: dict, match_threshold: float) -> dict:
    m = mod._dino_pair_metrics(fa, fb, match_threshold)
    g = float(m.get("global_similarity", np.nan))
    d = float(m.get("dense_best_mean", np.nan))
    coh = float(m.get("spatial_coherence", np.nan))
    if not math.isfinite(coh):
        coh = 0.0
    score = 0.45 * g + 0.35 * d + 0.20 * coh if math.isfinite(g) and math.isfinite(d) else np.nan
    m["visual_score"] = score
    return m


def attach_dino_to_transitions(
    mod,
    transitions: pd.DataFrame,
    features: dict,
    match_threshold: float,
    absolute_weak: float,
    robust_weak_z: float,
    robust_review_z: float,
    duplicate_global: float,
    duplicate_dense: float,
) -> pd.DataFrame:
    if transitions.empty:
        return transitions
    out = transitions.copy()
    metric_cols = [
        "global_similarity", "dense_best_mean", "mutual_match_count", "mutual_match_ratio",
        "mutual_similarity_mean", "mutual_similarity_median", "mutual_similarity_p10",
        "spatial_coherence", "median_patch_dx", "median_patch_dy", "visual_score",
    ]
    for c in metric_cols:
        out[c] = np.nan
    out["visual_analysis_status"] = "NOT_ANALYZED"
    for i, r in out.iterrows():
        a, b = str(r["image_a"]), str(r["image_b"])
        if a not in features or b not in features:
            out.at[i, "visual_analysis_status"] = "FEATURE_MISSING"
            continue
        m = pair_metrics(mod, features[a], features[b], match_threshold)
        for k, v in m.items():
            if k in out.columns:
                out.at[i, k] = v
        out.at[i, "visual_analysis_status"] = "ANALYZED" if pd.notna(out.at[i, "visual_score"]) else "METRIC_FAILED"

    out["visual_robust_z"] = np.nan
    analyzed = out["visual_analysis_status"].eq("ANALYZED")
    group_col = "circuit_run_id" if "circuit_run_id" in out.columns else None
    if group_col:
        for _, grp in out[analyzed].groupby(group_col, dropna=False, sort=False):
            out.loc[grp.index, "visual_robust_z"] = mod._robust_visual_z(grp["visual_score"])
    else:
        out.loc[analyzed, "visual_robust_z"] = mod._robust_visual_z(out.loc[analyzed, "visual_score"])

    statuses, reasons, overlap_classes = [], [], []
    for _, r in out.iterrows():
        if r["visual_analysis_status"] != "ANALYZED":
            statuses.append(r["visual_analysis_status"]); reasons.append("DINO metrics unavailable"); overlap_classes.append("NOT_ANALYZED"); continue
        score = float(r["visual_score"])
        z = float(r["visual_robust_z"]) if pd.notna(r["visual_robust_z"]) else 0.0
        glob = float(r["global_similarity"])
        dense = float(r["dense_best_mean"])
        very_high_similarity = glob >= duplicate_global and dense >= duplicate_dense
        visual_weak = score < absolute_weak or z <= robust_weak_z
        visual_review = visual_weak or z <= robust_review_z
        pass_problem = bool(r.get("pass_problem", False))
        chz = float(r.get("chainage_gap_robust_z")) if pd.notna(r.get("chainage_gap_robust_z")) else 0.0
        eoz = float(r.get("eo_gap_robust_z")) if pd.notna(r.get("eo_gap_robust_z")) else 0.0
        same_raw = bool(r.get("same_raw", False))
        span_step = r.get("span_step")
        expected = r.get("expected_span_step")
        label_jump = pd.notna(span_step) and pd.notna(expected) and abs(float(span_step) - float(expected)) > 1.01
        geom_jump = chz >= 4.0 or eoz >= 4.0
        overlap_classes.append("EXPECTED_HIGH_OVERLAP" if very_high_similarity and not same_raw else ("SAME_RAW_REUSE" if same_raw else "NORMAL"))

        # v0.2: high DINO similarity between neighbouring frames is expected in a
        # continuous-oblique sequence. Only actual same-raw reuse is a duplicate flag here.
        if same_raw:
            st, reason = "TRUE_DUPLICATE_SUSPECT", "The same trusted raw frame appears twice in the manual sequence"
        elif pass_problem:
            st, reason = "POSSIBLE_WRONG_PASS", "One or both source frames are not classified as a normal forward/backward pass"
        elif geom_jump and visual_weak:
            st, reason = "POSSIBLE_SEQUENCE_JUMP", "Geometry/acquisition jump and weak DINO continuity"
        elif geom_jump and not visual_review:
            st, reason = "GEOMETRY_JUMP_VISUAL_OK", "Geometry/acquisition jump but DINO continuity remains normal"
        elif label_jump and visual_weak:
            st, reason = "POSSIBLE_BAD_ORDER_OR_LABEL", "Manual span progression jumps and DINO continuity is weak"
        elif visual_weak:
            st, reason = "WEAK_VISUAL_OVERLAP", "DINO continuity is a strong outlier"
        elif visual_review:
            st, reason = "VISUAL_REVIEW", "DINO continuity is below the run distribution"
        else:
            st, reason = "CONTINUOUS", "Manual transition is visually and geometrically consistent"
        statuses.append(st); reasons.append(reason)
    out["audit_status"] = statuses
    out["review_reason"] = reasons
    out["overlap_class"] = overlap_classes
    return out


def build_span_continuity(
    images: pd.DataFrame,
    features: dict,
    mod,
    match_threshold: float,
    absolute_weak: float,
    reference_span_ids: set[str] | None = None,
) -> pd.DataFrame:
    good = images[images["parse_status"].eq("OK")].copy()
    span_groups = {int(k): g.copy() for k, g in good.dropna(subset=["span_from"]).groupby(good["span_from"].astype(int))}
    if not span_groups:
        return pd.DataFrame()
    topology = set(reference_span_ids or [])
    keys = sorted(span_groups)
    rows = []
    for sf in keys:
        nxt = sf + 1
        expected_next_id = f"{nxt}-{nxt+1}"
        if nxt not in span_groups:
            if sf != max(keys):
                topology_has_next = (expected_next_id in topology) if topology else None
                if topology and not topology_has_next:
                    status = "TOPOLOGY_NO_SUCH_SPAN"
                    reason = "The next numerical span is not present in the reference topology; this is not treated as a manual omission"
                else:
                    status = "MISSING_NEXT_MANUAL_SPAN"
                    reason = "Reference topology expects the next numerical span but no manual 163 image exists for it" if topology else "No manual images exist for the next numerical span; topology was unavailable"
                rows.append({
                    "span_a": f"{sf}-{sf+1}", "span_b": expected_next_id,
                    "image_a": None, "image_b": None, "best_pair_found": False,
                    "topology_has_next_span": topology_has_next,
                    "audit_status": status, "review_reason": reason,
                })
            continue
        ga, gb = span_groups[sf], span_groups[nxt]
        candidates = []
        for _, a in ga.iterrows():
            for _, b in gb.iterrows():
                fa, fb = features.get(str(a["filename"])), features.get(str(b["filename"]))
                if fa is None or fb is None:
                    continue
                gsim = float(np.dot(
                    mod._normalise_rows(np.asarray(fa["global"])[None, :])[0],
                    mod._normalise_rows(np.asarray(fb["global"])[None, :])[0],
                ))
                candidates.append((gsim, a, b))
        if not candidates:
            rows.append({
                "span_a": f"{sf}-{sf+1}", "span_b": expected_next_id,
                "image_a": None, "image_b": None, "best_pair_found": False,
                "topology_has_next_span": expected_next_id in topology if topology else None,
                "audit_status": "FEATURE_MISSING", "review_reason": "No DINO feature pair available",
            })
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, a, b = candidates[0]
        m = pair_metrics(mod, features[str(a["filename"])], features[str(b["filename"])], match_threshold)
        score = float(m["visual_score"]) if pd.notna(m["visual_score"]) else np.nan
        st = "CONTINUOUS" if math.isfinite(score) and score >= absolute_weak else "WEAK_SPAN_CONTINUITY"
        reason = "Best DINO link between adjacent manual spans is acceptable" if st == "CONTINUOUS" else "Even the best DINO link between adjacent manual spans is weak"
        rec = {
            "span_a": f"{sf}-{sf+1}", "span_b": expected_next_id,
            "image_a": a["filename"], "image_b": b["filename"], "best_pair_found": True,
            "topology_has_next_span": expected_next_id in topology if topology else None,
            "audit_status": st, "review_reason": reason,
            "candidate_pair_count": len(candidates),
        }
        rec.update(m)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty and "visual_score" in out.columns:
        mask = pd.to_numeric(out["visual_score"], errors="coerce").notna()
        out.loc[mask, "visual_robust_z"] = mod._robust_visual_z(out.loc[mask, "visual_score"])
    return out


def build_duplicate_candidates(
    images: pd.DataFrame,
    features: dict,
    mod,
    duplicate_global: float,
    duplicate_dense: float,
    match_threshold: float,
    local_eo_gap: int = 3,
    local_span_gap: int = 1,
    far_span_gap: int = 4,
    far_chainage_factor: float = 6.0,
    expected_overlap_quantile: float = 0.80,
) -> pd.DataFrame:
    """Classify high-similarity/reused-source pairs conservatively.

    v0.3 fixes an important v0.2 issue: ``matched_raw_filename`` is only evidence
    of literal raw reuse when BOTH manual->raw matches are HIGH/MEDIUM confidence.
    LOW/unresolved fingerprint collisions are reported separately and never become
    duplicate evidence on their own.

    Local highly-similar frames in the same normal flight run are expected for a
    continuous-oblique product and are neutral QA evidence.
    """
    names = [str(x) for x in images["filename"].tolist() if str(x) in features]
    if len(names) < 2:
        return pd.DataFrame()

    globals_mat = np.vstack([
        mod._normalise_rows(np.asarray(features[n]["global"])[None, :])[0] for n in names
    ])
    sim = globals_mat @ globals_mat.T
    meta = images.set_index(images["filename"].astype(str), drop=False).to_dict("index")

    run_steps: dict[str, float] = {}
    aud = images[images.get("geometry_auditable", pd.Series(False, index=images.index)).eq(True)].copy()
    if not aud.empty:
        for rid, grp in aud.groupby("idx_circuit_run_id", dropna=True):
            vals = sorted(pd.to_numeric(grp["idx_reference_chainage"], errors="coerce").dropna().astype(float).unique())
            if len(vals) >= 2:
                diffs = np.diff(vals)
                diffs = diffs[diffs > 1e-6]
                if len(diffs):
                    run_steps[str(rid)] = float(np.median(diffs))

    def context(ma: dict, mb: dict):
        run_a, run_b = _safe_str(ma.get("idx_circuit_run_id")), _safe_str(mb.get("idx_circuit_run_id"))
        same_run = run_a is not None and run_a == run_b
        dir_a, dir_b = _safe_str(ma.get("idx_pass_direction")), _safe_str(mb.get("idx_pass_direction"))
        normal_passes = dir_a in {"FORWARD_PASS","BACKWARD_PASS"} and dir_b in {"FORWARD_PASS","BACKWARD_PASS"}

        eo_a = pd.to_numeric(pd.Series([ma.get("idx_eo_order")]), errors="coerce").iloc[0]
        eo_b = pd.to_numeric(pd.Series([mb.get("idx_eo_order")]), errors="coerce").iloc[0]
        eo_gap = abs(float(eo_b-eo_a)) if pd.notna(eo_a) and pd.notna(eo_b) else np.nan

        sf_a = pd.to_numeric(pd.Series([ma.get("span_from")]), errors="coerce").iloc[0]
        sf_b = pd.to_numeric(pd.Series([mb.get("span_from")]), errors="coerce").iloc[0]
        span_gap = abs(float(sf_b-sf_a)) if pd.notna(sf_a) and pd.notna(sf_b) else np.nan

        ch_a = pd.to_numeric(pd.Series([ma.get("idx_reference_chainage")]), errors="coerce").iloc[0]
        ch_b = pd.to_numeric(pd.Series([mb.get("idx_reference_chainage")]), errors="coerce").iloc[0]
        chainage_gap = abs(float(ch_b-ch_a)) if pd.notna(ch_a) and pd.notna(ch_b) else np.nan
        typical_step = run_steps.get(run_a) if same_run else None

        local_by_chainage = bool(
            typical_step and math.isfinite(chainage_gap)
            and chainage_gap <= max(3.0*typical_step, typical_step+1.0)
        )
        local_pair = same_run and normal_passes and (
            (math.isfinite(eo_gap) and eo_gap <= local_eo_gap)
            or (math.isfinite(span_gap) and span_gap <= local_span_gap)
            or local_by_chainage
        )
        far_pair = (
            (run_a is not None and run_b is not None and run_a != run_b)
            or (math.isfinite(span_gap) and span_gap >= far_span_gap)
            or (
                typical_step is not None and math.isfinite(chainage_gap)
                and chainage_gap >= far_chainage_factor*typical_step
            )
        )
        return {
            "run_a": run_a, "run_b": run_b, "same_run": same_run,
            "normal_passes": normal_passes, "eo_gap": eo_gap, "span_gap": span_gap,
            "chainage_gap": chainage_gap, "typical_step": typical_step,
            "local_pair": local_pair, "far_pair": far_pair,
        }

    # Learn what "very similar but local" means from this product rather than forcing
    # the duplicate threshold onto normal overlapping frames.
    local_globals = []
    for i in range(len(names)):
        ma = meta.get(names[i], {})
        for j in range(i + 1, len(names)):
            mb = meta.get(names[j], {})
            c = context(ma, mb)
            if c["local_pair"]:
                local_globals.append(float(sim[i, j]))
    if local_globals:
        q = min(0.99, max(0.50, float(expected_overlap_quantile)))
        local_global_cut = float(np.quantile(np.asarray(local_globals, dtype=float), q))
    else:
        local_global_cut = float(duplicate_global)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g = float(sim[i, j])
            a, b = names[i], names[j]
            ma, mb = meta.get(a, {}), meta.get(b, {})
            c = context(ma, mb)

            raw_a = _safe_str(ma.get("matched_raw_filename"))
            raw_b = _safe_str(mb.get("matched_raw_filename"))
            conf_a = str(ma.get("source_match_confidence") or "UNRESOLVED").upper()
            conf_b = str(mb.get("source_match_confidence") or "UNRESOLVED").upper()
            trusted_a = conf_a in {"HIGH", "MEDIUM"}
            trusted_b = conf_b in {"HIGH", "MEDIUM"}
            same_raw_name = raw_a is not None and raw_a == raw_b
            same_raw_trusted = bool(same_raw_name and trusted_a and trusted_b)

            # Candidate gates:
            # - any apparent same-raw collision is retained for diagnostics;
            # - strict DINO threshold can support a duplicate/redundancy call;
            # - top local similarity is retained as EXPECTED_HIGH_OVERLAP.
            local_high = bool(c["local_pair"] and g >= local_global_cut)
            strict_global = g >= float(duplicate_global)
            if not same_raw_name and not strict_global and not local_high:
                continue

            m = pair_metrics(mod, features[a], features[b], match_threshold)
            dense = float(m.get("dense_best_mean", np.nan))
            dino_extreme = bool(
                strict_global and math.isfinite(dense) and dense >= float(duplicate_dense)
            )

            if same_raw_trusted:
                if c["far_pair"]:
                    status = "TRUE_DUPLICATE_SUSPECT"
                    reason = "The same HIGH/MEDIUM-confidence raw source is reused far apart in manual span/run space"
                    strong = True
                else:
                    status = "POSSIBLE_REDUNDANT_FRAME"
                    reason = "The same HIGH/MEDIUM-confidence raw source is reused locally; this may be deliberate but adds no new source frame"
                    strong = False
            elif same_raw_name and not same_raw_trusted:
                if dino_extreme and c["far_pair"]:
                    status = "TRUE_DUPLICATE_SUSPECT"
                    reason = "Low-confidence raw mapping collides and independent DINO evidence is also extreme far from the local sequence"
                    strong = True
                elif local_high:
                    status = "EXPECTED_HIGH_OVERLAP"
                    reason = "Raw-name collision is not trusted; DINO similarity is locally consistent with expected continuous-image overlap"
                    strong = False
                else:
                    status = "RAW_MAPPING_COLLISION_UNTRUSTED"
                    reason = "Two manual images fingerprint to the same raw filename, but at least one mapping is LOW/unresolved; not duplicate evidence"
                    strong = False
            elif dino_extreme and c["far_pair"]:
                status = "TRUE_DUPLICATE_SUSPECT"
                reason = "Different raw mappings but extreme DINO similarity occurs far apart in run/span/chainage space"
                strong = True
            elif local_high:
                status = "EXPECTED_HIGH_OVERLAP"
                reason = "High DINO similarity occurs locally in the same run and is consistent with expected continuous-image overlap"
                strong = False
            elif dino_extreme:
                status = "POSSIBLE_REDUNDANT_FRAME"
                reason = "Extreme DINO similarity is not clearly local or clearly far; retain as a mild review candidate"
                strong = False
            else:
                continue

            rows.append({
                "image_a": a, "image_b": b,
                "span_a": _safe_str(ma.get("span_id")), "span_b": _safe_str(mb.get("span_id")),
                "raw_a": raw_a, "raw_b": raw_b,
                "source_confidence_a": conf_a, "source_confidence_b": conf_b,
                "same_raw_source": same_raw_name,
                "same_raw_trusted": same_raw_trusted,
                "same_run": c["same_run"], "normal_passes": c["normal_passes"],
                "eo_order_gap": c["eo_gap"], "span_gap": c["span_gap"],
                "chainage_gap": c["chainage_gap"],
                "typical_run_chainage_step": c["typical_step"],
                "local_pair": c["local_pair"], "far_pair": c["far_pair"],
                "local_global_cut": local_global_cut,
                "dino_extreme": dino_extreme,
                "audit_status": status, "strong_duplicate_suspect": strong,
                "review_reason": reason,
                **m,
            })

    if not rows:
        return pd.DataFrame()
    rank = {
        "TRUE_DUPLICATE_SUSPECT": 0,
        "POSSIBLE_REDUNDANT_FRAME": 1,
        "EXPECTED_HIGH_OVERLAP": 2,
        "RAW_MAPPING_COLLISION_UNTRUSTED": 3,
    }
    out = pd.DataFrame(rows)
    out["_rank"] = out["audit_status"].map(rank).fillna(9)
    out = out.sort_values(
        ["_rank", "same_raw_trusted", "global_similarity"],
        ascending=[True, False, False]
    ).drop(columns="_rank")
    return out.reset_index(drop=True)


def build_label_suspects(
    images: pd.DataFrame,
    features: dict,
    mod,
    match_threshold: float,
    label_window: int,
    label_suspect_margin: float,
    absolute_weak: float,
) -> pd.DataFrame:
    parsed = images[images["parse_status"].eq("OK") & images["span_from"].notna()].copy()
    if parsed.empty:
        return pd.DataFrame()
    groups = {int(k): g.copy() for k, g in parsed.groupby(parsed["span_from"].astype(int))}
    rows = []
    for _, r in parsed.iterrows():
        fn = str(r["filename"])
        if fn not in features:
            continue
        sf = int(r["span_from"])
        adj_candidates = []
        nonadj_candidates = []
        fa = features[fn]
        ga = mod._normalise_rows(np.asarray(fa["global"])[None, :])[0]
        for d in range(-label_window, label_window + 1):
            if d == 0 or sf + d not in groups:
                continue
            for _, q in groups[sf + d].iterrows():
                qn = str(q["filename"])
                if qn not in features or qn == fn:
                    continue
                gb = mod._normalise_rows(np.asarray(features[qn]["global"])[None, :])[0]
                gsim = float(np.dot(ga, gb))
                item = (gsim, d, qn, q.get("span_id"))
                if abs(d) == 1:
                    adj_candidates.append(item)
                elif abs(d) >= 2:
                    nonadj_candidates.append(item)
        if not adj_candidates or not nonadj_candidates:
            continue
        adj_candidates.sort(reverse=True, key=lambda x: x[0])
        nonadj_candidates.sort(reverse=True, key=lambda x: x[0])
        _, ad, afn, asp = adj_candidates[0]
        _, nd, nfn, nsp = nonadj_candidates[0]
        am = pair_metrics(mod, fa, features[afn], match_threshold)
        nm = pair_metrics(mod, fa, features[nfn], match_threshold)
        av = float(am["visual_score"]) if pd.notna(am["visual_score"]) else np.nan
        nv = float(nm["visual_score"]) if pd.notna(nm["visual_score"]) else np.nan
        if not (math.isfinite(av) and math.isfinite(nv)):
            continue
        margin = nv - av
        # Conservative: only flag when a non-adjacent span is substantially more similar,
        # and the normal adjacent relation is itself weak/review-worthy.
        if margin >= label_suspect_margin and (av < absolute_weak or margin >= label_suspect_margin * 1.5):
            rows.append({
                "filename": fn,
                "manual_span": r.get("span_id"),
                "best_adjacent_image": afn,
                "best_adjacent_span": asp,
                "best_adjacent_step": ad,
                "adjacent_visual_score": av,
                "best_nonadjacent_image": nfn,
                "best_nonadjacent_span": nsp,
                "best_nonadjacent_step": nd,
                "nonadjacent_visual_score": nv,
                "score_margin_nonadj_minus_adj": margin,
                "audit_status": "MANUAL_LABEL_SUSPECT",
                "review_reason": "A non-adjacent span has materially stronger DINO continuity than the expected adjacent span",
            })
    return pd.DataFrame(rows).sort_values("score_margin_nonadj_minus_adj", ascending=False) if rows else pd.DataFrame()


def aggregate_image_status(
    images: pd.DataFrame,
    transitions: pd.DataFrame,
    high_similarity_pairs: pd.DataFrame,
    label_suspects: pd.DataFrame,
) -> pd.DataFrame:
    out = images.copy()
    issue_map: dict[str, set[str]] = defaultdict(set)
    reason_map: dict[str, list[str]] = defaultdict(list)

    for _, r in transitions.iterrows():
        st = _safe_str(r.get("audit_status"))
        if st and st != "CONTINUOUS":
            for col in ["image_a", "image_b"]:
                fn = _safe_str(r.get(col))
                if fn:
                    issue_map[fn].add(st)
                    reason_map[fn].append(_safe_str(r.get("review_reason")) or st)
    for _, r in high_similarity_pairs.iterrows():
        st = _safe_str(r.get("audit_status"))
        # Expected overlap is positive/neutral. Untrusted raw-name collisions are a
        # benchmark mapping diagnostic, not product-quality evidence.
        if st in {"EXPECTED_HIGH_OVERLAP", "RAW_MAPPING_COLLISION_UNTRUSTED"}:
            continue
        for col in ["image_a", "image_b"]:
            fn = _safe_str(r.get(col))
            if fn:
                issue_map[fn].add(st or "POSSIBLE_REDUNDANT_FRAME")
                reason_map[fn].append(_safe_str(r.get("review_reason")) or (st or "High-similarity pair"))
    for _, r in label_suspects.iterrows():
        fn = _safe_str(r.get("filename"))
        if fn:
            issue_map[fn].add("MANUAL_LABEL_SUSPECT")
            reason_map[fn].append(_safe_str(r.get("review_reason")) or "Manual label suspect")

    statuses, issue_texts, reason_texts, grades, grade_reasons, trusted_a, usable_ab = [], [], [], [], [], [], []
    for _, r in out.iterrows():
        fn = str(r["filename"])
        issues = set(issue_map.get(fn, set()))
        conf = str(r.get("source_match_confidence") or "UNRESOLVED").upper()
        pdirection = _safe_str(r.get("idx_pass_direction"))
        geometry_ok = bool(r.get("geometry_auditable"))
        if conf not in {"HIGH", "MEDIUM"}:
            issues.add("SOURCE_MATCH_UNCERTAIN")
        if conf in {"HIGH", "MEDIUM"} and pdirection in {"CROSSING_PASS", "UNKNOWN_PASS"}:
            issues.add("POSSIBLE_WRONG_PASS")
            reason_map[fn].append(f"Geometry classification={pdirection}; manual product nevertheless uses this source frame")
        if conf in {"HIGH", "MEDIUM"} and not geometry_ok:
            issues.add("GEOMETRY_NOT_AUDITABLE")
            reason_map[fn].append("Trusted raw match exists, but run/chainage metadata are insufficient for geometry continuity audit")

        strong = issues & {"TRUE_DUPLICATE_SUSPECT", "POSSIBLE_SEQUENCE_JUMP", "POSSIBLE_BAD_ORDER_OR_LABEL", "POSSIBLE_WRONG_PASS", "MANUAL_LABEL_SUSPECT"}
        mild = issues & {"WEAK_VISUAL_OVERLAP", "VISUAL_REVIEW", "GEOMETRY_JUMP_VISUAL_OK", "POSSIBLE_REDUNDANT_FRAME"}
        unresolved = "SOURCE_MATCH_UNCERTAIN" in issues
        insufficient = "GEOMETRY_NOT_AUDITABLE" in issues

        if strong:
            status = "REVIEW_STRONG"
            grade = "REVIEW"
            greason = "Strong QA flag requires manual review before use as calibration truth"
        elif unresolved:
            status = "UNRESOLVED_SOURCE_MATCH"
            grade = "REVIEW"
            greason = "Manual-final to raw mapping is not sufficiently reliable"
        elif insufficient:
            status = "REVIEW"
            grade = "REVIEW"
            greason = "Geometry audit is incomplete"
        elif mild:
            status = "REVIEW"
            grade = "B"
            greason = "Source/pass/geometry are auditable; only mild QA evidence remains"
        else:
            status = "TRUSTED_CANDIDATE"
            grade = "A" if (conf in {"HIGH","MEDIUM"} and pdirection in {"FORWARD_PASS","BACKWARD_PASS"} and geometry_ok) else "REVIEW"
            greason = "High-confidence source match, normal pass, auditable geometry, no material QA flags" if grade == "A" else "Insufficient independent evidence for Grade A"

        statuses.append(status)
        issue_texts.append(";".join(sorted(issues)) if issues else None)
        reason_texts.append(" | ".join(dict.fromkeys([x for x in reason_map.get(fn, []) if x])) or None)
        grades.append(grade)
        grade_reasons.append(greason)
        trusted_a.append(grade == "A")
        usable_ab.append(grade in {"A","B"})

    out["manual_audit_status"] = statuses
    out["manual_audit_flags"] = issue_texts
    out["manual_audit_reasons"] = reason_texts
    out["calibration_grade"] = grades
    out["calibration_grade_reason"] = grade_reasons
    out["trusted_for_calibration"] = trusted_a
    out["calibration_usable_A_or_B"] = usable_ab
    return out


def _thumb(path: Path, max_w: int = 900, max_h: int = 650):
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Image.open(path) as src:
            try:
                src.draft("RGB", (max_w * 2, max_h * 2))
            except Exception:
                pass
            im = src.convert("RGB")
            im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            return im.copy()


def render_pair_sheet(a_path: Path, b_path: Path, dest: Path, title: str, lines: list[str], quality: int = 88) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return "PILLOW_MISSING"
    if not a_path.exists() or not b_path.exists():
        return "SOURCE_MISSING"
    ia, ib = _thumb(a_path), _thumb(b_path)
    margin, header_h = 18, 150
    width = ia.width + ib.width + margin * 3
    height = max(ia.height, ib.height) + header_h + margin * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default(); small = ImageFont.load_default()
    draw.text((margin, 12), title, fill="black", font=font)
    y = 42
    for line in lines[:6]:
        draw.text((margin, y), str(line), fill="black", font=small); y += 18
    x1 = margin; x2 = margin * 2 + ia.width; y0 = header_h
    canvas.paste(ia, (x1, y0)); canvas.paste(ib, (x2, y0))
    draw.text((x1, y0 - 20), "A", fill="black", font=small)
    draw.text((x2, y0 - 20), "B", fill="black", font=small)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=quality, optimize=True)
    return "RENDERED"


def render_three_sheet(paths: list[Path], labels: list[str], dest: Path, title: str, lines: list[str], quality: int = 88) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return "PILLOW_MISSING"
    if len(paths) != 3 or any(not p.exists() for p in paths):
        return "SOURCE_MISSING"
    ims = [_thumb(p, 620, 520) for p in paths]
    margin, header_h = 18, 165
    width = sum(im.width for im in ims) + margin * 4
    height = max(im.height for im in ims) + header_h + margin * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 19); small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default(); small = ImageFont.load_default()
    draw.text((margin, 12), title, fill="black", font=font)
    y = 40
    for line in lines[:7]:
        draw.text((margin, y), str(line), fill="black", font=small); y += 18
    x = margin
    for im, lab in zip(ims, labels):
        draw.text((x, header_h - 20), lab, fill="black", font=small)
        canvas.paste(im, (x, header_h)); x += im.width + margin
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=quality, optimize=True)
    return "RENDERED"


def render_reviews(
    images: pd.DataFrame,
    transitions: pd.DataFrame,
    span_cont: pd.DataFrame,
    high_similarity_pairs: pd.DataFrame,
    label_suspects: pd.DataFrame,
    out_dir: Path,
    limit: int,
    jpeg_quality: int,
) -> pd.DataFrame:
    if limit == 0:
        return pd.DataFrame()
    path_map = dict(zip(images["filename"].astype(str), images["manual_path"].astype(str)))
    rows = []
    remaining = max(0, int(limit))

    def pair_category(status: str) -> str:
        if "DUPLICATE" in status: return "DUPLICATE"
        if "PASS" in status: return "WRONG_PASS"
        if "JUMP" in status or "BAD_ORDER" in status: return "SEQUENCE"
        if "WEAK" in status or "VISUAL" in status: return "WEAK_CONTINUITY"
        return "REVIEW"

    for _, r in transitions.iterrows():
        if remaining <= 0: break
        st = str(r.get("audit_status"))
        if st == "CONTINUOUS": continue
        a, b = _safe_str(r.get("image_a")), _safe_str(r.get("image_b"))
        if not a or not b: continue
        cat = pair_category(st)
        dest = out_dir / "review" / cat / f"{st}__{Path(a).stem}__{Path(b).stem}.jpg"
        result = render_pair_sheet(
            Path(path_map[a]), Path(path_map[b]), dest,
            f"Manual 163 audit: {st}",
            [
                f"A: {a} | span={r.get('span_a')}",
                f"B: {b} | span={r.get('span_b')}",
                f"visual_score={r.get('visual_score')}  robust_z={r.get('visual_robust_z')}",
                f"chainage_gap={r.get('chainage_gap_abs')}  eo_delta={r.get('eo_order_delta')}",
                f"reason: {r.get('review_reason')}",
            ], jpeg_quality,
        )
        rows.append({"category": cat, "status": st, "image_a": a, "image_b": b, "review_path": str(dest), "render_status": result})
        remaining -= 1

    for _, r in span_cont.iterrows():
        if remaining <= 0: break
        st = str(r.get("audit_status"))
        if st not in {"WEAK_SPAN_CONTINUITY"}: continue
        a, b = _safe_str(r.get("image_a")), _safe_str(r.get("image_b"))
        if not a or not b or a not in path_map or b not in path_map: continue
        dest = out_dir / "review" / "SPAN_CONTINUITY" / f"{st}__{Path(a).stem}__{Path(b).stem}.jpg"
        result = render_pair_sheet(
            Path(path_map[a]), Path(path_map[b]), dest,
            f"Manual span continuity: {st}",
            [f"{r.get('span_a')} -> {r.get('span_b')}", f"visual_score={r.get('visual_score')}", f"reason: {r.get('review_reason')}"], jpeg_quality,
        )
        rows.append({"category": "SPAN_CONTINUITY", "status": st, "image_a": a, "image_b": b, "review_path": str(dest), "render_status": result})
        remaining -= 1

    for _, r in high_similarity_pairs.iterrows():
        if remaining <= 0: break
        st = str(r.get("audit_status"))
        if st == "EXPECTED_HIGH_OVERLAP":
            continue
        a, b = str(r["image_a"]), str(r["image_b"])
        cat = "DUPLICATE" if st == "TRUE_DUPLICATE_SUSPECT" else "REDUNDANCY"
        dest = out_dir / "review" / cat / f"{st}__{Path(a).stem}__{Path(b).stem}.jpg"
        result = render_pair_sheet(
            Path(path_map[a]), Path(path_map[b]), dest,
            f"Manual 163 audit: {st}",
            [
                f"A span={r.get('span_a')} | B span={r.get('span_b')}",
                f"global={r.get('global_similarity')} dense={r.get('dense_best_mean')}",
                f"same_raw={r.get('same_raw_source')} same_run={r.get('same_run')}",
                f"eo_gap={r.get('eo_order_gap')} span_gap={r.get('span_gap')} chainage_gap={r.get('chainage_gap')}",
                f"reason: {r.get('review_reason')}",
            ], jpeg_quality,
        )
        rows.append({"category": cat, "status": st, "image_a": a, "image_b": b, "review_path": str(dest), "render_status": result})
        remaining -= 1

    for _, r in label_suspects.iterrows():
        if remaining <= 0: break
        a = str(r["filename"]); adj = str(r["best_adjacent_image"]); non = str(r["best_nonadjacent_image"])
        if any(x not in path_map for x in [a, adj, non]): continue
        dest = out_dir / "review" / "LABEL_SUSPECT" / f"LABEL__{Path(a).stem}.jpg"
        result = render_three_sheet(
            [Path(path_map[a]), Path(path_map[adj]), Path(path_map[non])],
            ["MANUAL IMAGE", "BEST ADJACENT SPAN", "BEST NON-ADJACENT"],
            dest,
            "Manual 163 audit: MANUAL_LABEL_SUSPECT",
            [
                f"manual span={r.get('manual_span')}",
                f"adjacent={r.get('best_adjacent_span')} score={r.get('adjacent_visual_score')}",
                f"non-adjacent={r.get('best_nonadjacent_span')} score={r.get('nonadjacent_visual_score')}",
                f"margin={r.get('score_margin_nonadj_minus_adj')}",
            ], jpeg_quality,
        )
        rows.append({"category": "LABEL_SUSPECT", "status": "MANUAL_LABEL_SUSPECT", "image_a": a, "image_b": non, "review_path": str(dest), "render_status": result})
        remaining -= 1

    return pd.DataFrame(rows)


def make_summary(images: pd.DataFrame, transitions: pd.DataFrame, span_cont: pd.DataFrame, high_similarity_pairs: pd.DataFrame, label_suspects: pd.DataFrame, feature_meta: dict) -> dict:
    grade_counts = dict(Counter(images.get("calibration_grade", pd.Series(dtype=str))))
    trusted_a = int((images.get("calibration_grade", pd.Series(index=images.index, dtype=str)) == "A").sum())
    usable_ab = int(images.get("calibration_grade", pd.Series(index=images.index, dtype=str)).isin(["A","B"]).sum())
    sim_counts = dict(Counter(high_similarity_pairs["audit_status"])) if not high_similarity_pairs.empty else {}
    return {
        "manual_images": int(len(images)),
        "manual_unique_spans": int(images["span_id"].nunique(dropna=True)),
        "source_matches_high": int(images["source_match_confidence"].eq("HIGH").sum()),
        "source_matches_medium": int(images["source_match_confidence"].eq("MEDIUM").sum()),
        "source_matches_unresolved_or_low": int((~images["source_match_confidence"].isin(["HIGH", "MEDIUM"])).sum()),
        "geometry_auditable_images": int(images["geometry_auditable"].sum()),
        "geometry_transitions": int(len(transitions)),
        "transition_status_counts": dict(Counter(transitions["audit_status"])) if not transitions.empty else {},
        "transition_overlap_class_counts": dict(Counter(transitions["overlap_class"])) if not transitions.empty and "overlap_class" in transitions else {},
        "span_continuity_status_counts": dict(Counter(span_cont["audit_status"])) if not span_cont.empty else {},
        "high_similarity_pair_counts": sim_counts,
        "true_duplicate_suspects": int(sim_counts.get("TRUE_DUPLICATE_SUSPECT", 0)),
        "possible_redundant_pairs": int(sim_counts.get("POSSIBLE_REDUNDANT_FRAME", 0)),
        "expected_high_overlap_pairs": int(sim_counts.get("EXPECTED_HIGH_OVERLAP", 0)),
        "raw_mapping_collisions_untrusted": int(sim_counts.get("RAW_MAPPING_COLLISION_UNTRUSTED", 0)),
        "label_suspects": int(len(label_suspects)),
        "manual_image_status_counts": dict(Counter(images["manual_audit_status"])),
        "calibration_grade_counts": grade_counts,
        "trusted_grade_A": trusted_a,
        "trusted_grade_A_percent": (100.0 * trusted_a / len(images)) if len(images) else None,
        "calibration_pool_A_or_B": usable_ab,
        "calibration_pool_A_or_B_percent": (100.0 * usable_ab / len(images)) if len(images) else None,
        "trusted_for_calibration": trusted_a,
        "trusted_for_calibration_percent": (100.0 * trusted_a / len(images)) if len(images) else None,
        "dino_model": feature_meta.get("model_id"),
        "dino_device": feature_meta.get("device"),
        "dino_features_cached": feature_meta.get("cached"),
        "dino_features_computed": feature_meta.get("computed"),
        "method_note": "DINO and geometry are QA evidence only. LOW/unresolved manual->raw fingerprint collisions are not duplicate evidence. Expected local high overlap is neutral. Grade A is the clean calibration subset; Grade B has only mild QA flags; REVIEW is excluded from calibration.",
    }


def summary_csv(summary: dict) -> pd.DataFrame:
    rows = []
    for k, v in summary.items():
        if isinstance(v, dict):
            for sk, sv in v.items(): rows.append({"metric": f"{k}:{sk}", "value": sv})
        elif isinstance(v, (str, int, float, bool)) or v is None:
            rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)


def write_html_report(summary: dict, images: pd.DataFrame, transitions: pd.DataFrame, span_cont: pd.DataFrame, high_similarity_pairs: pd.DataFrame, label_suspects: pd.DataFrame, path: Path):
    def esc(x): return html.escape(str(x))
    cards = [
        ("Manual images", summary.get("manual_images")),
        ("Unique spans", summary.get("manual_unique_spans")),
        ("Geometry-auditable", summary.get("geometry_auditable_images")),
        ("Calibration Grade A", summary.get("trusted_grade_A")),
        ("Calibration pool A+B", summary.get("calibration_pool_A_or_B")),
        ("True duplicate suspects", summary.get("true_duplicate_suspects")),
        ("Expected high-overlap pairs", summary.get("expected_high_overlap_pairs")),
        ("Untrusted raw-map collisions", summary.get("raw_mapping_collisions_untrusted")),
        ("Label suspects", summary.get("label_suspects")),
    ]
    def table_from_counts(title, counts):
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k,v in counts.items())
        return f"<h2>{esc(title)}</h2><table><tr><th>Status</th><th>Count</th></tr>{rows}</table>"

    flagged = images[~images.get("calibration_grade", pd.Series(index=images.index, dtype=str)).eq("A")].copy().head(80)
    cols = ["filename","span_id","source_match_confidence","idx_pass_direction","idx_primary_span","calibration_grade","manual_audit_status","manual_audit_flags"]
    cols = [c for c in cols if c in flagged.columns]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = ""
    for _, r in flagged.iterrows():
        body += "<tr>" + "".join(f"<td>{esc(r.get(c,''))}</td>" for c in cols) + "</tr>"

    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>Manual 163 internal audit</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#222}}.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}.card{{border:1px solid #bbb;border-radius:8px;padding:12px 16px;min-width:175px;background:#fafafa}}.card b{{display:block;font-size:13px;color:#555}}.card span{{display:block;font-size:24px;margin-top:4px}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 24px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#eee}}</style></head><body>
<h1>Manual Continuous Oblique 163 internal audit</h1>
<p>This audit does not assume the manual product is perfect ground truth. DINOv3, raw/EO mapping and geometry are used as independent QA signals. No image is automatically deleted or relabelled.</p>
<div class='cards'>{''.join(f"<div class='card'><b>{esc(k)}</b><span>{esc(v)}</span></div>" for k,v in cards)}</div>
<p>Grade A: <b>{summary.get('trusted_grade_A_percent',0):.2f}%</b>. Calibration pool A+B: <b>{summary.get('calibration_pool_A_or_B_percent',0):.2f}%</b>.</p>
{table_from_counts('Calibration grades', summary.get('calibration_grade_counts',{}))}
{table_from_counts('Manual image audit status', summary.get('manual_image_status_counts',{}))}
{table_from_counts('High-similarity pair classification', summary.get('high_similarity_pair_counts',{}))}
{table_from_counts('Geometry-run transition status', summary.get('transition_status_counts',{}))}
{table_from_counts('Adjacent-span continuity status', summary.get('span_continuity_status_counts',{}))}
<h2>First images requiring review</h2><table><tr>{head}</tr>{body}</table>
<h2>Interpretation</h2><ul>
<li><b>Grade A</b> means HIGH/MEDIUM raw-source match, normal forward/backward pass, auditable geometry and no material QA flags.</li>
<li><b>Grade B</b> keeps images with only mild QA evidence such as visual review or possible local redundancy; these are usable with caution.</li>
<li><b>REVIEW</b> is excluded from calibration until inspected; it is not proof that the manual product is wrong.</li>
<li><b>EXPECTED_HIGH_OVERLAP</b> is normal for consecutive continuous-oblique frames and does not reduce trust.</li>
<li><b>RAW_MAPPING_COLLISION_UNTRUSTED</b> means two manual finals fingerprinted to the same raw name but at least one mapping is LOW/unresolved; this is not treated as duplicate evidence.</li>
<li><b>SOURCE_MATCH_UNCERTAIN</b> means the existing manual-final to raw-image benchmark is insufficiently confident; these rows are not suitable for calibrating the AI process.</li>
<li>DINO similarity is used for continuity, duplicate and label-suspicion signals; geometry/pass metadata remain separate evidence.</li>
</ul>
<p>Full detail: <code>manual_image_audit.csv</code>, <code>geometry_run_transitions.csv</code>, <code>span_continuity.csv</code>, <code>high_similarity_pairs.csv</code>, <code>duplicate_candidates.csv</code>, <code>label_suspects.csv</code>. Review images are under <code>review/</code>.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def seed_dino_cache(source_dir: Path | None, out_dir: Path) -> int:
    """Copy compatible prior-audit feature-cache files into the v0.3 cache directory.

    The underlying DINO helper still validates source size/mtime before using a cache file,
    so stale entries are harmless and will be recomputed.
    """
    if source_dir is None or not source_dir.exists():
        return 0
    dest = out_dir / "qa" / "visual_continuity" / "feature_cache"
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in source_dir.glob("*.npz"):
        dst = dest / src.name
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            pass
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit manual EPIC 163 product using DINOv3 + geometry without assuming manual ground truth is perfect.")
    ap.add_argument("project_root")
    ap.add_argument("--ai-run", default=None, help="AI run folder basename. Default: latest _continuous_oblique_*v014* folder.")
    ap.add_argument("--continuous-script", default=None, help="Path to continuous_oblique_v014.py. Default: <project_root>/continuous_oblique_v014.py")
    ap.add_argument("--output", default=None, help="Output directory. Default: <project_root>/_audit_manual_163_v03")
    ap.add_argument("--feature-source", choices=["manual", "raw-when-trusted"], default="raw-when-trusted", help="DINO input source. Default raw-when-trusted to avoid manual annotation bias when raw mapping is reliable.")
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--dino-device", default="auto")
    ap.add_argument("--dino-image-size", type=int, default=224)
    ap.add_argument("--dino-batch-size", type=int, default=4)
    ap.add_argument("--dino-match-threshold", type=float, default=0.55)
    ap.add_argument("--dino-absolute-weak", type=float, default=0.58)
    ap.add_argument("--dino-robust-weak-z", type=float, default=-3.0)
    ap.add_argument("--dino-robust-review-z", type=float, default=-2.0)
    ap.add_argument("--dino-duplicate-global", type=float, default=0.995)
    ap.add_argument("--dino-duplicate-dense", type=float, default=0.95)
    ap.add_argument("--duplicate-local-eo-gap", type=int, default=3, help="EO-order separation still treated as local expected overlap. Default: 3")
    ap.add_argument("--duplicate-local-span-gap", type=int, default=1, help="Span-index separation still treated as local expected overlap. Default: 1")
    ap.add_argument("--duplicate-far-span-gap", type=int, default=4, help="Span separation that can support a true duplicate suspect when DINO similarity is extreme. Default: 4")
    ap.add_argument("--duplicate-far-chainage-factor", type=float, default=6.0, help="Far-pair threshold as multiple of typical run image step. Default: 6")
    ap.add_argument("--expected-overlap-quantile", type=float, default=0.80, help="Within-run global-similarity quantile used only to label normal EXPECTED_HIGH_OVERLAP pairs. Default: 0.80")
    ap.add_argument("--label-window", type=int, default=5)
    ap.add_argument("--label-suspect-margin", type=float, default=0.08)
    ap.add_argument("--review-limit", type=int, default=120)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--disable-dino-cache", action="store_true")
    ap.add_argument("--reuse-cache-from", default=None, help="Optional existing DINO feature_cache directory. Default: auto-reuse v0.2, then v0.1 cache when present.")
    ap.add_argument("--no-auto-reuse-prior-cache", action="store_true", help="Do not automatically seed v0.3 cache from a previous audit cache.")
    ap.add_argument("--max-images", type=int, default=None, help="Smoke test only: audit first N manual images by span/output ordering.")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project_root).resolve()
    paths = discover_paths(root, args.ai_run, args.continuous_script)
    out_dir = Path(args.output).resolve() if args.output else root / "_audit_manual_163_v03"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/9] Indexing manual 163 product...")
    manual = index_manual_folder(paths["manual_dir"])
    if manual.empty:
        raise RuntimeError("No manual 163 JPGs found")
    manual = manual.sort_values(["span_from", "output_id"], na_position="last").reset_index(drop=True)
    if args.max_images is not None:
        manual = manual.head(max(0, int(args.max_images))).copy()

    print("[2/9] Loading manual-final -> raw benchmark and geometry metadata...")
    bench = normalize_benchmark(_read_csv(paths["benchmark_csv"], required=True))
    idx = normalize_image_index(_read_csv(paths["image_index_csv"], required=True))
    reference_span_ids = load_reference_span_ids(_read_csv(paths.get("reference_spans_csv"), required=False))
    images = build_manual_audit_index(manual, bench, idx, paths["raw_dir"], args.feature_source)

    print("[3/9] Loading DINOv3 helpers from continuous_oblique_v014.py...")
    mod = load_continuous_module(paths["continuous_script"])

    print("[4/9] Extracting DINOv3 features for manual product...")
    cache_source = Path(args.reuse_cache_from).resolve() if args.reuse_cache_from else None
    if cache_source is None and not args.no_auto_reuse_prior_cache:
        for prior_name in ["_audit_manual_163_v02", "_audit_manual_163_v01"]:
            candidate_cache = root / prior_name / "qa" / "visual_continuity" / "feature_cache"
            if candidate_cache.exists():
                cache_source = candidate_cache
                break
    seeded_cache_files = 0 if args.disable_dino_cache else seed_dino_cache(cache_source, out_dir)
    if seeded_cache_files:
        print(f"  seeded {seeded_cache_files} feature-cache files from {cache_source}")
    dino_manifest = pd.DataFrame({
        "filename": images["filename"].astype(str),
        "raw_path": images["dino_source_path"].astype(str),
    })
    needed = set(dino_manifest["filename"].astype(str))
    features, feature_index, feature_meta = mod.extract_dinov3_features(
        dino_manifest, needed, out_dir, args.dino_model, args.dino_image_size,
        args.dino_batch_size, args.dino_device, not args.disable_dino_cache,
    )
    print(f"  features={len(features)}/{len(needed)} device={feature_meta.get('device')} cached={feature_meta.get('cached')} computed={feature_meta.get('computed')}")

    print("[5/9] Auditing geometry-run continuity...")
    transitions = build_geometry_transitions(images)
    transitions = attach_dino_to_transitions(
        mod, transitions, features,
        args.dino_match_threshold, args.dino_absolute_weak,
        args.dino_robust_weak_z, args.dino_robust_review_z,
        args.dino_duplicate_global, args.dino_duplicate_dense,
    )

    print("[6/9] Auditing adjacent-span continuity, duplicates and label suspects...")
    span_cont = build_span_continuity(
        images, features, mod, args.dino_match_threshold, args.dino_absolute_weak, reference_span_ids
    )
    high_similarity_pairs = build_duplicate_candidates(
        images, features, mod, args.dino_duplicate_global, args.dino_duplicate_dense, args.dino_match_threshold,
        args.duplicate_local_eo_gap, args.duplicate_local_span_gap,
        args.duplicate_far_span_gap, args.duplicate_far_chainage_factor,
        args.expected_overlap_quantile,
    )
    label_suspects = build_label_suspects(
        images, features, mod, args.dino_match_threshold, args.label_window,
        args.label_suspect_margin, args.dino_absolute_weak,
    )
    images = aggregate_image_status(images, transitions, high_similarity_pairs, label_suspects)

    print("[7/9] Writing audit tables...")
    images.to_csv(out_dir / "manual_image_audit.csv", index=False)
    transitions.to_csv(out_dir / "geometry_run_transitions.csv", index=False)
    span_cont.to_csv(out_dir / "span_continuity.csv", index=False)
    high_similarity_pairs.to_csv(out_dir / "high_similarity_pairs.csv", index=False)
    true_dups = high_similarity_pairs[high_similarity_pairs.get("audit_status", pd.Series(index=high_similarity_pairs.index, dtype=str)).eq("TRUE_DUPLICATE_SUSPECT")].copy() if not high_similarity_pairs.empty else pd.DataFrame()
    redundant = high_similarity_pairs[high_similarity_pairs.get("audit_status", pd.Series(index=high_similarity_pairs.index, dtype=str)).eq("POSSIBLE_REDUNDANT_FRAME")].copy() if not high_similarity_pairs.empty else pd.DataFrame()
    expected_overlap = high_similarity_pairs[high_similarity_pairs.get("audit_status", pd.Series(index=high_similarity_pairs.index, dtype=str)).eq("EXPECTED_HIGH_OVERLAP")].copy() if not high_similarity_pairs.empty else pd.DataFrame()
    raw_collisions = high_similarity_pairs[high_similarity_pairs.get("audit_status", pd.Series(index=high_similarity_pairs.index, dtype=str)).eq("RAW_MAPPING_COLLISION_UNTRUSTED")].copy() if not high_similarity_pairs.empty else pd.DataFrame()
    true_dups.to_csv(out_dir / "duplicate_candidates.csv", index=False)
    redundant.to_csv(out_dir / "redundant_frame_candidates.csv", index=False)
    expected_overlap.to_csv(out_dir / "expected_high_overlap_pairs.csv", index=False)
    raw_collisions.to_csv(out_dir / "raw_mapping_collisions.csv", index=False)
    label_suspects.to_csv(out_dir / "label_suspects.csv", index=False)
    missing_manual_spans = span_cont[span_cont.get("audit_status", pd.Series(index=span_cont.index, dtype=str)).eq("MISSING_NEXT_MANUAL_SPAN")].copy() if not span_cont.empty else pd.DataFrame()
    missing_manual_spans.to_csv(out_dir / "missing_manual_spans.csv", index=False)
    images[images["calibration_grade"].eq("A")].to_csv(out_dir / "trusted_manual_grade_A.csv", index=False)
    images[images["calibration_grade"].isin(["A","B"])].to_csv(out_dir / "calibration_pool_A_B.csv", index=False)
    images[images["calibration_grade"].eq("A")].to_csv(out_dir / "trusted_manual_subset.csv", index=False)
    feature_index.to_csv(out_dir / "dino_feature_index.csv", index=False)

    print("[8/9] Rendering review contact sheets...")
    review = pd.DataFrame()
    if not args.no_review and args.review_limit != 0:
        review = render_reviews(images, transitions, span_cont, high_similarity_pairs, label_suspects, out_dir, args.review_limit, args.jpeg_quality)
    review.to_csv(out_dir / "review_manifest.csv", index=False)

    print("[9/9] Writing summary and HTML report...")
    summary = make_summary(images, transitions, span_cont, high_similarity_pairs, label_suspects, feature_meta)
    summary.update({
        "reused_feature_cache_source": str(cache_source) if cache_source else None,
        "seeded_feature_cache_files": int(seeded_cache_files),
        "reference_topology_span_count": int(len(reference_span_ids)),
        "missing_manual_spans_expected_by_topology": int((span_cont.get("audit_status", pd.Series(index=span_cont.index, dtype=str)) == "MISSING_NEXT_MANUAL_SPAN").sum()) if not span_cont.empty else 0,
        "numerical_gaps_absent_from_topology": int((span_cont.get("audit_status", pd.Series(index=span_cont.index, dtype=str)) == "TOPOLOGY_NO_SUCH_SPAN").sum()) if not span_cont.empty else 0,
        "project_root": str(root),
        "manual_dir": str(paths["manual_dir"]),
        "ai_run": str(paths["ai_root"]),
        "feature_source_mode": args.feature_source,
        "output": str(out_dir),
    })
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary_csv(summary).to_csv(out_dir / "audit_summary.csv", index=False)
    write_html_report(summary, images, transitions, span_cont, high_similarity_pairs, label_suspects, out_dir / "report.html")

    print("\nManual 163 internal audit complete")
    print(f"  Manual images:                 {summary['manual_images']}")
    print(f"  Geometry-auditable images:     {summary['geometry_auditable_images']}")
    print(f"  Calibration Grade A:           {summary['trusted_grade_A']} ({summary['trusted_grade_A_percent']:.2f}%)")
    print(f"  Calibration pool A+B:          {summary['calibration_pool_A_or_B']} ({summary['calibration_pool_A_or_B_percent']:.2f}%)")
    print(f"  Expected high-overlap pairs:   {summary['expected_high_overlap_pairs']}")
    print(f"  Possible redundant pairs:      {summary['possible_redundant_pairs']}")
    print(f"  Untrusted raw-map collisions:  {summary['raw_mapping_collisions_untrusted']}")
    print(f"  True duplicate suspects:       {summary['true_duplicate_suspects']}")
    print(f"  Label suspects:                {summary['label_suspects']}")
    print(f"  Transition statuses:           {summary['transition_status_counts']}")
    print(f"  Span continuity statuses:      {summary['span_continuity_status_counts']}")
    print(f"  Output:                        {out_dir}")
    print(f"  Report:                        {out_dir / 'report.html'}")
    print("\nImportant: flags are review candidates, not automatic proof that the manual product is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
