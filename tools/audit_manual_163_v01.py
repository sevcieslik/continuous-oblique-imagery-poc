#!/usr/bin/env python
r"""
Audit a manually produced EPIC 163 Continuous Oblique Imagery product for internal continuity.

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
<output>/duplicate_candidates.csv
<output>/label_suspects.csv
<output>/trusted_manual_subset.csv
<output>/dino_feature_index.csv
<output>/review_manifest.csv
<output>/report.html
<output>/review/<CATEGORY>/*.jpg
<output>/qa/visual_continuity/feature_cache/*.npz

Example
-------
python audit_manual_163_v01.py "D:\\Sev\\Obliques\\DH1"

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

    statuses, reasons = [], []
    for _, r in out.iterrows():
        if r["visual_analysis_status"] != "ANALYZED":
            statuses.append(r["visual_analysis_status"]); reasons.append("DINO metrics unavailable"); continue
        score = float(r["visual_score"])
        z = float(r["visual_robust_z"]) if pd.notna(r["visual_robust_z"]) else 0.0
        glob = float(r["global_similarity"])
        dense = float(r["dense_best_mean"])
        possible_dup = glob >= duplicate_global and dense >= duplicate_dense
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

        if same_raw or possible_dup:
            st, reason = "POSSIBLE_DUPLICATE", "Same raw image or extremely high DINO similarity"
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
    return out


def build_span_continuity(images: pd.DataFrame, features: dict, mod, match_threshold: float, absolute_weak: float) -> pd.DataFrame:
    good = images[images["parse_status"].eq("OK")].copy()
    span_groups = {int(k): g.copy() for k, g in good.dropna(subset=["span_from"]).groupby(good["span_from"].astype(int))}
    if not span_groups:
        return pd.DataFrame()
    keys = sorted(span_groups)
    rows = []
    for sf in keys:
        nxt = sf + 1
        if nxt not in span_groups:
            # do not call the last span a gap
            if sf != max(keys):
                rows.append({
                    "span_a": f"{sf}-{sf+1}", "span_b": f"{nxt}-{nxt+1}",
                    "image_a": None, "image_b": None, "best_pair_found": False,
                    "audit_status": "MISSING_NEXT_MANUAL_SPAN", "review_reason": "No manual images exist for the next numerical span",
                })
            continue
        ga, gb = span_groups[sf], span_groups[nxt]
        candidates = []
        for _, a in ga.iterrows():
            for _, b in gb.iterrows():
                fa, fb = features.get(str(a["filename"])), features.get(str(b["filename"]))
                if fa is None or fb is None:
                    continue
                # global similarity is cheap and sufficient for choosing the best candidate pair
                gsim = float(np.dot(
                    mod._normalise_rows(np.asarray(fa["global"])[None, :])[0],
                    mod._normalise_rows(np.asarray(fb["global"])[None, :])[0],
                ))
                candidates.append((gsim, a, b))
        if not candidates:
            rows.append({
                "span_a": f"{sf}-{sf+1}", "span_b": f"{nxt}-{nxt+1}",
                "image_a": None, "image_b": None, "best_pair_found": False,
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
            "span_a": f"{sf}-{sf+1}", "span_b": f"{nxt}-{nxt+1}",
            "image_a": a["filename"], "image_b": b["filename"], "best_pair_found": True,
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


def build_duplicate_candidates(images: pd.DataFrame, features: dict, mod, duplicate_global: float, duplicate_dense: float, match_threshold: float) -> pd.DataFrame:
    names = [str(x) for x in images["filename"].tolist() if str(x) in features]
    if len(names) < 2:
        return pd.DataFrame()
    globals_mat = np.vstack([mod._normalise_rows(np.asarray(features[n]["global"])[None, :])[0] for n in names])
    sim = globals_mat @ globals_mat.T
    rows = []
    same_raw_map = dict(zip(images["filename"].astype(str), images["matched_raw_filename"].map(_safe_str)))
    span_map = dict(zip(images["filename"].astype(str), images["span_id"].map(_safe_str)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g = float(sim[i, j])
            same_raw = same_raw_map.get(names[i]) is not None and same_raw_map.get(names[i]) == same_raw_map.get(names[j])
            # Dense comparison only for very high global similarity or exact same trusted raw reuse.
            if g < max(duplicate_global - 0.01, 0.985) and not same_raw:
                continue
            m = pair_metrics(mod, features[names[i]], features[names[j]], match_threshold)
            dense = float(m.get("dense_best_mean", np.nan))
            confirmed = same_raw or (g >= duplicate_global and math.isfinite(dense) and dense >= duplicate_dense)
            if not confirmed:
                continue
            rows.append({
                "image_a": names[i], "image_b": names[j],
                "span_a": span_map.get(names[i]), "span_b": span_map.get(names[j]),
                "same_raw_source": same_raw,
                "audit_status": "POSSIBLE_DUPLICATE",
                "review_reason": "Same trusted raw source reused" if same_raw else "Extremely high DINO global and dense similarity",
                **m,
            })
    return pd.DataFrame(rows).sort_values(["same_raw_source", "global_similarity"], ascending=[False, False]) if rows else pd.DataFrame()


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
    duplicates: pd.DataFrame,
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
    for _, r in duplicates.iterrows():
        for col in ["image_a", "image_b"]:
            fn = _safe_str(r.get(col))
            if fn:
                issue_map[fn].add("POSSIBLE_DUPLICATE")
                reason_map[fn].append(_safe_str(r.get("review_reason")) or "Possible duplicate")
    for _, r in label_suspects.iterrows():
        fn = _safe_str(r.get("filename"))
        if fn:
            issue_map[fn].add("MANUAL_LABEL_SUSPECT")
            reason_map[fn].append(_safe_str(r.get("review_reason")) or "Manual label suspect")

    statuses, issue_texts, reason_texts, trusted = [], [], [], []
    for _, r in out.iterrows():
        fn = str(r["filename"])
        issues = set(issue_map.get(fn, set()))
        conf = str(r.get("source_match_confidence") or "UNRESOLVED").upper()
        pdirection = _safe_str(r.get("idx_pass_direction"))
        if conf not in {"HIGH", "MEDIUM"}:
            issues.add("SOURCE_MATCH_UNCERTAIN")
        if conf in {"HIGH", "MEDIUM"} and pdirection in {"CROSSING_PASS", "UNKNOWN_PASS"}:
            issues.add("POSSIBLE_WRONG_PASS")
            reason_map[fn].append(f"Geometry classification={pdirection}; manual product nevertheless uses this source frame")

        strong = issues & {"POSSIBLE_DUPLICATE", "POSSIBLE_SEQUENCE_JUMP", "POSSIBLE_BAD_ORDER_OR_LABEL", "POSSIBLE_WRONG_PASS", "MANUAL_LABEL_SUSPECT"}
        review = issues & {"WEAK_VISUAL_OVERLAP", "VISUAL_REVIEW", "GEOMETRY_JUMP_VISUAL_OK"}
        if strong:
            status = "REVIEW_STRONG"
        elif review:
            status = "REVIEW"
        elif "SOURCE_MATCH_UNCERTAIN" in issues:
            status = "UNRESOLVED_SOURCE_MATCH"
        else:
            status = "TRUSTED_CANDIDATE"
        is_trusted = (status == "TRUSTED_CANDIDATE" and conf in {"HIGH", "MEDIUM"} and pdirection in {"FORWARD_PASS", "BACKWARD_PASS"} and bool(r.get("geometry_auditable")))
        statuses.append(status)
        issue_texts.append(";".join(sorted(issues)) if issues else None)
        reason_texts.append(" | ".join(dict.fromkeys([x for x in reason_map.get(fn, []) if x])) or None)
        trusted.append(bool(is_trusted))
    out["manual_audit_status"] = statuses
    out["manual_audit_flags"] = issue_texts
    out["manual_audit_reasons"] = reason_texts
    out["trusted_for_calibration"] = trusted
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
    duplicates: pd.DataFrame,
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

    for _, r in duplicates.iterrows():
        if remaining <= 0: break
        a, b = str(r["image_a"]), str(r["image_b"])
        dest = out_dir / "review" / "DUPLICATE" / f"DUP__{Path(a).stem}__{Path(b).stem}.jpg"
        result = render_pair_sheet(
            Path(path_map[a]), Path(path_map[b]), dest,
            "Manual 163 audit: POSSIBLE_DUPLICATE",
            [f"A span={r.get('span_a')} | B span={r.get('span_b')}", f"global={r.get('global_similarity')} dense={r.get('dense_best_mean')}", f"same_raw={r.get('same_raw_source')}"], jpeg_quality,
        )
        rows.append({"category": "DUPLICATE", "status": "POSSIBLE_DUPLICATE", "image_a": a, "image_b": b, "review_path": str(dest), "render_status": result})
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


def make_summary(images: pd.DataFrame, transitions: pd.DataFrame, span_cont: pd.DataFrame, duplicates: pd.DataFrame, label_suspects: pd.DataFrame, feature_meta: dict) -> dict:
    trusted = int(images["trusted_for_calibration"].sum()) if "trusted_for_calibration" in images else 0
    return {
        "manual_images": int(len(images)),
        "manual_unique_spans": int(images["span_id"].nunique(dropna=True)),
        "source_matches_high": int(images["source_match_confidence"].eq("HIGH").sum()),
        "source_matches_medium": int(images["source_match_confidence"].eq("MEDIUM").sum()),
        "source_matches_unresolved_or_low": int((~images["source_match_confidence"].isin(["HIGH", "MEDIUM"])).sum()),
        "geometry_auditable_images": int(images["geometry_auditable"].sum()),
        "geometry_transitions": int(len(transitions)),
        "transition_status_counts": dict(Counter(transitions["audit_status"])) if not transitions.empty else {},
        "span_continuity_status_counts": dict(Counter(span_cont["audit_status"])) if not span_cont.empty else {},
        "duplicate_candidates": int(len(duplicates)),
        "label_suspects": int(len(label_suspects)),
        "manual_image_status_counts": dict(Counter(images["manual_audit_status"])),
        "trusted_for_calibration": trusted,
        "trusted_for_calibration_percent": (100.0 * trusted / len(images)) if len(images) else None,
        "dino_model": feature_meta.get("model_id"),
        "dino_device": feature_meta.get("device"),
        "dino_features_cached": feature_meta.get("cached"),
        "dino_features_computed": feature_meta.get("computed"),
        "method_note": "DINO and geometry are QA evidence only. Flags are review candidates, not automatic corrections.",
    }


def summary_csv(summary: dict) -> pd.DataFrame:
    rows = []
    for k, v in summary.items():
        if isinstance(v, dict):
            for sk, sv in v.items(): rows.append({"metric": f"{k}:{sk}", "value": sv})
        elif isinstance(v, (str, int, float, bool)) or v is None:
            rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)


def write_html_report(summary: dict, images: pd.DataFrame, transitions: pd.DataFrame, span_cont: pd.DataFrame, duplicates: pd.DataFrame, label_suspects: pd.DataFrame, path: Path):
    def esc(x): return html.escape(str(x))
    cards = [
        ("Manual images", summary.get("manual_images")),
        ("Unique spans", summary.get("manual_unique_spans")),
        ("Geometry-auditable", summary.get("geometry_auditable_images")),
        ("Trusted calibration subset", summary.get("trusted_for_calibration")),
        ("Duplicate candidates", summary.get("duplicate_candidates")),
        ("Label suspects", summary.get("label_suspects")),
    ]
    def table_from_counts(title, counts):
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k,v in counts.items())
        return f"<h2>{esc(title)}</h2><table><tr><th>Status</th><th>Count</th></tr>{rows}</table>"

    flagged = images[~images["manual_audit_status"].eq("TRUSTED_CANDIDATE")].copy().head(80)
    cols = ["filename","span_id","source_match_confidence","idx_pass_direction","idx_primary_span","manual_audit_status","manual_audit_flags"]
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
<p>Trusted calibration subset: <b>{summary.get('trusted_for_calibration_percent',0):.2f}%</b> of manual images.</p>
{table_from_counts('Manual image audit status', summary.get('manual_image_status_counts',{}))}
{table_from_counts('Geometry-run transition status', summary.get('transition_status_counts',{}))}
{table_from_counts('Adjacent-span continuity status', summary.get('span_continuity_status_counts',{}))}
<h2>First images requiring review</h2><table><tr>{head}</tr>{body}</table>
<h2>Interpretation</h2><ul>
<li><b>TRUSTED_CANDIDATE</b> means the image has a HIGH/MEDIUM raw-source match, belongs to a normal forward/backward pass, and no strong audit flag was raised.</li>
<li><b>REVIEW</b> / <b>REVIEW_STRONG</b> are review queues, not proof of manual error.</li>
<li><b>SOURCE_MATCH_UNCERTAIN</b> means the existing manual-final to raw-image benchmark is insufficiently confident; these rows are not suitable for calibrating the AI process.</li>
<li>DINO similarity is used for continuity, duplicate and label-suspicion signals; geometry/pass metadata remain separate evidence.</li>
</ul>
<p>Full detail: <code>manual_image_audit.csv</code>, <code>geometry_run_transitions.csv</code>, <code>span_continuity.csv</code>, <code>duplicate_candidates.csv</code>, <code>label_suspects.csv</code>. Review images are under <code>review/</code>.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit manual EPIC 163 product using DINOv3 + geometry without assuming manual ground truth is perfect.")
    ap.add_argument("project_root")
    ap.add_argument("--ai-run", default=None, help="AI run folder basename. Default: latest _continuous_oblique_*v014* folder.")
    ap.add_argument("--continuous-script", default=None, help="Path to continuous_oblique_v014.py. Default: <project_root>/continuous_oblique_v014.py")
    ap.add_argument("--output", default=None, help="Output directory. Default: <project_root>/_audit_manual_163_v01")
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
    ap.add_argument("--label-window", type=int, default=5)
    ap.add_argument("--label-suspect-margin", type=float, default=0.08)
    ap.add_argument("--review-limit", type=int, default=120)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--disable-dino-cache", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="Smoke test only: audit first N manual images by span/output ordering.")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project_root).resolve()
    paths = discover_paths(root, args.ai_run, args.continuous_script)
    out_dir = Path(args.output).resolve() if args.output else root / "_audit_manual_163_v01"
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
    images = build_manual_audit_index(manual, bench, idx, paths["raw_dir"], args.feature_source)

    print("[3/9] Loading DINOv3 helpers from continuous_oblique_v014.py...")
    mod = load_continuous_module(paths["continuous_script"])

    print("[4/9] Extracting DINOv3 features for manual product...")
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
    span_cont = build_span_continuity(images, features, mod, args.dino_match_threshold, args.dino_absolute_weak)
    duplicates = build_duplicate_candidates(
        images, features, mod, args.dino_duplicate_global, args.dino_duplicate_dense, args.dino_match_threshold
    )
    label_suspects = build_label_suspects(
        images, features, mod, args.dino_match_threshold, args.label_window,
        args.label_suspect_margin, args.dino_absolute_weak,
    )
    images = aggregate_image_status(images, transitions, duplicates, label_suspects)

    print("[7/9] Writing audit tables...")
    images.to_csv(out_dir / "manual_image_audit.csv", index=False)
    transitions.to_csv(out_dir / "geometry_run_transitions.csv", index=False)
    span_cont.to_csv(out_dir / "span_continuity.csv", index=False)
    duplicates.to_csv(out_dir / "duplicate_candidates.csv", index=False)
    label_suspects.to_csv(out_dir / "label_suspects.csv", index=False)
    images[images["trusted_for_calibration"].eq(True)].to_csv(out_dir / "trusted_manual_subset.csv", index=False)
    feature_index.to_csv(out_dir / "dino_feature_index.csv", index=False)

    print("[8/9] Rendering review contact sheets...")
    review = pd.DataFrame()
    if not args.no_review and args.review_limit != 0:
        review = render_reviews(images, transitions, span_cont, duplicates, label_suspects, out_dir, args.review_limit, args.jpeg_quality)
    review.to_csv(out_dir / "review_manifest.csv", index=False)

    print("[9/9] Writing summary and HTML report...")
    summary = make_summary(images, transitions, span_cont, duplicates, label_suspects, feature_meta)
    summary.update({
        "project_root": str(root),
        "manual_dir": str(paths["manual_dir"]),
        "ai_run": str(paths["ai_root"]),
        "feature_source_mode": args.feature_source,
        "output": str(out_dir),
    })
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary_csv(summary).to_csv(out_dir / "audit_summary.csv", index=False)
    write_html_report(summary, images, transitions, span_cont, duplicates, label_suspects, out_dir / "report.html")

    print("\nManual 163 internal audit complete")
    print(f"  Manual images:                 {summary['manual_images']}")
    print(f"  Geometry-auditable images:     {summary['geometry_auditable_images']}")
    print(f"  Trusted calibration subset:    {summary['trusted_for_calibration']} ({summary['trusted_for_calibration_percent']:.2f}%)")
    print(f"  Duplicate candidates:          {summary['duplicate_candidates']}")
    print(f"  Label suspects:                {summary['label_suspects']}")
    print(f"  Transition statuses:           {summary['transition_status_counts']}")
    print(f"  Span continuity statuses:      {summary['span_continuity_status_counts']}")
    print(f"  Output:                        {out_dir}")
    print(f"  Report:                        {out_dir / 'report.html'}")
    print("\nImportant: flags are review candidates, not automatic proof that the manual product is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
