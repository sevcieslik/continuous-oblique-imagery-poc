#!/usr/bin/env python
r"""
Compare manually produced EPIC 163 Continuous Oblique Imagery against the AI-generated v0.14 deliverable.

Designed for the DH1 project layout, but paths are discovered from a project root.
The comparison deliberately reuses existing_163_benchmark.csv for manual-final -> raw-image mapping,
so it does not repeat the expensive full image-fingerprint benchmark.

Outputs
-------
<output>/benchmark_summary.json
<output>/benchmark_summary.csv
<output>/image_level_comparison.csv
<output>/span_level_comparison.csv
<output>/manual_only_spans.csv
<output>/ai_only_spans.csv
<output>/source_data_or_sequence_exclusions.csv
<output>/structure_detector_summary.csv
<output>/report.html
<output>/review/<CATEGORY>/*.jpg   (optional contact sheets)

Example
-------
python compare_manual_vs_ai_v01.py "D:\\Sev\\Obliques\\DH1"

Optional
--------
python compare_manual_vs_ai_v01.py "D:\\Sev\\Obliques\\DH1" --review-limit 150
python compare_manual_vs_ai_v01.py "D:\\Sev\\Obliques\\DH1" --no-review
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

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


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
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


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    if df is None or df.empty and len(df.columns) == 0:
        if required:
            raise KeyError(f"None of required columns found: {candidates}")
        return None
    exact = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in exact:
            return exact[c.lower()]
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): str(c) for c in df.columns}
    for c in candidates:
        key = re.sub(r"[^a-z0-9]+", "", c.lower())
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"None of required columns found: {candidates}. Available: {list(df.columns)}")
    return None


def _safe_str(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _boolish(v) -> bool | None:
    if v is None or pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def parse_final_name(name: str) -> dict:
    m = FINAL_163_RE.match(Path(name).name)
    if not m:
        return {
            "filename": Path(name).name,
            "parse_status": "UNPARSED",
            "circuit": None,
            "span_id": None,
            "span_from": None,
            "span_to": None,
            "output_id": None,
            "capture_date_code": None,
            "version": None,
        }
    d = m.groupdict()
    return {
        "filename": Path(name).name,
        "parse_status": "OK",
        "circuit": d["circuit"],
        "span_id": f"{int(d['span_from'])}-{int(d['span_to'])}",
        "span_from": int(d["span_from"]),
        "span_to": int(d["span_to"]),
        "output_id": int(d["output_id"]),
        "capture_date_code": d["date"],
        "version": d["version"],
    }


def index_final_folder(folder: Path) -> pd.DataFrame:
    if not folder.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            r = parse_final_name(p.name)
            r["path"] = str(p)
            rows.append(r)
    return pd.DataFrame(rows)


def _span_tuple(span: str | None) -> tuple[int, int] | None:
    s = _safe_str(span)
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def span_relation(manual_span: str | None, ai_span: str | None) -> str:
    a = _span_tuple(manual_span)
    b = _span_tuple(ai_span)
    if not a or not b:
        return "NOT_COMPARABLE"
    if a == b:
        return "EXACT"
    d0, d1 = b[0] - a[0], b[1] - a[1]
    if d0 == d1 == 1:
        return "OFF_BY_ONE_FORWARD"
    if d0 == d1 == -1:
        return "OFF_BY_ONE_BACKWARD"
    if a[1] == b[0] or b[1] == a[0]:
        return "ADJACENT_SPAN"
    if set(a) & set(b):
        return "SHARED_STRUCTURE"
    return "DIFFERENT"


def discover_paths(root: Path, ai_run: str | None = None) -> dict:
    manual_dir = _first_existing([
        root / "163_continuous_oblique_imagery" / "v1",
        root / "163_continuous_oblique_imagery",
    ])
    if manual_dir is None:
        raise FileNotFoundError("Could not find manual 163_continuous_oblique_imagery under project root")

    if ai_run:
        ai_root = root / ai_run
    else:
        candidates = [p for p in root.glob("_continuous_oblique_*v014*") if p.is_dir()]
        if not candidates:
            candidates = [p for p in root.glob("_continuous_oblique_*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError("Could not find _continuous_oblique_* AI run folder")
        ai_root = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    deliverable = ai_root / "deliverable"
    ai_final_dir = _first_existing([
        deliverable / "163_continuous_oblique_imagery" / "v1",
        ai_root / "163_candidate" / "backward",
        ai_root / "163_candidate" / "forward",
    ])
    if ai_final_dir is None:
        raise FileNotFoundError(f"Could not find AI 163 images under {ai_root}")

    benchmark_csv = _first_existing([
        ai_root / "existing_163_benchmark.csv",
        deliverable / "qa_metadata" / "existing_163_benchmark.csv",
    ])
    if benchmark_csv is None:
        raise FileNotFoundError(
            f"existing_163_benchmark.csv not found under {ai_root}. "
            "This comparator intentionally reuses the existing manual-final -> raw benchmark."
        )

    manifest_csv = _first_existing([
        deliverable / "continuous_163_manifest.csv",
        ai_root / "product_manifest.csv",
        deliverable / "qa_metadata" / "product_manifest.csv",
    ])
    if manifest_csv is None:
        raise FileNotFoundError(f"AI product manifest not found under {ai_root}")

    image_index_csv = _first_existing([
        ai_root / "image_index.csv",
        deliverable / "qa_metadata" / "image_index.csv",
    ])

    crop_manifest_csv = _first_existing([
        deliverable / "structure_crops" / "structure_crops_manifest.csv",
        ai_root / "structure_crop_candidates.csv",
    ])
    structure_coverage_csv = _first_existing([
        deliverable / "structure_crops" / "structure_coverage.csv",
    ])

    raw_dir = _first_existing([
        root / "130_raw_overlapping_oblique_imagery" / "v1",
        root / "130_raw_overlapping_oblique_imagery",
    ])

    return {
        "root": root,
        "manual_dir": manual_dir,
        "ai_root": ai_root,
        "deliverable": deliverable,
        "ai_final_dir": ai_final_dir,
        "benchmark_csv": benchmark_csv,
        "manifest_csv": manifest_csv,
        "image_index_csv": image_index_csv,
        "crop_manifest_csv": crop_manifest_csv,
        "structure_coverage_csv": structure_coverage_csv,
        "raw_dir": raw_dir,
    }


def normalize_ai_manifest(df: pd.DataFrame, ai_final_index: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    raw_col = _find_col(df, ["filename", "source_filename", "raw_filename", "matched_raw_filename"], required=True)
    gen_col = _find_col(df, ["generated_filename", "delivery_filename", "output_filename", "final_filename"])
    span_col = _find_col(df, ["reference_id", "primary_span", "span_id", "span", "selected_span"])
    direction_col = _find_col(df, ["product_direction", "view_direction", "pass_direction"])
    run_col = _find_col(df, ["circuit_run_id", "run_id"])
    sequence_col = _find_col(df, ["generated_output_id", "sequence_in_direction", "sequence_in_run", "output_id"])
    rendered_col = _find_col(df, ["rendered_path", "delivery_path", "output_path", "path"])
    keep_col = _find_col(df, ["sequence_keep", "valid_for_163", "valid_for_product"])

    out = pd.DataFrame(index=df.index)
    out["ai_raw_filename"] = df[raw_col].map(lambda x: Path(str(x)).name if _safe_str(x) else None)
    out["ai_generated_filename"] = df[gen_col].map(lambda x: Path(str(x)).name if _safe_str(x) else None) if gen_col else None
    out["ai_span"] = df[span_col].map(_safe_str) if span_col else None
    out["ai_direction"] = df[direction_col].map(_safe_str) if direction_col else None
    out["ai_run"] = df[run_col].map(_safe_str) if run_col else None
    out["ai_sequence"] = df[sequence_col] if sequence_col else None
    out["ai_rendered_path"] = df[rendered_col].map(_safe_str) if rendered_col else None
    if keep_col:
        out["ai_keep"] = df[keep_col].map(_boolish)
    else:
        out["ai_keep"] = True

    # Prefer filename-parsed span if the manifest reference is unavailable or non-span.
    parsed_map = {}
    path_map = {}
    if not ai_final_index.empty:
        for _, r in ai_final_index.iterrows():
            parsed_map[str(r["filename"])] = r.get("span_id")
            path_map[str(r["filename"])] = r.get("path")

    for i in out.index:
        g = _safe_str(out.at[i, "ai_generated_filename"])
        if g:
            if not _span_tuple(out.at[i, "ai_span"]) and g in parsed_map:
                out.at[i, "ai_span"] = parsed_map[g]
            rp = _safe_str(out.at[i, "ai_rendered_path"])
            if (not rp or not Path(rp).exists()) and g in path_map:
                out.at[i, "ai_rendered_path"] = path_map[g]

    return out.reset_index(drop=True)


def normalize_benchmark(df: pd.DataFrame, manual_index: pd.DataFrame) -> pd.DataFrame:
    req = {
        "manual_filename": ["existing_163_filename", "manual_filename", "filename"],
        "manual_span": ["existing_span", "manual_span", "span_id"],
        "raw_filename": ["matched_raw_filename", "raw_filename", "source_filename"],
        "confidence": ["source_match_confidence", "match_confidence", "confidence"],
    }
    cols = {k: _find_col(df, v, required=True) for k, v in req.items()}
    optional = {
        "appearance_score": _find_col(df, ["appearance_score", "match_score"]),
        "appearance_margin": _find_col(df, ["appearance_margin", "match_margin"]),
        "generated_primary_span": _find_col(df, ["generated_primary_span", "primary_span"]),
        "benchmark_status": _find_col(df, ["benchmark_status"]),
        "generated_valid_for_product": _find_col(df, ["generated_valid_for_product", "valid_for_product"]),
        "product_status": _find_col(df, ["product_status"]),
        "generated_pass_id": _find_col(df, ["generated_pass_id", "pass_id"]),
        "generated_circuit_run_id": _find_col(df, ["generated_circuit_run_id", "circuit_run_id"]),
        "generated_view_direction": _find_col(df, ["generated_view_direction", "view_direction"]),
        "capture_date": _find_col(df, ["capture_date"]),
    }

    out = pd.DataFrame(index=df.index)
    out["manual_filename"] = df[cols["manual_filename"]].map(lambda x: Path(str(x)).name if _safe_str(x) else None)
    out["manual_span"] = df[cols["manual_span"]].map(_safe_str)
    out["matched_raw_filename"] = df[cols["raw_filename"]].map(lambda x: Path(str(x)).name if _safe_str(x) else None)
    out["source_match_confidence"] = df[cols["confidence"]].astype(str).str.upper().str.strip()
    for k, c in optional.items():
        out[k] = df[c] if c else None

    # Fill path/output/date metadata from actual manual folder where possible.
    mi = manual_index[[c for c in ["filename", "path", "output_id", "capture_date_code", "span_id"] if c in manual_index.columns]].copy()
    if not mi.empty:
        mi = mi.rename(columns={
            "filename": "manual_filename",
            "path": "manual_path",
            "output_id": "manual_output_id",
            "capture_date_code": "manual_capture_date_code",
            "span_id": "manual_span_from_filename",
        })
        out = out.merge(mi, on="manual_filename", how="left")
        out["manual_span"] = out["manual_span"].where(out["manual_span"].notna(), out["manual_span_from_filename"])
    else:
        out["manual_path"] = None
        out["manual_output_id"] = None
        out["manual_capture_date_code"] = None
    return out


def enrich_image_index(image_index: pd.DataFrame) -> pd.DataFrame:
    if image_index.empty:
        return image_index
    fn = _find_col(image_index, ["filename", "raw_filename"], required=True)
    wanted = {
        "idx_product_status": ["product_status"],
        "idx_rejection_reason": ["rejection_reason"],
        "idx_pass_direction": ["pass_direction"],
        "idx_view_direction": ["view_direction"],
        "idx_valid_for_product": ["valid_for_product"],
        "idx_sequence_keep": ["sequence_keep", "valid_for_163"],
        "idx_coverage_status": ["coverage_status"],
        "idx_primary_span": ["primary_span"],
        "idx_eo_source": ["eo_source"],
        "idx_circuit_run_id": ["circuit_run_id"],
        "idx_reference_confidence": ["reference_confidence"],
    }
    out = pd.DataFrame({"matched_raw_filename": image_index[fn].map(lambda x: Path(str(x)).name if _safe_str(x) else None)})
    for outname, candidates in wanted.items():
        c = _find_col(image_index, candidates)
        out[outname] = image_index[c] if c else None
    return out.drop_duplicates("matched_raw_filename")


def build_comparison(
    bench: pd.DataFrame,
    ai: pd.DataFrame,
    image_idx: pd.DataFrame,
    raw_dir: Path | None,
) -> pd.DataFrame:
    ai_by_raw: dict[str, pd.DataFrame] = {k: g.copy() for k, g in ai.groupby("ai_raw_filename", dropna=True)}
    ai_by_span: dict[str, pd.DataFrame] = {k: g.copy() for k, g in ai.groupby("ai_span", dropna=True)}
    idx_lookup = {}
    if not image_idx.empty:
        idx_lookup = image_idx.set_index("matched_raw_filename").to_dict("index")

    rows = []
    for _, r in bench.iterrows():
        manual_span = _safe_str(r.get("manual_span"))
        raw_name = _safe_str(r.get("matched_raw_filename"))
        confidence = str(r.get("source_match_confidence") or "").upper()
        comparable = confidence in {"HIGH", "MEDIUM"} and bool(raw_name)
        selected = ai_by_raw.get(raw_name) if raw_name else None
        same_span = ai_by_span.get(manual_span) if manual_span else None
        idx = idx_lookup.get(raw_name, {}) if raw_name else {}

        ai_selected = selected is not None and not selected.empty
        ai_spans = []
        ai_names = []
        ai_paths = []
        ai_dirs = []
        if ai_selected:
            ai_spans = sorted({_safe_str(x) for x in selected["ai_span"] if _safe_str(x)})
            ai_names = [x for x in selected["ai_generated_filename"].map(_safe_str).tolist() if x]
            ai_paths = [x for x in selected["ai_rendered_path"].map(_safe_str).tolist() if x]
            ai_dirs = sorted({_safe_str(x) for x in selected["ai_direction"] if _safe_str(x)})

        chosen_ai_span = None
        relation = "NOT_COMPARABLE"
        if ai_spans:
            # Prefer exact, then off-by-one/adjacent, then first deterministic value.
            ranked = []
            rank_order = {"EXACT": 0, "OFF_BY_ONE_FORWARD": 1, "OFF_BY_ONE_BACKWARD": 1,
                          "ADJACENT_SPAN": 2, "SHARED_STRUCTURE": 3, "DIFFERENT": 4, "NOT_COMPARABLE": 5}
            for s in ai_spans:
                rel = span_relation(manual_span, s)
                ranked.append((rank_order.get(rel, 99), s, rel))
            ranked.sort(key=lambda x: (x[0], x[1]))
            _, chosen_ai_span, relation = ranked[0]

        same_span_any = same_span is not None and not same_span.empty
        if not comparable:
            selection_status = "UNRESOLVED_SOURCE_MATCH"
        elif ai_selected and relation == "EXACT":
            selection_status = "SAME_RAW_SAME_SPAN"
        elif ai_selected:
            selection_status = "SAME_RAW_DIFFERENT_SPAN"
        elif same_span_any:
            selection_status = "DIFFERENT_RAW_SAME_SPAN"
        else:
            selection_status = "MANUAL_RAW_NOT_SELECTED"

        if ai_selected:
            ai_compare_path = next((p for p in ai_paths if p and Path(p).exists()), ai_paths[0] if ai_paths else None)
            ai_compare_name = ai_names[0] if ai_names else None
        elif same_span_any:
            candidate = same_span.sort_values("ai_sequence", na_position="last").iloc[0]
            ai_compare_path = _safe_str(candidate.get("ai_rendered_path"))
            ai_compare_name = _safe_str(candidate.get("ai_generated_filename"))
            chosen_ai_span = _safe_str(candidate.get("ai_span"))
            relation = span_relation(manual_span, chosen_ai_span)
        else:
            ai_compare_path = None
            ai_compare_name = None

        raw_path = None
        if raw_dir and raw_name:
            p = raw_dir / raw_name
            if p.exists():
                raw_path = str(p)

        rows.append({
            "manual_filename": r.get("manual_filename"),
            "manual_path": r.get("manual_path"),
            "manual_span": manual_span,
            "manual_output_id": r.get("manual_output_id"),
            "manual_capture_date_code": r.get("manual_capture_date_code"),
            "matched_raw_filename": raw_name,
            "matched_raw_path": raw_path,
            "source_match_confidence": confidence,
            "appearance_score": r.get("appearance_score"),
            "appearance_margin": r.get("appearance_margin"),
            "comparable_source_match": comparable,
            "ai_selected_same_raw": ai_selected,
            "ai_selected_same_span_any_raw": same_span_any,
            "ai_span_for_same_raw": ";".join(ai_spans) if ai_spans else None,
            "ai_generated_filename_for_same_raw": ";".join(ai_names) if ai_names else None,
            "ai_direction_for_same_raw": ";".join(ai_dirs) if ai_dirs else None,
            "comparison_ai_filename": ai_compare_name,
            "comparison_ai_path": ai_compare_path,
            "comparison_ai_span": chosen_ai_span,
            "span_relation": relation,
            "selection_status": selection_status,
            "v014_benchmark_generated_primary_span": r.get("generated_primary_span"),
            "v014_benchmark_status": r.get("benchmark_status"),
            "image_index_primary_span": idx.get("idx_primary_span"),
            "image_index_product_status": idx.get("idx_product_status"),
            "image_index_rejection_reason": idx.get("idx_rejection_reason"),
            "image_index_pass_direction": idx.get("idx_pass_direction"),
            "image_index_view_direction": idx.get("idx_view_direction"),
            "image_index_valid_for_product": idx.get("idx_valid_for_product"),
            "image_index_sequence_keep": idx.get("idx_sequence_keep"),
            "image_index_coverage_status": idx.get("idx_coverage_status"),
            "image_index_eo_source": idx.get("idx_eo_source"),
            "image_index_circuit_run_id": idx.get("idx_circuit_run_id"),
            "image_index_reference_confidence": idx.get("idx_reference_confidence"),
        })
    return pd.DataFrame(rows)


def build_span_level(manual_index: pd.DataFrame, ai_index: pd.DataFrame) -> pd.DataFrame:
    m = manual_index[manual_index["parse_status"].eq("OK")].groupby("span_id").size().rename("manual_image_count")
    a = ai_index[ai_index["parse_status"].eq("OK")].groupby("span_id").size().rename("ai_image_count")
    spans = sorted(set(m.index) | set(a.index), key=lambda s: _span_tuple(s) or (999999, 999999))
    rows = []
    for s in spans:
        mc = int(m.get(s, 0))
        ac = int(a.get(s, 0))
        if mc and ac:
            state = "BOTH"
        elif mc:
            state = "MANUAL_ONLY"
        else:
            state = "AI_ONLY"
        rows.append({
            "span_id": s,
            "manual_image_count": mc,
            "ai_image_count": ac,
            "count_delta_ai_minus_manual": ac - mc,
            "span_presence_state": state,
        })
    return pd.DataFrame(rows)


def detector_summary(crops: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not crops.empty:
        sid = _find_col(crops, ["structure_id", "structure", "structure_name"])
        sidx = _find_col(crops, ["structure_idx", "reference_idx"])
        status = _find_col(crops, ["crop_render_status", "tower_detector_status", "tower_assignment_status"])
        bucket = _find_col(crops, ["crop_bucket", "direction", "product_direction"])
        source = _find_col(crops, ["filename", "source_filename", "raw_filename"])
        rows.append({"metric": "crop_manifest_rows", "value": int(len(crops))})
        if sid:
            rows.append({"metric": "unique_structure_ids", "value": int(crops[sid].nunique(dropna=True))})
        elif sidx:
            rows.append({"metric": "unique_structure_ids", "value": int(crops[sidx].nunique(dropna=True))})
        if source:
            rows.append({"metric": "unique_source_images_in_crop_manifest", "value": int(crops[source].nunique(dropna=True))})
        if status:
            for k, v in crops[status].fillna("<NA>").astype(str).value_counts().items():
                rows.append({"metric": f"status:{k}", "value": int(v)})
        if bucket:
            for k, v in crops[bucket].fillna("<NA>").astype(str).value_counts().items():
                rows.append({"metric": f"bucket:{k}", "value": int(v)})
    if not coverage.empty:
        rows.append({"metric": "structure_coverage_rows", "value": int(len(coverage))})
        for col in coverage.columns:
            lc = str(col).lower()
            if "crop" in lc and pd.api.types.is_numeric_dtype(coverage[col]):
                try:
                    rows.append({"metric": f"coverage_sum:{col}", "value": float(pd.to_numeric(coverage[col], errors="coerce").sum())})
                except Exception:
                    pass
    if not rows:
        rows.append({"metric": "detector_metadata", "value": "NOT_AVAILABLE"})
    return pd.DataFrame(rows)


def classify_exclusion(row: pd.Series) -> str:
    if not bool(row.get("comparable_source_match")):
        return "UNRESOLVED_SOURCE_MATCH"
    if bool(row.get("ai_selected_same_raw")):
        return "SELECTED"
    reason = _safe_str(row.get("image_index_rejection_reason"))
    pdirection = _safe_str(row.get("image_index_pass_direction"))
    pstatus = _safe_str(row.get("image_index_product_status"))
    coverage = _safe_str(row.get("image_index_coverage_status"))
    valid = _boolish(row.get("image_index_valid_for_product"))
    keep = _boolish(row.get("image_index_sequence_keep"))
    if reason:
        return f"REJECTED:{reason}"
    if pdirection == "CROSSING_PASS":
        return "CROSSING_PASS"
    if valid is False:
        return f"NOT_VALID_FOR_PRODUCT:{pstatus or 'UNSPECIFIED'}"
    if keep is False:
        return "VALID_BUT_NOT_SEQUENCE_KEEP"
    if coverage and "MISSING" in coverage.upper():
        return f"COVERAGE:{coverage}"
    return "RAW_NOT_SELECTED_REASON_UNRESOLVED"


def _thumbnail(path: Path, max_w: int, max_h: int):
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


def render_review_sheet(row: pd.Series, dest: Path, quality: int = 88) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return "PILLOW_MISSING"

    manual_path = Path(str(row.get("manual_path"))) if _safe_str(row.get("manual_path")) else None
    ai_path = Path(str(row.get("comparison_ai_path"))) if _safe_str(row.get("comparison_ai_path")) else None
    raw_path = Path(str(row.get("matched_raw_path"))) if _safe_str(row.get("matched_raw_path")) else None

    panels = []
    labels = []
    if manual_path and manual_path.exists():
        panels.append(_thumbnail(manual_path, 850, 650)); labels.append("MANUAL 163")
    if ai_path and ai_path.exists():
        panels.append(_thumbnail(ai_path, 850, 650)); labels.append("AI 163")
    elif raw_path and raw_path.exists():
        panels.append(_thumbnail(raw_path, 850, 650)); labels.append("MATCHED RAW SOURCE")

    if not panels:
        return "NO_IMAGES_FOUND"

    margin = 22
    header_h = 150
    label_h = 34
    width = sum(im.width for im in panels) + margin * (len(panels) + 1)
    height = header_h + label_h + max(im.height for im in panels) + margin * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    title = f"{row.get('selection_status')} | span relation: {row.get('span_relation')}"
    draw.text((margin, 12), title, fill="black", font=font)
    lines = [
        f"Manual: {row.get('manual_filename')} | span {row.get('manual_span')}",
        f"AI: {row.get('comparison_ai_filename') or '<none>'} | span {row.get('comparison_ai_span') or '<none>'}",
        f"Raw: {row.get('matched_raw_filename') or '<unresolved>'} | source confidence {row.get('source_match_confidence')}",
        f"Exclusion/sequence: {row.get('exclusion_class')}",
    ]
    y = 48
    for line in lines:
        draw.text((margin, y), line, fill="black", font=small)
        y += 24

    x = margin
    panel_y = header_h + label_h
    for label, im in zip(labels, panels):
        draw.text((x, header_h), label, fill="black", font=small)
        canvas.paste(im, (x, panel_y))
        x += im.width + margin

    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=quality, optimize=True)
    return "RENDERED"


def render_reviews(comp: pd.DataFrame, out_dir: Path, limit: int, jpeg_quality: int) -> pd.DataFrame:
    if limit == 0 or comp.empty:
        return pd.DataFrame()
    todo = comp[~comp["selection_status"].eq("SAME_RAW_SAME_SPAN")].copy()
    # Most actionable disagreements first.
    order = {
        "SAME_RAW_DIFFERENT_SPAN": 0,
        "DIFFERENT_RAW_SAME_SPAN": 1,
        "MANUAL_RAW_NOT_SELECTED": 2,
        "UNRESOLVED_SOURCE_MATCH": 3,
    }
    todo["_rank"] = todo["selection_status"].map(order).fillna(9)
    todo = todo.sort_values(["_rank", "manual_output_id"], na_position="last").head(max(0, int(limit)))
    rows = []
    for _, r in todo.iterrows():
        cat = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(r.get("selection_status") or "OTHER"))
        stem = Path(str(r.get("manual_filename") or "manual")).stem
        dest = out_dir / "review" / cat / f"{stem}__{cat}.jpg"
        status = render_review_sheet(r, dest, jpeg_quality)
        rows.append({
            "manual_filename": r.get("manual_filename"),
            "selection_status": r.get("selection_status"),
            "span_relation": r.get("span_relation"),
            "review_status": status,
            "review_path": str(dest) if status == "RENDERED" else None,
        })
    return pd.DataFrame(rows)


def pct(n: int | float, d: int | float) -> float | None:
    if not d:
        return None
    return 100.0 * float(n) / float(d)


def build_summary(
    manual_index: pd.DataFrame,
    ai_index: pd.DataFrame,
    comp: pd.DataFrame,
    span_level: pd.DataFrame,
    detector: pd.DataFrame,
    paths: dict,
) -> dict:
    comparable = comp[comp["comparable_source_match"].eq(True)]
    same_raw = comparable[comparable["ai_selected_same_raw"].eq(True)]
    exact = same_raw[same_raw["span_relation"].eq("EXACT")]
    status_counts = {str(k): int(v) for k, v in comp["selection_status"].value_counts().items()}
    relation_counts = {str(k): int(v) for k, v in comp["span_relation"].value_counts().items()}
    exclusion_counts = {str(k): int(v) for k, v in comp["exclusion_class"].value_counts().items()}

    manual_spans = int((span_level["manual_image_count"] > 0).sum())
    ai_spans = int((span_level["ai_image_count"] > 0).sum())
    common_spans = int(span_level["span_presence_state"].eq("BOTH").sum())
    manual_only = int(span_level["span_presence_state"].eq("MANUAL_ONLY").sum())
    ai_only = int(span_level["span_presence_state"].eq("AI_ONLY").sum())

    return {
        "project_root": str(paths["root"]),
        "ai_run": str(paths["ai_root"]),
        "inputs": {k: str(v) if isinstance(v, Path) else v for k, v in paths.items() if k != "root"},
        "manual": {
            "images": int(len(manual_index)),
            "unique_spans": manual_spans,
        },
        "ai": {
            "images": int(len(ai_index)),
            "unique_spans": ai_spans,
        },
        "source_mapping": {
            "benchmark_rows": int(len(comp)),
            "comparable_high_medium": int(len(comparable)),
            "low_or_unresolved": int(len(comp) - len(comparable)),
            "ai_selected_same_raw": int(same_raw.shape[0]),
            "manual_source_selection_rate_percent": pct(len(same_raw), len(comparable)),
        },
        "attribution_on_same_raw": {
            "same_raw_comparable": int(len(same_raw)),
            "exact_span": int(len(exact)),
            "exact_span_percent": pct(len(exact), len(same_raw)),
            "span_relation_counts": relation_counts,
        },
        "product_selection": {
            "selection_status_counts": status_counts,
        },
        "span_presence": {
            "common_spans": common_spans,
            "manual_only_spans": manual_only,
            "ai_only_spans": ai_only,
            "manual_span_coverage_percent": pct(common_spans, manual_spans),
            "ai_spans_also_present_in_manual_percent": pct(common_spans, ai_spans),
        },
        "exclusions": exclusion_counts,
        "detector_metadata": {str(r["metric"]): r["value"] for _, r in detector.iterrows()},
        "notes": [
            "Manual-final to raw-image mapping is reused from existing_163_benchmark.csv.",
            "HIGH and MEDIUM source matches are treated as comparable. LOW matches are not scored as selection errors.",
            "Structure-crop detector metadata is summarized only; there is no manual structure-crop ground truth in the standard 163 folder, so detector accuracy is not inferred from crop counts.",
            "AI output IDs are not compared to manual output IDs because the two products use different sequence numbering.",
        ],
    }


def write_html_report(summary: dict, comp: pd.DataFrame, span_level: pd.DataFrame, review: pd.DataFrame, path: Path):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    cards = [
        ("Manual images", summary["manual"]["images"]),
        ("AI images", summary["ai"]["images"]),
        ("Comparable manual images", summary["source_mapping"]["comparable_high_medium"]),
        ("Same raw selected by AI", summary["source_mapping"]["ai_selected_same_raw"]),
        ("Exact span on same raw", summary["attribution_on_same_raw"]["exact_span"]),
        ("Common spans", summary["span_presence"]["common_spans"]),
        ("Manual-only spans", summary["span_presence"]["manual_only_spans"]),
        ("AI-only spans", summary["span_presence"]["ai_only_spans"]),
    ]

    status_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["product_selection"]["selection_status_counts"].items()
    )
    relation_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["attribution_on_same_raw"]["span_relation_counts"].items()
    )
    exclusion_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in summary["exclusions"].items()
    )
    card_html = "".join(f"<div class='card'><b>{html.escape(k)}</b><span>{fmt(v)}</span></div>" for k, v in cards)

    worst = comp[~comp["selection_status"].eq("SAME_RAW_SAME_SPAN")].head(80)
    cols = ["manual_filename", "manual_span", "matched_raw_filename", "source_match_confidence",
            "comparison_ai_filename", "comparison_ai_span", "selection_status", "span_relation", "exclusion_class"]
    table_rows = []
    for _, r in worst.iterrows():
        tds = "".join(f"<td>{html.escape(str(r.get(c) if pd.notna(r.get(c)) else ''))}</td>" for c in cols)
        table_rows.append(f"<tr>{tds}</tr>")

    notes = "".join(f"<li>{html.escape(n)}</li>" for n in summary["notes"])
    doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>DH1 Manual vs AI 163 benchmark</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#222}}h1,h2{{margin-bottom:8px}}
.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}.card{{border:1px solid #bbb;border-radius:8px;padding:12px 16px;min-width:170px;background:#fafafa}}
.card b{{display:block;font-size:13px;color:#555}}.card span{{display:block;font-size:24px;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 24px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#eee;position:sticky;top:0}}
.metric{{font-size:18px}}code{{background:#f3f3f3;padding:2px 4px}}
</style></head><body>
<h1>Manual vs AI Continuous Oblique 163 benchmark</h1>
<p>Project: <code>{html.escape(summary['project_root'])}</code></p>
<div class='cards'>{card_html}</div>
<p class='metric'>Manual source selection rate: <b>{fmt(summary['source_mapping']['manual_source_selection_rate_percent'])}%</b><br>
Exact span attribution when the same raw image is selected: <b>{fmt(summary['attribution_on_same_raw']['exact_span_percent'])}%</b><br>
Manual span presence covered by AI: <b>{fmt(summary['span_presence']['manual_span_coverage_percent'])}%</b></p>
<h2>Product selection status</h2><table><tr><th>Status</th><th>Count</th></tr>{status_rows}</table>
<h2>Span relation on comparison rows</h2><table><tr><th>Relation</th><th>Count</th></tr>{relation_rows}</table>
<h2>Why comparable manual sources were not selected</h2><table><tr><th>Class</th><th>Count</th></tr>{exclusion_rows}</table>
<h2>First disagreement rows</h2><table><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in cols)}</tr>{''.join(table_rows)}</table>
<h2>Interpretation notes</h2><ul>{notes}</ul>
<p>Full detail is in <code>image_level_comparison.csv</code> and <code>span_level_comparison.csv</code>. Review contact sheets are under <code>review/</code>.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare manual 163 imagery against AI v0.14 output.")
    parser.add_argument("project_root", help="Project root, e.g. D:\\Sev\\Obliques\\DH1")
    parser.add_argument("--ai-run", default=None, help="AI run folder name under project root. Default: auto-discover latest.")
    parser.add_argument("--output", default=None, help="Output folder. Default: <project_root>\\_benchmark_manual_vs_ai_v01")
    parser.add_argument("--review-limit", type=int, default=150, help="Maximum disagreement contact sheets. Default: 150")
    parser.add_argument("--no-review", action="store_true", help="Do not render disagreement contact sheets.")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="Review JPEG quality. Default: 88")
    args = parser.parse_args(argv)

    root = Path(args.project_root).expanduser().resolve()
    paths = discover_paths(root, args.ai_run)
    out_dir = Path(args.output).expanduser().resolve() if args.output else root / "_benchmark_manual_vs_ai_v01"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Indexing manual and AI final 163 imagery...")
    manual_index = index_final_folder(paths["manual_dir"])
    ai_final_index = index_final_folder(paths["ai_final_dir"])
    if manual_index.empty:
        raise SystemExit("Manual 163 folder contains no JPGs")
    if ai_final_index.empty:
        raise SystemExit("AI 163 deliverable contains no JPGs")

    print("[2/8] Loading existing manual-final -> raw benchmark and AI manifest...")
    bench_raw = _read_csv(paths["benchmark_csv"], required=True)
    manifest_raw = _read_csv(paths["manifest_csv"], required=True)
    image_idx_raw = _read_csv(paths["image_index_csv"], required=False)
    crops_raw = _read_csv(paths["crop_manifest_csv"], required=False)
    coverage_raw = _read_csv(paths["structure_coverage_csv"], required=False)

    bench = normalize_benchmark(bench_raw, manual_index)
    ai = normalize_ai_manifest(manifest_raw, ai_final_index)
    image_idx = enrich_image_index(image_idx_raw)

    print("[3/8] Comparing manual source selection and span attribution...")
    comp = build_comparison(bench, ai, image_idx, paths["raw_dir"])
    comp["exclusion_class"] = comp.apply(classify_exclusion, axis=1)

    print("[4/8] Comparing aggregate span presence/counts...")
    span_level = build_span_level(manual_index, ai_final_index)
    manual_only = span_level[span_level["span_presence_state"].eq("MANUAL_ONLY")].copy()
    ai_only = span_level[span_level["span_presence_state"].eq("AI_ONLY")].copy()

    print("[5/8] Summarizing structure detector package metadata...")
    detector = detector_summary(crops_raw, coverage_raw)

    print("[6/8] Writing CSV and JSON outputs...")
    comp.to_csv(out_dir / "image_level_comparison.csv", index=False)
    span_level.to_csv(out_dir / "span_level_comparison.csv", index=False)
    manual_only.to_csv(out_dir / "manual_only_spans.csv", index=False)
    ai_only.to_csv(out_dir / "ai_only_spans.csv", index=False)
    comp[~comp["exclusion_class"].eq("SELECTED")].to_csv(out_dir / "source_data_or_sequence_exclusions.csv", index=False)
    detector.to_csv(out_dir / "structure_detector_summary.csv", index=False)
    manual_index.to_csv(out_dir / "manual_163_index.csv", index=False)
    ai_final_index.to_csv(out_dir / "ai_163_index.csv", index=False)

    summary = build_summary(manual_index, ai_final_index, comp, span_level, detector, paths)
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    flat_summary = []
    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}.{k}" if prefix else str(k), v)
        elif not isinstance(obj, (list, tuple)):
            flat_summary.append({"metric": prefix, "value": obj})
    flatten("", summary)
    pd.DataFrame(flat_summary).to_csv(out_dir / "benchmark_summary.csv", index=False)

    print("[7/8] Rendering disagreement contact sheets...")
    review = pd.DataFrame()
    if not args.no_review and args.review_limit != 0:
        review = render_reviews(comp, out_dir, args.review_limit, args.jpeg_quality)
        review.to_csv(out_dir / "review_manifest.csv", index=False)
    else:
        pd.DataFrame(columns=["manual_filename", "selection_status", "span_relation", "review_status", "review_path"]).to_csv(
            out_dir / "review_manifest.csv", index=False
        )

    print("[8/8] Writing HTML report...")
    write_html_report(summary, comp, span_level, review, out_dir / "report.html")

    print()
    print("Manual vs AI benchmark complete")
    print(f"  Manual images:                    {summary['manual']['images']}")
    print(f"  AI images:                        {summary['ai']['images']}")
    print(f"  Manual unique spans:              {summary['manual']['unique_spans']}")
    print(f"  AI unique spans:                  {summary['ai']['unique_spans']}")
    print(f"  Comparable manual source matches: {summary['source_mapping']['comparable_high_medium']}")
    print(f"  Same raw selected by AI:          {summary['source_mapping']['ai_selected_same_raw']}")
    msr = summary['source_mapping']['manual_source_selection_rate_percent']
    print(f"  Manual source selection rate:     {msr:.2f}%" if msr is not None else "  Manual source selection rate:     n/a")
    print(f"  Exact span on same raw:           {summary['attribution_on_same_raw']['exact_span']}/{summary['attribution_on_same_raw']['same_raw_comparable']}")
    esp = summary['attribution_on_same_raw']['exact_span_percent']
    print(f"  Exact span attribution:           {esp:.2f}%" if esp is not None else "  Exact span attribution:           n/a")
    print(f"  Common spans:                     {summary['span_presence']['common_spans']}")
    print(f"  Manual-only spans:                {summary['span_presence']['manual_only_spans']}")
    print(f"  AI-only spans:                    {summary['span_presence']['ai_only_spans']}")
    print(f"  Output:                           {out_dir}")
    print(f"  Report:                           {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
