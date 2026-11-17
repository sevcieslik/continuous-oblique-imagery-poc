#!/usr/bin/env python
r"""
Evaluate structure detection from a continuous-oblique run, with optional human adjudication.

Step 1 - automatic metrics + adjudication pack:
    python evaluate_structure_detector_v01.py <run_output>
  writes <run_output>/_detector_eval/
    detector_eval_summary.json     status/confidence/registration counts, crop yield, residual stats
    detector_eval_by_pass.csv      per-pass yield and residuals
    adjudication_template.csv      one row per attempted (structure, image) pair to label
    adjudication.html              QA overlays + crops side by side for fast labelling

Step 2 - fill `label` in adjudication_template.csv (save as adjudication.csv) and rerun:
    python evaluate_structure_detector_v01.py <run_output> --adjudication adjudication.csv
  Labels for rows WITH a crop:     CORRECT | WRONG_STRUCTURE | NOT_A_TOWER | PARTIAL
  Labels for rows WITHOUT a crop:  MISSED  (target tower clearly visible) | NOT_VISIBLE | UNSURE
  Adds precision, wrong-structure rate and recall estimate to the summary, split by confidence.

The script never changes run outputs outside _detector_eval.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SKIP_STATUSES = {"NOT_IN_RENDER_SUBSET", "NOT_RENDERED_OTHER_VIEW", "NOT_RUN", "SOURCE_MISSING"}
CROP_LABELS = {"CORRECT", "WRONG_STRUCTURE", "NOT_A_TOWER", "PARTIAL"}
MISS_LABELS = {"MISSED", "NOT_VISIBLE", "UNSURE"}


def _parts(p) -> list[str]:
    return [x for x in re.split(r"[\\/]+", str(p or "")) if x]


def _local(path, run: Path, anchor: str) -> Path | None:
    s = str(path or "")
    if not s or s.lower() == "nan":
        return None
    if Path(s).exists():
        return Path(s)
    parts = _parts(s)
    if anchor in parts:
        q = run.joinpath(*parts[parts.index(anchor):])
        if q.exists():
            return q
    return None


def pair_id(r) -> str:
    return f"{Path(_parts(r.get('filename'))[-1]).stem}__{r.get('structure_id')}"


def evaluate(run: Path, adjudication: Path | None, out: Path, html_limit: int) -> dict:
    crops = pd.read_csv(run / "structure_crop_candidates.csv", low_memory=False)
    status = crops.get("crop_render_status", pd.Series("NOT_RUN", index=crops.index)).fillna("NOT_RUN").astype(str)
    att = crops[~status.isin(SKIP_STATUSES)].copy()
    att["pair_id"] = att.apply(pair_id, axis=1)
    att["has_crop"] = att["crop_render_status"].astype(str).eq("RENDERED")
    conf = att.get("tower_assignment_confidence", pd.Series(index=att.index, dtype=object))
    fallback = pd.Series(np.where(att["has_crop"], "UNGRADED", "NONE"), index=att.index)
    conf = conf.astype(object).where(conf.notna() & conf.astype(str).str.strip().ne(""), fallback)
    att["confidence"] = conf.astype(str)
    out.mkdir(parents=True, exist_ok=True)

    def rstats(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return {"n": int(len(s)), "median": float(s.median()) if len(s) else None,
                "p90": float(s.quantile(0.9)) if len(s) else None}

    prior_res = att.get("prior_residual_px")
    if prior_res is None or pd.to_numeric(prior_res, errors="coerce").notna().sum() == 0:
        cx = (pd.to_numeric(att.get("tower_box_xmin"), errors="coerce") + pd.to_numeric(att.get("tower_box_xmax"), errors="coerce")) / 2
        cy = (pd.to_numeric(att.get("tower_box_ymin"), errors="coerce") + pd.to_numeric(att.get("tower_box_ymax"), errors="coerce")) / 2
        prior_res = np.hypot(cx - pd.to_numeric(att.get("geometry_prior_x_px"), errors="coerce"),
                             cy - pd.to_numeric(att.get("geometry_prior_y_px"), errors="coerce"))
    summary = {
        "run": str(run),
        "attempted_pairs": int(len(att)),
        "pairs_with_crop": int(att["has_crop"].sum()),
        "crop_yield_percent": round(100.0 * att["has_crop"].mean(), 2) if len(att) else None,
        "detector_status_counts": dict(Counter(att.get("tower_detector_status", pd.Series(dtype=object)).fillna("NONE").astype(str))),
        "assignment_confidence_counts": dict(Counter(att["confidence"].astype(str))),
        "registration_status_counts": dict(Counter(att.get("image_registration_status", pd.Series(dtype=object)).dropna().astype(str))),
        "prior_residual_px": rstats(prior_res),
        "georef_residual_px": rstats(att.get("georef_residual_px", pd.Series(dtype=float))),
        "images_attempted": int(att["filename"].nunique()),
        "structures_attempted": int(att["structure_id"].nunique()),
        "structures_with_crop": int(att.loc[att["has_crop"], "structure_id"].nunique()),
    }
    by_pass = att.groupby(att.get("pass_id", pd.Series("ALL", index=att.index)).astype(str)).agg(
        attempted=("pair_id", "size"), with_crop=("has_crop", "sum")).reset_index()
    by_pass["yield_percent"] = 100.0 * by_pass["with_crop"] / by_pass["attempted"].clip(lower=1)
    by_pass.to_csv(out / "detector_eval_by_pass.csv", index=False)

    tmpl_cols = ["pair_id", "filename", "structure_id", "reference_idx", "pass_id", "crop_bucket", "has_crop",
                 "confidence", "tower_detector_status", "tower_detector_final_score", "tower_ownership",
                 "image_registration_status", "crop_path", "crop_debug_path"]
    tmpl = att[[c for c in tmpl_cols if c in att.columns]].copy()
    tmpl["label"] = ""
    tmpl["note"] = ""
    tmpl.to_csv(out / "adjudication_template.csv", index=False)

    # HTML pack: reviews first, then medium/ungraded, then high.
    order = {"NONE": 0, "MEDIUM": 1, "UNGRADED": 2, "HIGH": 3}
    pack = att.assign(_o=att["confidence"].map(order).fillna(9)).sort_values(["_o", "filename"]).head(html_limit)
    rows_html = []
    for _, r in pack.iterrows():
        qa = _local(r.get("crop_debug_path"), run, "qa")
        cr = _local(r.get("crop_path"), run, "structure_crops")
        cell = lambda p: f"<img src='{html.escape(os.path.relpath(p, out))}' loading='lazy'>" if p else "<i>n/a</i>"
        rows_html.append(
            f"<tr><td><b>{html.escape(str(r['pair_id']))}</b><br>{html.escape(str(r.get('tower_detector_status')))}"
            f"<br>conf={html.escape(str(r['confidence']))}</td><td>{cell(qa)}</td><td>{cell(cr)}</td></tr>")
    (out / "adjudication.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Detector adjudication</title>"
        "<style>body{font-family:sans-serif}td{vertical-align:top;border-bottom:1px solid #ccc;padding:6px}"
        "img{max-width:700px;max-height:520px}</style>"
        "<p>Label each pair_id in adjudication_template.csv. Legend for overlays: white cross = corrected prior, "
        "grey = raw prior, red = competitor hypotheses, yellow = projected staking base, green = chosen box, "
        "magenta = plausible detections, cyan = ROI.</p><table>" + "".join(rows_html) + "</table>",
        encoding="utf-8")

    if adjudication is not None:
        adj = pd.read_csv(adjudication)
        adj["label"] = adj["label"].astype(str).str.strip().str.upper()
        m = att.merge(adj[["pair_id", "label"]], on="pair_id", how="inner")
        m = m[m["label"].isin(CROP_LABELS | MISS_LABELS)]
        res = {"labelled_pairs": int(len(m))}
        for name, g in [("all", m)] + [(f"confidence_{c}", m[m["confidence"].eq(c)]) for c in sorted(m["confidence"].unique())]:
            crop = g[g["has_crop"] & g["label"].isin(CROP_LABELS)]
            nocrop = g[~g["has_crop"] & g["label"].isin(MISS_LABELS)]
            correct = int(crop["label"].eq("CORRECT").sum())
            missed = int(nocrop["label"].eq("MISSED").sum())
            res[name] = {
                "labelled_crops": int(len(crop)),
                "precision": round(correct / len(crop), 4) if len(crop) else None,
                "wrong_structure_rate": round(crop["label"].eq("WRONG_STRUCTURE").mean(), 4) if len(crop) else None,
                "not_a_tower_rate": round(crop["label"].eq("NOT_A_TOWER").mean(), 4) if len(crop) else None,
                "labelled_no_crop": int(len(nocrop)),
                "missed_visible_towers": missed,
                "recall_estimate": round(correct / (correct + missed), 4) if (correct + missed) else None,
            }
        summary["adjudication"] = res
    (out / "detector_eval_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate structure detection and build an adjudication pack.")
    ap.add_argument("run_output", type=Path)
    ap.add_argument("--adjudication", type=Path, default=None, help="Filled adjudication CSV (pair_id,label)")
    ap.add_argument("--output", type=Path, default=None, help="Default: <run_output>/_detector_eval")
    ap.add_argument("--html-limit", type=int, default=400)
    args = ap.parse_args(argv)
    run = args.run_output.resolve()
    s = evaluate(run, args.adjudication, (args.output or run / "_detector_eval").resolve(), args.html_limit)
    print(json.dumps({k: v for k, v in s.items() if k != "run"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
