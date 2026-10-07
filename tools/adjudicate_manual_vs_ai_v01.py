#!/usr/bin/env python
"""
adjudicate_manual_vs_ai_v01.py

Focused adjudication of:
A) clean manual-vs-AI span disagreements from the audited manual 163 product;
B) manual images that the current geometry/pass model classifies as CROSSING/UNKNOWN/etc.

This script does NOT change either product. It builds review tables and contact sheets.

Expected project layout
-----------------------
<root>/
  163_continuous_oblique_imagery/v1/
  130_raw_overlapping_oblique_imagery/v1/
  _continuous_oblique_DH1_full_v014/
      image_index.csv
      flight_passes.csv
      reference_spans.csv
      existing_163_benchmark.csv
  _audit_manual_163_v03/
      manual_image_audit.csv
      geometry_run_transitions.csv

Outputs
-------
<root>/_adjudication_manual_vs_ai_v01/
  adjudication_summary.json
  adjudication_summary.csv
  span_disagreements.csv
  wrong_pass_cases.csv
  human_review_template.csv
  report.html
  review/span_disagreements/*.jpg
  review/wrong_pass/*.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def s(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def n(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def truthy(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def require(path: Path, label: str):
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def parse_span(span):
    m = re.fullmatch(r"\s*(\d+)-(\d+)\s*", s(span))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def span_relation(manual_span, ai_span):
    a = parse_span(manual_span)
    b = parse_span(ai_span)
    if not a or not b:
        return "NOT_COMPARABLE"
    if a == b:
        return "EXACT"
    if b[0] == a[0] - 1 and b[1] == a[1] - 1:
        return "OFF_BY_ONE_BACKWARD"
    if b[0] == a[0] + 1 and b[1] == a[1] + 1:
        return "OFF_BY_ONE_FORWARD"
    return "DIFFERENT"


def span_meta_map(spans: pd.DataFrame):
    out = {}
    for _, r in spans.iterrows():
        sid = s(r.get("span_id"))
        if not sid:
            continue
        a, b = n(r.get("chainage_from")), n(r.get("chainage_to"))
        if math.isfinite(a) and math.isfinite(b):
            lo, hi = min(a, b), max(a, b)
            out[sid] = {
                "chainage_from": lo,
                "chainage_to": hi,
                "centre": (lo + hi) / 2.0,
                "length": hi - lo,
                "from_structure": s(r.get("from_structure")),
                "to_structure": s(r.get("to_structure")),
            }
    return out


def point_span_metrics(ch, meta):
    if not math.isfinite(ch) or not meta:
        return {"inside": False, "distance_to_span": np.nan, "distance_to_centre": np.nan, "distance_to_boundary": np.nan}
    lo, hi, c = meta["chainage_from"], meta["chainage_to"], meta["centre"]
    inside = lo <= ch <= hi
    dist_span = 0.0 if inside else min(abs(ch - lo), abs(ch - hi))
    return {
        "inside": inside,
        "distance_to_span": dist_span,
        "distance_to_centre": abs(ch - c),
        "distance_to_boundary": min(abs(ch - lo), abs(ch - hi)),
    }


def geometry_adjudication(target_ch, manual_span, ai_span, span_map, assignment_status):
    mm = span_map.get(s(manual_span))
    am = span_map.get(s(ai_span))
    m = point_span_metrics(target_ch, mm)
    a = point_span_metrics(target_ch, am)

    if s(assignment_status).upper() == "BOUNDARY_AMBIGUOUS":
        cls = "AMBIGUOUS_BOUNDARY"
    elif not mm or not am or not math.isfinite(target_ch):
        cls = "INSUFFICIENT_GEOMETRY"
    elif m["inside"] and not a["inside"]:
        cls = "MANUAL_SUPPORTED"
    elif a["inside"] and not m["inside"]:
        cls = "AI_SUPPORTED"
    elif m["inside"] and a["inside"]:
        cls = "AMBIGUOUS_BOUNDARY"
    else:
        dm, da = m["distance_to_centre"], a["distance_to_centre"]
        scale = max(mm["length"], am["length"], 1.0)
        if abs(dm - da) <= 0.10 * scale:
            cls = "BOTH_PLAUSIBLE"
        elif dm < da:
            cls = "MANUAL_SUPPORTED_WEAK"
        else:
            cls = "AI_SUPPORTED_WEAK"

    return cls, m, a


def build_transition_lookup(transitions: pd.DataFrame):
    lookup = {}
    if transitions.empty:
        return lookup
    for _, r in transitions.iterrows():
        a, b = s(r.get("image_a")), s(r.get("image_b"))
        if not a or not b:
            continue
        rec = {
            "visual_status": s(r.get("visual_status")),
            "visual_score": n(r.get("visual_score")),
            "global_similarity": n(r.get("global_similarity")),
            "dense_best_mean": n(r.get("dense_best_mean")),
            "chainage_gap_abs": n(r.get("chainage_gap_abs")),
            "eo_order_delta": n(r.get("eo_order_delta")),
        }
        lookup[(a, b)] = rec
        lookup[(b, a)] = rec
    return lookup


def neighbor_rows(audit: pd.DataFrame, row, count=1):
    rid = s(row.get("idx_circuit_run_id"))
    eo = n(row.get("idx_eo_order"))
    if rid and math.isfinite(eo):
        g = audit[audit["idx_circuit_run_id"].astype(str).eq(rid)].copy()
        g["_eo"] = pd.to_numeric(g.get("idx_eo_order"), errors="coerce")
        g = g[g["_eo"].notna()].sort_values("_eo")
        idxs = list(g.index)
        try:
            pos = idxs.index(row.name)
        except ValueError:
            pos = None
        if pos is not None:
            prev = [g.loc[i] for i in idxs[max(0, pos-count):pos]]
            nxt = [g.loc[i] for i in idxs[pos+1:pos+1+count]]
            return prev, nxt

    # fallback: by manual span number then output_id
    g = audit.copy()
    g["_sf"] = g["span_id"].apply(lambda x: parse_span(x)[0] if parse_span(x) else np.nan)
    g["_oid"] = pd.to_numeric(g.get("output_id"), errors="coerce")
    g = g.sort_values(["_sf", "_oid"], na_position="last")
    idxs = list(g.index)
    try:
        pos = idxs.index(row.name)
    except ValueError:
        return [], []
    return [g.loc[i] for i in idxs[max(0, pos-count):pos]], [g.loc[i] for i in idxs[pos+1:pos+1+count]]


def image_map(directory: Path):
    return {p.name: p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def raw_image_map(directory: Path):
    return image_map(directory)


def load_thumb(path: Path, max_w=720, max_h=540):
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as im:
            try:
                im.draft("RGB", (max_w*2, max_h*2))
            except Exception:
                pass
            im = im.convert("RGB")
            im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            return im.copy()


def text_panel(lines, width=720, height=160):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    y = 8
    for line in lines:
        d.text((8, y), str(line), fill="black", font=font)
        y += 18
        if y > height - 18:
            break
    return im


def stack_panels(panels, footer_lines, dest: Path):
    from PIL import Image, ImageDraw, ImageFont
    panels = [p for p in panels if p is not None]
    if not panels:
        return "NO_IMAGES"
    gap = 8
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels) + gap*(len(panels)-1)
    footer_h = max(120, 18*len(footer_lines)+16)
    canvas = Image.new("RGB", (w, h + footer_h), "white")
    y = 0
    for p in panels:
        x = (w - p.width)//2
        canvas.paste(p, (x, y))
        y += p.height + gap
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    yy = h + 8
    for line in footer_lines:
        d.text((8, yy), str(line), fill="black", font=font)
        yy += 18
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "JPEG", quality=90)
    return "RENDERED"


def annotate_thumb(path: Path, title: str, details: list[str], max_w=720, max_h=500):
    from PIL import Image, ImageDraw, ImageFont
    im = load_thumb(path, max_w=max_w, max_h=max_h)
    footer_h = 70 + 18*len(details)
    out = Image.new("RGB", (max(im.width, 500), im.height + footer_h), "white")
    out.paste(im, ((out.width-im.width)//2, 0))
    d = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    y = im.height + 8
    d.text((8, y), title, fill="black", font=font)
    y += 22
    for line in details:
        d.text((8, y), str(line), fill="black", font=font)
        y += 18
    return out


def transition_for(lookup, a, b):
    return lookup.get((s(a), s(b)), {})


def safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def write_html(out_dir: Path, summary: dict, span_df: pd.DataFrame, pass_df: pd.DataFrame):
    def esc(x):
        import html
        return html.escape(s(x))

    span_cols = [
        "manual_filename","manual_span","ai_span","span_relation","geometry_adjudication",
        "target_chainage","ai_span_is_missing_from_manual","priority"
    ]
    pass_cols = [
        "manual_filename","manual_span","matched_raw_filename","model_pass_direction",
        "model_pass_id","alignment_progression_ratio","manual_images_from_same_pass",
        "dino_context","pass_review_class"
    ]

    def table(df, cols, limit=100):
        if df.empty:
            return "<p>No rows.</p>"
        cols = [c for c in cols if c in df.columns]
        head = "".join(f"<th>{esc(c)}</th>" for c in cols)
        rows = []
        for _, r in df.head(limit).iterrows():
            rows.append("<tr>" + "".join(f"<td>{esc(r.get(c))}</td>" for c in cols) + "</tr>")
        return f"<table><tr>{head}</tr>{''.join(rows)}</table>"

    cards = "".join(
        f"<div class='card'><b>{esc(k)}</b><span>{esc(v)}</span></div>"
        for k,v in [
            ("Clean span disagreements", summary.get("clean_span_disagreements")),
            ("Off-by-one backward", summary.get("off_by_one_backward")),
            ("Off-by-one forward", summary.get("off_by_one_forward")),
            ("AI span missing in manual", summary.get("ai_span_is_missing_manual")),
            ("Wrong-pass review cases", summary.get("wrong_pass_cases")),
            ("Classifier conflicts", summary.get("classifier_conflict_cases")),
        ]
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>DH1 adjudication</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#222}}
.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}
.card{{border:1px solid #bbb;border-radius:8px;padding:12px 16px;min-width:180px;background:#fafafa}}
.card b{{display:block;font-size:13px;color:#555}}.card span{{display:block;font-size:24px;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 24px}}
th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#eee}}
code{{background:#f3f3f3;padding:2px 4px}}
</style></head><body>
<h1>Manual vs AI adjudication</h1>
<p>This report does not auto-correct either product. It isolates the clean label disagreements and the pass-classifier conflicts for human adjudication.</p>
<div class='cards'>{cards}</div>
<h2>A. Clean span disagreements</h2>
{table(span_df, span_cols)}
<h2>B. Possible wrong-pass / classifier conflicts</h2>
{table(pass_df, pass_cols)}
<h2>Review outputs</h2>
<p>Contact sheets are under <code>review/span_disagreements</code> and <code>review/wrong_pass</code>.</p>
<p>Fill <code>human_review_template.csv</code> with one of: MANUAL_SUPPORTED, AI_SUPPORTED, AMBIGUOUS_BOUNDARY, BOTH_PLAUSIBLE, KEEP_MANUAL_FRAME, EXCLUDE_MANUAL_FRAME, CLASSIFIER_ERROR, NEEDS_MORE_EVIDENCE.</p>
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_root")
    ap.add_argument("--output", default=None)
    ap.add_argument("--render-limit", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.project_root)
    ai = root / "_continuous_oblique_DH1_full_v014"
    audit_dir = root / "_audit_manual_163_v03"
    manual_dir = root / "163_continuous_oblique_imagery" / "v1"
    raw_dir = root / "130_raw_overlapping_oblique_imagery" / "v1"
    out_dir = Path(args.output) if args.output else root / "_adjudication_manual_vs_ai_v01"

    for p, label in [
        (audit_dir / "manual_image_audit.csv", "manual audit"),
        (audit_dir / "geometry_run_transitions.csv", "manual geometry transitions"),
        (ai / "image_index.csv", "AI image index"),
        (ai / "reference_spans.csv", "reference spans"),
        (ai / "flight_passes.csv", "flight passes"),
        (manual_dir, "manual 163 directory"),
        (raw_dir, "raw imagery directory"),
    ]:
        require(p, label)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review" / "span_disagreements").mkdir(parents=True, exist_ok=True)
    (out_dir / "review" / "wrong_pass").mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading audit and geometry data...")
    audit = pd.read_csv(audit_dir / "manual_image_audit.csv")
    transitions = pd.read_csv(audit_dir / "geometry_run_transitions.csv")
    img = pd.read_csv(ai / "image_index.csv")
    spans = pd.read_csv(ai / "reference_spans.csv")
    passes = pd.read_csv(ai / "flight_passes.csv")

    for col in ["filename","span_id","matched_raw_filename","source_match_confidence",
                "calibration_grade","manual_audit_status","manual_audit_flags",
                "idx_primary_span","idx_circuit_run_id","idx_pass_id","idx_pass_direction"]:
        if col not in audit.columns:
            audit[col] = np.nan

    span_map = span_meta_map(spans)
    manual_span_set = set(audit["span_id"].dropna().astype(str))
    transition_lookup = build_transition_lookup(transitions)

    img_by_file = {}
    if "filename" in img.columns:
        for _, r in img.drop_duplicates("filename").iterrows():
            img_by_file[s(r.get("filename"))] = r.to_dict()

    pass_by_id = {}
    if "pass_id" in passes.columns:
        for _, r in passes.drop_duplicates("pass_id").iterrows():
            pass_by_id[s(r.get("pass_id"))] = r.to_dict()

    manual_paths = image_map(manual_dir)
    raw_paths = raw_image_map(raw_dir)

    print("[2/7] Building clean span-disagreement adjudication set...")
    grade_a = audit[audit["calibration_grade"].astype(str).eq("A")].copy()
    disagreements = grade_a[
        grade_a["span_id"].notna()
        & grade_a["idx_primary_span"].notna()
        & ~grade_a["span_id"].astype(str).eq(grade_a["idx_primary_span"].astype(str))
    ].copy()

    span_rows = []
    for idx, r in disagreements.iterrows():
        manual_span = s(r.get("span_id"))
        ai_span = s(r.get("idx_primary_span"))
        raw_name = s(r.get("matched_raw_filename"))
        ir = img_by_file.get(raw_name, {})

        target_ch = n(ir.get("optical_target_chainage"))
        if not math.isfinite(target_ch):
            target_ch = n(ir.get("reference_chainage"))
        if not math.isfinite(target_ch):
            target_ch = n(r.get("idx_reference_chainage"))

        assignment = s(ir.get("reference_assignment_status"))
        adj, mm, am = geometry_adjudication(target_ch, manual_span, ai_span, span_map, assignment)

        prevs, nexts = neighbor_rows(audit, r, count=1)
        prev = prevs[-1] if prevs else None
        nxt = nexts[0] if nexts else None
        prev_fn = s(prev.get("filename")) if prev is not None else ""
        next_fn = s(nxt.get("filename")) if nxt is not None else ""
        pm = transition_for(transition_lookup, prev_fn, s(r.get("filename"))) if prev_fn else {}
        nm = transition_for(transition_lookup, s(r.get("filename")), next_fn) if next_fn else {}

        ai_missing_manual = ai_span not in manual_span_set
        rel = span_relation(manual_span, ai_span)
        if ai_missing_manual and adj.startswith("AI_SUPPORTED"):
            priority = "HIGH"
        elif rel.startswith("OFF_BY_ONE"):
            priority = "HIGH"
        elif adj in {"MANUAL_SUPPORTED","AI_SUPPORTED","AMBIGUOUS_BOUNDARY"}:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        span_rows.append({
            "manual_filename": s(r.get("filename")),
            "manual_span": manual_span,
            "ai_span": ai_span,
            "span_relation": rel,
            "geometry_adjudication": adj,
            "priority": priority,
            "matched_raw_filename": raw_name,
            "source_match_confidence": s(r.get("source_match_confidence")),
            "target_chainage": target_ch,
            "reference_assignment_status": assignment,
            "reference_confidence": s(ir.get("reference_confidence")),
            "primary_span_method": s(ir.get("primary_span_method")),
            "optical_target_confidence": s(ir.get("optical_target_confidence")),
            "manual_span_from": span_map.get(manual_span, {}).get("chainage_from", np.nan),
            "manual_span_to": span_map.get(manual_span, {}).get("chainage_to", np.nan),
            "manual_target_inside": mm["inside"],
            "manual_distance_to_span": mm["distance_to_span"],
            "manual_distance_to_centre": mm["distance_to_centre"],
            "manual_distance_to_boundary": mm["distance_to_boundary"],
            "ai_span_from": span_map.get(ai_span, {}).get("chainage_from", np.nan),
            "ai_span_to": span_map.get(ai_span, {}).get("chainage_to", np.nan),
            "ai_target_inside": am["inside"],
            "ai_distance_to_span": am["distance_to_span"],
            "ai_distance_to_centre": am["distance_to_centre"],
            "ai_distance_to_boundary": am["distance_to_boundary"],
            "ai_span_is_missing_from_manual": ai_missing_manual,
            "previous_manual_filename": prev_fn,
            "previous_manual_span": s(prev.get("span_id")) if prev is not None else "",
            "previous_ai_span": s(prev.get("idx_primary_span")) if prev is not None else "",
            "prev_transition_status": s(pm.get("visual_status")),
            "prev_dino_visual_score": pm.get("visual_score", np.nan),
            "prev_dino_global_similarity": pm.get("global_similarity", np.nan),
            "next_manual_filename": next_fn,
            "next_manual_span": s(nxt.get("span_id")) if nxt is not None else "",
            "next_ai_span": s(nxt.get("idx_primary_span")) if nxt is not None else "",
            "next_transition_status": s(nm.get("visual_status")),
            "next_dino_visual_score": nm.get("visual_score", np.nan),
            "next_dino_global_similarity": nm.get("global_similarity", np.nan),
            "human_decision": "",
            "human_notes": "",
        })

    span_df = pd.DataFrame(span_rows)
    if not span_df.empty:
        span_df = span_df.sort_values(["priority","manual_span"], ascending=[True, True])

    print("[3/7] Building possible wrong-pass / classifier-conflict set...")
    wp = audit[
        audit["source_match_confidence"].astype(str).isin(["HIGH","MEDIUM"])
        & audit["manual_audit_flags"].astype(str).str.contains("POSSIBLE_WRONG_PASS", na=False)
    ].copy()

    # Count manual images associated with each model pass.
    pass_counts = audit[audit["source_match_confidence"].astype(str).isin(["HIGH","MEDIUM"])].groupby(
        audit["idx_pass_id"].astype(str)
    ).size().to_dict()

    pass_rows = []
    for _, r in wp.iterrows():
        raw_name = s(r.get("matched_raw_filename"))
        ir = img_by_file.get(raw_name, {})
        pid = s(ir.get("pass_id") or r.get("idx_pass_id"))
        pdirection = s(ir.get("pass_direction") or r.get("idx_pass_direction"))
        pr = pass_by_id.get(pid, {})

        same_pass_n = int(pass_counts.get(pid, 0)) if pid else 0

        prevs, nexts = neighbor_rows(audit, r, count=1)
        prev = prevs[-1] if prevs else None
        nxt = nexts[0] if nexts else None
        prev_fn = s(prev.get("filename")) if prev is not None else ""
        next_fn = s(nxt.get("filename")) if nxt is not None else ""
        pm = transition_for(transition_lookup, prev_fn, s(r.get("filename"))) if prev_fn else {}
        nm = transition_for(transition_lookup, s(r.get("filename")), next_fn) if next_fn else {}
        statuses = [s(pm.get("visual_status")), s(nm.get("visual_status"))]
        statuses = [x for x in statuses if x]
        dino_context = ";".join(statuses) if statuses else "NO_AUDITABLE_NEIGHBOR_TRANSITION"
        dino_ok = statuses and all(x in {"CONTINUOUS","GEOMETRY_JUMP_VISUAL_OK"} for x in statuses)

        if pdirection in {"CROSSING_PASS","UNKNOWN_PASS","OUTSIDE_ALIGNMENT"} and same_pass_n >= 2 and dino_ok:
            review_class = "CLASSIFIER_CONFLICT_MANUAL_SEQUENCE_COHERENT"
        elif pdirection in {"CROSSING_PASS","UNKNOWN_PASS","OUTSIDE_ALIGNMENT"} and same_pass_n >= 2:
            review_class = "CLASSIFIER_CONFLICT_MULTI_MANUAL_FRAMES"
        elif pdirection in {"CROSSING_PASS","UNKNOWN_PASS","OUTSIDE_ALIGNMENT"}:
            review_class = "ISOLATED_WRONG_PASS_SUSPECT"
        else:
            review_class = "PASS_CLASSIFICATION_REVIEW"

        # Acquisition neighbours from full image index, not only manual product.
        prev_raw = next_raw = ""
        prev_raw_dir = next_raw_dir = ""
        prev_raw_ch = next_raw_ch = np.nan
        eo_source = s(ir.get("eo_source"))
        eo_order = n(ir.get("eo_order"))
        if eo_source and math.isfinite(eo_order) and "eo_source" in img.columns and "eo_order" in img.columns:
            g = img[img["eo_source"].astype(str).eq(eo_source)].copy()
            g["_eo"] = pd.to_numeric(g["eo_order"], errors="coerce")
            g = g[g["_eo"].notna()].sort_values("_eo")
            before = g[g["_eo"] < eo_order].tail(1)
            after = g[g["_eo"] > eo_order].head(1)
            if len(before):
                rr = before.iloc[0]
                prev_raw = s(rr.get("filename")); prev_raw_dir = s(rr.get("pass_direction")); prev_raw_ch = n(rr.get("camera_chainage"))
            if len(after):
                rr = after.iloc[0]
                next_raw = s(rr.get("filename")); next_raw_dir = s(rr.get("pass_direction")); next_raw_ch = n(rr.get("camera_chainage"))

        pass_rows.append({
            "manual_filename": s(r.get("filename")),
            "manual_span": s(r.get("span_id")),
            "matched_raw_filename": raw_name,
            "source_match_confidence": s(r.get("source_match_confidence")),
            "model_primary_span": s(ir.get("primary_span")),
            "model_pass_id": pid,
            "model_pass_direction": pdirection,
            "alignment_progression_ratio": n(pr.get("alignment_progression_ratio")),
            "pass_image_count": n(pr.get("image_count")),
            "pass_chainage_start": n(pr.get("chainage_start")),
            "pass_chainage_end": n(pr.get("chainage_end")),
            "manual_images_from_same_pass": same_pass_n,
            "camera_chainage": n(ir.get("camera_chainage")),
            "reference_chainage": n(ir.get("reference_chainage")),
            "previous_raw": prev_raw,
            "previous_raw_pass_direction": prev_raw_dir,
            "previous_raw_camera_chainage": prev_raw_ch,
            "next_raw": next_raw,
            "next_raw_pass_direction": next_raw_dir,
            "next_raw_camera_chainage": next_raw_ch,
            "previous_manual_filename": prev_fn,
            "next_manual_filename": next_fn,
            "previous_transition_status": s(pm.get("visual_status")),
            "next_transition_status": s(nm.get("visual_status")),
            "dino_context": dino_context,
            "pass_review_class": review_class,
            "human_decision": "",
            "human_notes": "",
        })

    pass_df = pd.DataFrame(pass_rows)

    print("[4/7] Writing CSV adjudication tables...")
    span_df.to_csv(out_dir / "span_disagreements.csv", index=False)
    pass_df.to_csv(out_dir / "wrong_pass_cases.csv", index=False)

    # One human-review sheet containing both problem classes.
    review_rows = []
    for _, r in span_df.iterrows():
        review_rows.append({
            "review_type": "SPAN_DISAGREEMENT",
            "manual_filename": r.get("manual_filename"),
            "manual_span": r.get("manual_span"),
            "ai_span": r.get("ai_span"),
            "auto_evidence_class": r.get("geometry_adjudication"),
            "priority": r.get("priority"),
            "human_decision": "",
            "human_notes": "",
        })
    for _, r in pass_df.iterrows():
        review_rows.append({
            "review_type": "PASS_CLASSIFICATION",
            "manual_filename": r.get("manual_filename"),
            "manual_span": r.get("manual_span"),
            "ai_span": r.get("model_primary_span"),
            "auto_evidence_class": r.get("pass_review_class"),
            "priority": "HIGH",
            "human_decision": "",
            "human_notes": "",
        })
    pd.DataFrame(review_rows).to_csv(out_dir / "human_review_template.csv", index=False)

    print("[5/7] Rendering span-disagreement contact sheets...")
    render_count = 0
    for _, r in span_df.iterrows():
        if args.render_limit >= 0 and render_count >= args.render_limit:
            break
        panels = []
        for role, fn, mspan, aspan in [
            ("PREVIOUS", r.get("previous_manual_filename"), r.get("previous_manual_span"), r.get("previous_ai_span")),
            ("CURRENT", r.get("manual_filename"), r.get("manual_span"), r.get("ai_span")),
            ("NEXT", r.get("next_manual_filename"), r.get("next_manual_span"), r.get("next_ai_span")),
        ]:
            fn = s(fn)
            if fn and fn in manual_paths:
                panels.append(annotate_thumb(
                    manual_paths[fn],
                    f"{role}: {fn}",
                    [f"manual span: {s(mspan)}", f"AI/model span: {s(aspan)}"],
                ))
        dest = out_dir / "review" / "span_disagreements" / f"{s(r.get('manual_filename')).replace('.jpg','')}__{s(r.get('geometry_adjudication'))}.jpg"
        status = stack_panels(
            panels,
            [
                f"manual: {r.get('manual_span')}   AI: {r.get('ai_span')}   relation: {r.get('span_relation')}",
                f"geometry: {r.get('geometry_adjudication')}   target chainage: {r.get('target_chainage')}",
                f"manual centre distance: {r.get('manual_distance_to_centre')}   AI centre distance: {r.get('ai_distance_to_centre')}",
                f"AI span missing from manual product: {r.get('ai_span_is_missing_from_manual')}",
                f"prev DINO: {r.get('prev_transition_status')} / {r.get('prev_dino_visual_score')}",
                f"next DINO: {r.get('next_transition_status')} / {r.get('next_dino_visual_score')}",
            ],
            dest
        )
        render_count += status == "RENDERED"

    print("[6/7] Rendering wrong-pass contact sheets...")
    render_count = 0
    for _, r in pass_df.iterrows():
        if args.render_limit >= 0 and render_count >= args.render_limit:
            break
        panels = []
        for role, rawfn in [("RAW PREVIOUS", r.get("previous_raw")), ("RAW CURRENT", r.get("matched_raw_filename")), ("RAW NEXT", r.get("next_raw"))]:
            rawfn = s(rawfn)
            if rawfn and rawfn in raw_paths:
                rr = img_by_file.get(rawfn, {})
                panels.append(annotate_thumb(
                    raw_paths[rawfn],
                    f"{role}: {rawfn}",
                    [
                        f"pass: {s(rr.get('pass_direction'))}  pass_id: {s(rr.get('pass_id'))}",
                        f"camera chainage: {s(rr.get('camera_chainage'))}",
                        f"primary span: {s(rr.get('primary_span'))}",
                    ],
                ))
        # include delivered manual frame if available
        mf = s(r.get("manual_filename"))
        if mf in manual_paths:
            panels.append(annotate_thumb(
                manual_paths[mf],
                f"MANUAL DELIVERED: {mf}",
                [f"manual span: {s(r.get('manual_span'))}", f"model span: {s(r.get('model_primary_span'))}"],
            ))
        dest = out_dir / "review" / "wrong_pass" / f"{mf.replace('.jpg','')}__{s(r.get('model_pass_direction'))}.jpg"
        status = stack_panels(
            panels,
            [
                f"class: {r.get('pass_review_class')}",
                f"pass direction: {r.get('model_pass_direction')}   progression ratio: {r.get('alignment_progression_ratio')}",
                f"manual images from same pass: {r.get('manual_images_from_same_pass')}",
                f"DINO context: {r.get('dino_context')}",
            ],
            dest
        )
        render_count += status == "RENDERED"

    print("[7/7] Writing summary and HTML report...")
    summary = {
        "clean_span_disagreements": int(len(span_df)),
        "off_by_one_backward": int((span_df.get("span_relation", pd.Series(dtype=str)) == "OFF_BY_ONE_BACKWARD").sum()),
        "off_by_one_forward": int((span_df.get("span_relation", pd.Series(dtype=str)) == "OFF_BY_ONE_FORWARD").sum()),
        "other_span_disagreements": int((~span_df.get("span_relation", pd.Series(dtype=str)).isin(["OFF_BY_ONE_BACKWARD","OFF_BY_ONE_FORWARD"])).sum()) if len(span_df) else 0,
        "geometry_adjudication_counts": span_df["geometry_adjudication"].value_counts().to_dict() if len(span_df) else {},
        "ai_span_is_missing_manual": int(pd.Series(span_df.get("ai_span_is_missing_from_manual", [])).fillna(False).astype(bool).sum()) if len(span_df) else 0,
        "wrong_pass_cases": int(len(pass_df)),
        "wrong_pass_direction_counts": pass_df["model_pass_direction"].value_counts().to_dict() if len(pass_df) else {},
        "pass_review_class_counts": pass_df["pass_review_class"].value_counts().to_dict() if len(pass_df) else {},
        "classifier_conflict_cases": int(pass_df["pass_review_class"].astype(str).str.startswith("CLASSIFIER_CONFLICT").sum()) if len(pass_df) else 0,
        "project_root": str(root),
        "manual_audit_source": str(audit_dir),
        "ai_run_source": str(ai),
        "method_note": "Evidence pack only. No automatic correction is applied to manual or AI labels/pass membership.",
    }
    (out_dir / "adjudication_summary.json").write_text(json.dumps(summary, indent=2, default=safe_json), encoding="utf-8")
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k,v in summary.items()}
    pd.DataFrame([flat]).to_csv(out_dir / "adjudication_summary.csv", index=False)
    write_html(out_dir, summary, span_df, pass_df)

    print()
    print("Manual vs AI adjudication pack complete")
    print(f"  Clean span disagreements:        {summary['clean_span_disagreements']}")
    print(f"  Off-by-one backward:             {summary['off_by_one_backward']}")
    print(f"  Off-by-one forward:              {summary['off_by_one_forward']}")
    print(f"  AI span missing in manual:       {summary['ai_span_is_missing_manual']}")
    print(f"  Wrong-pass review cases:         {summary['wrong_pass_cases']}")
    print(f"  Classifier-conflict cases:       {summary['classifier_conflict_cases']}")
    print(f"  Output:                          {out_dir}")
    print(f"  Report:                          {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
