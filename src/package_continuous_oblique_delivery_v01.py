#!/usr/bin/env python3
"""Package a continuous-oblique run into a clean delivery tree.

This script does not change image selection, geometry, detector results or 163 labels.
It only reorganises already-rendered outputs, creates clean + tagged structure-crop
packages, writes manifests/coverage tables and optionally creates a ZIP of crops.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

Image.MAX_IMAGE_PIXELS = None


def clean_token(value, fallback="NA") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    s = str(value).strip()
    if not s:
        return fallback
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-._") or fallback


def as_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def finite_text(value, digits=3, fallback="n/a") -> str:
    try:
        f = float(value)
        if math.isfinite(f):
            return f"{f:.{digits}f}"
    except Exception:
        pass
    return fallback


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
        ]
    candidates += [
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


def add_tag_footer(src: Path, dest: Path, row: pd.Series, circuit: str, jpeg_quality: int = 92):
    with Image.open(src) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
        w, h = im.size
        font_size = max(18, int(round(min(w, h) * 0.028)))
        small_size = max(15, int(round(font_size * 0.78)))
        font_b = load_font(font_size, bold=True)
        font_s = load_font(small_size, bold=False)
        pad = max(10, int(round(font_size * 0.55)))
        gap = max(4, int(round(font_size * 0.20)))

        ref_idx = row.get("reference_idx")
        try:
            ref_text = f"{int(float(ref_idx)):03d}"
        except Exception:
            ref_text = "---"
        structure_id = row.get("structure_id")
        direction = str(row.get("crop_bucket") or row.get("pass_direction") or "UNKNOWN").upper()
        source = Path(str(row.get("filename") or "UNKNOWN")).name
        run_id = row.get("circuit_run_id")
        view = row.get("view_direction")
        chainage = finite_text(row.get("chainage"), 1)
        detector = finite_text(row.get("tower_detector_score"), 3)
        assignment = finite_text(row.get("tower_detector_final_score"), 3)
        dino = finite_text(row.get("tower_dino_similarity"), 3)

        line1 = f"{circuit}  |  Structure {ref_text}: {structure_id}  |  {direction}"
        line2 = f"Source: {source}  |  Run: {run_id}  |  View: {view}  |  Chainage: {chainage}"
        line3 = f"Detector: {detector}  |  Assignment: {assignment}  |  Tower DINO: {dino}"

        tmp = Image.new("RGB", (w, 10), "white")
        d = ImageDraw.Draw(tmp)
        b1 = d.textbbox((0, 0), line1, font=font_b)
        b2 = d.textbbox((0, 0), line2, font=font_s)
        b3 = d.textbbox((0, 0), line3, font=font_s)
        footer_h = pad * 2 + (b1[3]-b1[1]) + (b2[3]-b2[1]) + (b3[3]-b3[1]) + gap * 2
        out = Image.new("RGB", (w, h + footer_h), "white")
        out.paste(im, (0, 0))
        draw = ImageDraw.Draw(out)
        draw.line((0, h, w, h), fill=(40, 40, 40), width=max(2, int(round(w / 900))))
        y = h + pad
        draw.text((pad, y), line1, fill=(0, 0, 0), font=font_b)
        y += (b1[3]-b1[1]) + gap
        draw.text((pad, y), line2, fill=(20, 20, 20), font=font_s)
        y += (b2[3]-b2[1]) + gap
        draw.text((pad, y), line3, fill=(20, 20, 20), font=font_s)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, "JPEG", quality=jpeg_quality, optimize=True)


def safe_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def zip_tree(source_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(source_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(source_dir.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Package a completed continuous-oblique run into delivery folders.")
    ap.add_argument("run_output", type=Path, help="Output directory produced by continuous_oblique_v014.py")
    ap.add_argument("--circuit", required=True)
    ap.add_argument("--output-version", default="v1")
    ap.add_argument("--delivery-root", type=Path, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=92, help="Quality for tagged crop previews only.")
    ap.add_argument("--no-tagged-previews", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)

    run = args.run_output.resolve()
    if not run.exists():
        raise SystemExit(f"Run output does not exist: {run}")

    product_csv = run / "product_manifest.csv"
    crop_csv = run / "structure_crop_candidates.csv"
    if not product_csv.exists():
        raise SystemExit(f"Missing {product_csv}")
    if not crop_csv.exists():
        raise SystemExit(f"Missing {crop_csv}")

    circuit = clean_token(args.circuit, "CIRCUIT")
    version = clean_token(args.output_version, "v1")
    delivery = (args.delivery_root or (run / "deliverable")).resolve()
    if delivery.exists():
        shutil.rmtree(delivery)
    delivery.mkdir(parents=True, exist_ok=True)

    pman = pd.read_csv(product_csv, low_memory=False)
    crops = pd.read_csv(crop_csv, low_memory=False)

    # 1) Canonical 163 delivery, flat version folder to mirror the product layout.
    product_dest = delivery / "163_continuous_oblique_imagery" / version
    product_dest.mkdir(parents=True, exist_ok=True)
    product_rows = []
    for _, r in pman.iterrows():
        status = str(r.get("render_status", ""))
        src_s = str(r.get("rendered_path", ""))
        if not status.startswith("RENDERED") or not src_s or src_s.lower() == "nan":
            continue
        src = Path(src_s)
        if not src.exists():
            continue
        generated = str(r.get("generated_filename") or src.name)
        dest = product_dest / Path(generated).name
        safe_copy(src, dest)
        rr = r.to_dict()
        rr["delivery_path"] = str(dest)
        product_rows.append(rr)
    delivery_product_manifest = pd.DataFrame(product_rows)
    delivery_product_manifest.to_csv(delivery / "continuous_163_manifest.csv", index=False)

    # 2) Structure crops. Keep unmodified clean pixels plus separately tagged review copies.
    crop_root = delivery / "structure_crops"
    by_structure = crop_root / "by_structure"
    by_structure.mkdir(parents=True, exist_ok=True)

    rendered = crops[crops.get("crop_render_status", pd.Series(index=crops.index, dtype=object)).astype(str).eq("RENDERED")].copy()
    package_rows = []
    obs_counter = Counter()

    for _, r in rendered.iterrows():
        crop_s = str(r.get("crop_path", ""))
        if not crop_s or crop_s.lower() == "nan":
            continue
        src = Path(crop_s)
        if not src.exists():
            continue

        sid = clean_token(r.get("structure_id"), "UNKNOWN_STRUCTURE")
        try:
            ridx = int(float(r.get("reference_idx")))
            ridx_txt = f"{ridx:03d}"
        except Exception:
            ridx_txt = "---"
        bucket = clean_token(r.get("crop_bucket") or r.get("pass_direction"), "other_views").lower()
        key = (ridx_txt, sid, bucket)
        obs_counter[key] += 1
        obs_no = obs_counter[key]
        source_stem = clean_token(Path(str(r.get("filename") or src.name)).stem, "SOURCE")
        out_name = f"{circuit}__STR_{ridx_txt}_{sid}__{bucket.upper()}__OBS_{obs_no:03d}__SRC_{source_stem}.jpg"

        structure_dir = by_structure / f"{ridx_txt}_{sid}"
        clean_dest = structure_dir / bucket / "clean" / out_name
        tagged_dest = structure_dir / bucket / "tagged" / out_name
        safe_copy(src, clean_dest)
        if not args.no_tagged_previews:
            add_tag_footer(src, tagged_dest, r, circuit, args.jpeg_quality)

        rr = r.to_dict()
        rr.update({
            "package_structure_ref": ridx_txt,
            "package_direction": bucket,
            "package_observation": obs_no,
            "clean_delivery_path": str(clean_dest),
            "tagged_delivery_path": "" if args.no_tagged_previews else str(tagged_dest),
            "package_filename": out_name,
        })
        package_rows.append(rr)

    crop_manifest = pd.DataFrame(package_rows)
    crop_manifest.to_csv(crop_root / "structure_crops_manifest.csv", index=False)

    # Per-structure observations.csv files.
    if not crop_manifest.empty:
        for (ridx_txt, sid), grp in crop_manifest.groupby(["package_structure_ref", "structure_id"], dropna=False):
            structure_dir = by_structure / f"{clean_token(ridx_txt, '---')}_{clean_token(sid, 'UNKNOWN_STRUCTURE')}"
            grp.to_csv(structure_dir / "observations.csv", index=False)

    # 3) Structure coverage, including reference structures with no rendered crop.
    topo_path = run / "reference_topology.csv"
    coverage_rows = []
    if topo_path.exists():
        topo = pd.read_csv(topo_path, low_memory=False)
        if "is_reference_structure" in topo.columns:
            topo = topo[as_bool_series(topo["is_reference_structure"])].copy()
        counts = Counter()
        bucket_counts = Counter()
        if not crop_manifest.empty:
            for _, r in crop_manifest.iterrows():
                sid = str(r.get("structure_id"))
                counts[sid] += 1
                bucket_counts[(sid, str(r.get("package_direction")))] += 1
        for _, r in topo.iterrows():
            sid = str(r.get("structure_id"))
            coverage_rows.append({
                "reference_idx": r.get("reference_idx"),
                "structure_idx": r.get("structure_idx"),
                "structure_id": sid,
                "chainage": r.get("chainage"),
                "rendered_crop_count": counts[sid],
                "forward_crop_count": bucket_counts[(sid, "forward")],
                "backward_crop_count": bucket_counts[(sid, "backward")],
                "other_view_crop_count": bucket_counts[(sid, "other_views")],
                "crop_status": "CROPPED" if counts[sid] else "NO_RENDERED_CROP",
            })
        pd.DataFrame(coverage_rows).to_csv(crop_root / "structure_coverage.csv", index=False)

    # 4) Copy key QA/metadata files without duplicating giant debug JPGs.
    qa_meta = delivery / "qa_metadata"
    qa_meta.mkdir(parents=True, exist_ok=True)
    meta_names = [
        "summary.json", "direction_coverage.csv", "direction_coverage_confirmed.csv",
        "direction_coverage_163.csv", "direction_gaps.csv", "direction_gaps_163.csv",
        "gap_events.csv", "transitions.csv", "visual_transitions.csv", "flight_passes.csv",
        "circuit_runs.csv", "reference_topology.csv", "reference_spans.csv", "span_completeness.csv",
        "existing_163_benchmark.csv", "opk_benchmark.csv", "tower_detector_summary.json",
    ]
    copied_meta = []
    for name in meta_names:
        src = run / name
        if src.exists() and src.is_file():
            safe_copy(src, qa_meta / name)
            copied_meta.append(name)

    crop_count = len(crop_manifest)
    structure_count = int(crop_manifest["structure_id"].nunique()) if not crop_manifest.empty and "structure_id" in crop_manifest.columns else 0
    expected_structures = len(coverage_rows)
    missing_structures = sum(1 for x in coverage_rows if x["crop_status"] != "CROPPED")
    directions = dict(Counter(crop_manifest["package_direction"])) if not crop_manifest.empty else {}

    readme = f"""Continuous Oblique Delivery Package
===================================
Circuit: {circuit}
Output version: {version}
Source run: {run}

163 deliverable
---------------
Folder: 163_continuous_oblique_imagery/{version}
Images copied: {len(delivery_product_manifest)}
Manifest: continuous_163_manifest.csv

Structure crops
---------------
Folder: structure_crops/by_structure
Rendered crops: {crop_count}
Structures represented: {structure_count}
Expected reference structures: {expected_structures if expected_structures else 'n/a'}
Reference structures without a rendered crop: {missing_structures if expected_structures else 'n/a'}
Directions: {json.dumps(directions, sort_keys=True)}

Each structure folder contains clean, unmodified crops and tagged review copies.
The clean images are intended for downstream matching/analysis. Tagged images are for human QA only.
Global manifest: structure_crops/structure_crops_manifest.csv
Coverage table: structure_crops/structure_coverage.csv
Each structure folder also contains observations.csv.

QA metadata
-----------
Folder: qa_metadata
Copied files: {', '.join(copied_meta) if copied_meta else 'none'}

Important
---------
Packaging does not alter continuous-163 membership, span labels, detector decisions or crop geometry.
All selection/geometry decisions come from the source run.
"""
    (delivery / "README.txt").write_text(readme, encoding="utf-8")

    package_summary = {
        "circuit": circuit,
        "output_version": version,
        "source_run": str(run),
        "delivery_root": str(delivery),
        "continuous_163_images": int(len(delivery_product_manifest)),
        "structure_crops": int(crop_count),
        "structures_represented": int(structure_count),
        "expected_reference_structures": int(expected_structures),
        "reference_structures_without_crop": int(missing_structures),
        "crop_directions": directions,
        "qa_metadata_files": copied_meta,
    }
    (delivery / "delivery_summary.json").write_text(json.dumps(package_summary, indent=2), encoding="utf-8")

    zip_path = None
    if not args.no_zip:
        zip_path = delivery / f"{circuit}_structure_crops_{version}.zip"
        # Zip only structure_crops, not full-resolution 163 imagery.
        zip_tree(crop_root, zip_path)

    print("\nDelivery package complete")
    print(f"  163 images:          {len(delivery_product_manifest)}")
    print(f"  structure crops:     {crop_count}")
    print(f"  structures covered:  {structure_count}/{expected_structures if expected_structures else '?'}")
    print(f"  delivery root:       {delivery}")
    if zip_path:
        print(f"  structure crop ZIP:  {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
