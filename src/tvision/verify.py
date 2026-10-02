"""Step 4: independent verification of the YOLO conversion.

Expected labels are recomputed from the raw CSVs and masks with code separate from `tvision.convert`
and compared with the images, label files and data.yaml actually on disk. Step 3's manifest.csv and
annotations.csv are only cross-checked (`audit_files_agree`), never used as ground truth. Nothing under
the raw or processed directories is written. Overlays rebuild boxes from the label text alone.
"""

import hashlib
import json
import math
import random
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from tvision import nucls_data as nd
from tvision.config import Classes, Convert, Paths
from tvision.draw import CLASS_COLORS, draw_boxes, side_by_side
from tvision.explore import EDGE_TOLERANCE_PX, _block

SPLITS = ["train", "val", "test"]
KINDS = {"images": ".png", "labels": ".txt"}
STATUSES = ["kept", "dropped_type", "dropped_duplicate", "dropped_visibility", "dropped_size"]
SRC = ["xmin", "ymin", "xmax", "ymax"]
BOX = ["x0", "y0", "x1", "y1"]
RECT = ["crop_x0", "crop_y0", "crop_x1", "crop_y1"]
MATCH_TOL_PX = 0.01  # 6-decimal labels reproduce edges within 1e-6 × crop size (≤ 0.001 px here)
EDGE_ROUNDING_TOL = 1e-6
OVERLAY_SCALE = 2

def independent_fov_rect(mask_path: Path, width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """FOV rectangle (x0, y0, x1, y1), exclusive end, clipped to the RGB image; and the mask (height, width)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        raise OSError(f"Cannot read mask: {mask_path}")
    red = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[..., 0]
    x, y, w, h = cv2.boundingRect((red == nd.FOV_CODE).astype(np.uint8))
    x1, y1 = min(x + w, width), min(y + h, height)
    if w == 0 or h == 0 or x1 <= x or y1 <= y:
        raise ValueError(f"No FOV rectangle inside the image: {mask_path.name}")
    return (x, y, x1, y1), mask.shape[:2]

def expected_fov(csv_path: Path, rect: tuple[int, int, int, int], width: int, height: int,
                 classes: Classes, cfg: Convert) -> pd.DataFrame:
    """Per CSV row, in row order: class, source box, status and, if kept, the crop-frame box."""
    x0, y0, x1, y1 = rect
    csv = pd.read_csv(csv_path, index_col=0)
    rows, seen = [], set()
    for r in csv[["super_classification", "type", *SRC]].itertuples():
        if r.super_classification not in classes.super_to_class:
            raise ValueError(f"{csv_path.name}: unmapped super class {r.super_classification}")
        name = classes.super_to_class[r.super_classification]
        row = {"ann_id": r.Index, "project_class": name, "class_id": classes.names.index(name), "type": r.type,
               **dict(zip(SRC, (r.xmin, r.ymin, r.xmax, r.ymax))), "visibility": np.nan, "status": "kept",
               **dict.fromkeys(BOX, np.nan)}
        rows.append(row)
        if r.type in classes.drop_types:
            row["status"] = "dropped_type"
            continue
        if r.xmax <= r.xmin or r.ymax <= r.ymin or r.xmin < 0 or r.ymin < 0 or \
                r.xmax > width + EDGE_TOLERANCE_PX or r.ymax > height + EDGE_TOLERANCE_PX:
            raise ValueError(f"{csv_path.name} row {r.Index}: box violates the Step 1 RGB edge rule")
        key = (row["class_id"], r.xmin, r.ymin, r.xmax, r.ymax)
        if key in seen:
            row["status"] = "dropped_duplicate"
            continue
        seen.add(key)
        bx1, by1 = min(r.xmax, width), min(r.ymax, height)
        ix0, iy0, ix1, iy1 = max(r.xmin, x0), max(r.ymin, y0), min(bx1, x1), min(by1, y1)
        iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
        row["visibility"] = iw * ih / ((bx1 - r.xmin) * (by1 - r.ymin))
        if row["visibility"] < cfg.min_visibility:
            row["status"] = "dropped_visibility"
        elif iw < cfg.min_size_px or ih < cfg.min_size_px:
            row["status"] = "dropped_size"
        else:
            row.update(zip(BOX, (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0)))
    return pd.DataFrame(rows, columns=["ann_id", "project_class", "class_id", "type", *SRC, "visibility",
                                       "status", *BOX])

def parse_line(line: str, n_classes: int) -> tuple[int, float, float, float, float] | None:
    """`class cx cy w h` with single spaces, or None if the line is not a valid normalized YOLO box."""
    fields = line.split(" ")
    if len(fields) != 5:
        return None
    try:
        c, values = int(fields[0]), [float(f) for f in fields[1:]]
    except ValueError:
        return None
    cx, cy, w, h = values
    if not (0 <= c < n_classes and w > 0 and h > 0 and all(math.isfinite(v) and 0 <= v <= 1 for v in values)):
        return None
    if not all(-EDGE_ROUNDING_TOL <= e <= 1 + EDGE_ROUNDING_TOL for e in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)):
        return None
    return c, cx, cy, w, h

def parse_label_text(text: str, n_classes: int) -> tuple[list[str], list[tuple | None], bool]:
    """Lines, each parsed (or None), and whether the file is empty or ends with a newline."""
    lines = text.split("\n")
    layout_ok = lines[-1] == ""
    lines = lines[:-1] if layout_ok else lines
    return lines, [parse_line(l, n_classes) for l in lines], layout_ok

def label_boxes_px(parsed: list[tuple | None], crop_w: int, crop_h: int) -> list[tuple[int, float, float, float, float]]:
    """Valid parsed lines as (class_id, x0, y0, x1, y1) in crop pixels."""
    return [(c, (cx - w / 2) * crop_w, (cy - h / 2) * crop_h, (cx + w / 2) * crop_w, (cy + h / 2) * crop_h)
            for c, cx, cy, w, h in (p for p in parsed if p is not None)]

def files_on_disk(processed: Path) -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    """Stems per kind and split, and every other path (Ultralytics' labels/<split>.cache is allowed)."""
    found, stray = {k: {s: set() for s in SPLITS} for k in KINDS}, []
    for kind, ext in KINDS.items():
        root = processed / kind
        for p in sorted(root.iterdir()) if root.is_dir() else []:
            if p.is_dir() and p.name in SPLITS:
                for q in sorted(p.iterdir()):
                    if q.is_file() and q.suffix == ext:
                        found[kind][p.name].add(q.stem)
                    else:
                        stray.append(q.relative_to(processed).as_posix())
            elif not (kind == "labels" and p.is_file() and p.suffix == ".cache" and p.stem in SPLITS):
                stray.append(p.relative_to(processed).as_posix())
    return found, stray

def check_files(found: dict, stray: list[str], splits: pd.DataFrame) -> dict:
    detail, ok = {}, not stray
    for kind in KINDS:
        for s in SPLITS:
            expected = set(splits.loc[splits["split"] == s, "stem"])
            missing, extra = len(expected - found[kind][s]), len(found[kind][s] - expected)
            detail[f"{kind}/{s}"] = {"files": len(found[kind][s]), "missing": missing, "extra": extra}
            ok &= not missing and not extra
    per_split = Counter(stem for s in SPLITS for stem in found["images"][s] | found["labels"][s])
    in_two = sum(n > 1 for n in per_split.values())
    return {"passed": ok and not in_two, "detail": {**detail, "in_two_splits": in_two, "stray": stray[:10],
                                                    "n_stray": len(stray)}}

def check_leakage(found: dict) -> dict:
    groups = {s: found["images"][s] | found["labels"][s] for s in SPLITS}
    detail = {}
    for a, b in combinations(SPLITS, 2):
        slides_a, slides_b = {nd.slide_of(x) for x in groups[a]}, {nd.slide_of(x) for x in groups[b]}
        detail[f"{a}/{b}"] = {
            "slides": len(slides_a & slides_b),
            "patients": len({nd.patient_of(x) for x in slides_a} & {nd.patient_of(x) for x in slides_b}),
            "hospitals": len({nd.hospital_of(x) for x in slides_a} & {nd.hospital_of(x) for x in slides_b})}
    return {"passed": not any(v for d in detail.values() for v in d.values()), "detail": detail}

def check_data_yaml(processed: Path, classes: Classes) -> dict:
    try:
        data = yaml.safe_load((processed / "data.yaml").read_text())
    except (OSError, yaml.YAMLError) as e:
        return {"passed": False, "detail": f"unreadable: {e}"}
    if not isinstance(data, dict):
        return {"passed": False, "detail": "not a mapping"}
    problems = [] if data.get("path") == str(processed.resolve()) else ["path"]
    problems += [s for s in SPLITS if data.get(s) != f"images/{s}" or not (processed / "images" / s).is_dir()]
    if data.get("names") != dict(enumerate(classes.names)):
        problems.append("names")
    if "nc" in data and data["nc"] != len(classes.names):
        problems.append("nc")
    return {"passed": not problems, "detail": problems or 0}

def check_fov(processed: Path, stem: str, split: str, rgb: np.ndarray, rect: tuple[int, int, int, int],
              exp: pd.DataFrame, n_classes: int) -> dict:
    """Image and label comparison for one FOV against its independent recomputation."""
    x0, y0, x1, y1 = rect
    kept = exp[exp["status"] == "kept"]
    shifted = kept[SRC].to_numpy(float) - [x0, y0, x0, y0]
    out = {"stem": stem, "split": split, **dict(zip(RECT, rect)), "crop_w": x1 - x0, "crop_h": y1 - y0,
           "n_expected": len(kept), "n_dropped": len(exp) - len(kept),
           "n_clipped": int((np.abs(kept[BOX].to_numpy(float) - shifted) > 1e-9).any(axis=1).sum()),
           "label_exists": False, "layout_ok": True, "lines": 0, "invalid": 0, "duplicates": 0,
           "class_mismatch": 0, "coord_mismatch": 0, "max_err_px": np.nan, "labels_match": False,
           "label_classes": []}
    path = processed / "images" / split / f"{stem}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
    if img is None:
        out["image"] = "unreadable" if path.exists() else "missing"
    elif img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        out["image"] = "wrong_format"
    else:
        out["image"] = "ok" if np.array_equal(img, rgb[y0:y1, x0:x1]) else "pixels_differ"
    path = processed / "labels" / split / f"{stem}.txt"
    if not path.exists():
        return out
    lines, parsed, layout_ok = parse_label_text(path.read_text(), n_classes)
    out.update(label_exists=True, layout_ok=layout_ok, lines=len(lines),
               invalid=sum(p is None for p in parsed), duplicates=len(lines) - len(set(lines)),
               label_classes=[p[0] for p in parsed if p is not None])
    if img is None or img.ndim < 2:
        return out
    boxes = label_boxes_px(parsed, img.shape[1], img.shape[0])
    errors = [max(abs(a - b) for a, b in zip(box[1:], e[1:])) for box, e in
              zip(boxes, kept[["class_id", *BOX]].itertuples(index=False))]
    out.update(class_mismatch=sum(box[0] != e for box, e in zip(boxes, kept["class_id"])),
               coord_mismatch=sum(err > MATCH_TOL_PX for err in errors),
               max_err_px=max(errors, default=np.nan))
    out["labels_match"] = not out["invalid"] and len(lines) == len(kept) and \
        not out["class_mismatch"] and not out["coord_mismatch"]
    return out

def check_drop_accounting(raw: Path, ann: pd.DataFrame, fovs: pd.DataFrame, classes: Classes) -> dict:
    """Recomputed statuses must cover every raw CSV row, and recomputed kept counts must equal label lines."""
    source = sum(len(pd.read_csv(p, index_col=0)) for p in sorted((raw / "csv").glob("*.csv")))
    statuses = ann["status"].value_counts().reindex(STATUSES, fill_value=0)
    kept = ann[ann["status"] == "kept"].groupby(["split", "class_id"]).size()
    labels = fovs[["split", "label_classes"]].explode("label_classes").dropna() \
        .astype({"label_classes": int}).groupby(["split", "label_classes"]).size()
    mismatch = [{"split": s, "class": classes.names[c], "kept": int(kept.get((s, c), 0)),
                 "label_lines": int(labels.get((s, c), 0))}
                for s in SPLITS for c in range(len(classes.names)) if kept.get((s, c), 0) != labels.get((s, c), 0)]
    unknown = int((~ann["status"].isin(STATUSES)).sum())
    return {"passed": int(statuses.sum()) == source == len(ann) and not unknown and not mismatch,
            "detail": {"source": source, "recomputed": len(ann), **{k: int(v) for k, v in statuses.items()},
                       "kept_vs_label_mismatch": mismatch}}

def check_audit_files(processed: Path, ann: pd.DataFrame, fovs: pd.DataFrame) -> dict:
    try:
        manifest = pd.read_csv(processed / "manifest.csv")
        audit = pd.read_csv(processed / "annotations.csv")
    except OSError as e:
        return {"passed": False, "detail": f"unreadable: {e}"}
    m = manifest.set_index("stem")[["split", *RECT]]
    f = fovs.set_index("stem")[["split", *RECT]]
    rect_mismatch = len(m.index.symmetric_difference(f.index)) + \
        int((m.loc[m.index.intersection(f.index)] != f.loc[m.index.intersection(f.index)]).any(axis=1).sum())
    a = audit[["stem", "ann_id", "status"]].merge(ann[["stem", "ann_id", "status"]], on=["stem", "ann_id"],
                                                 how="outer", suffixes=("_audit", "_verify"), indicator=True)
    unmatched = int((a["_merge"] != "both").sum())
    status_mismatch = int((a.loc[a["_merge"] == "both", "status_audit"] != a.loc[a["_merge"] == "both", "status_verify"]).sum())
    return {"passed": not rect_mismatch and not unmatched and not status_mismatch,
            "detail": {"manifest_rect_mismatch": rect_mismatch, "annotation_rows_unmatched": unmatched,
                       "status_mismatch": status_mismatch}}

def run_checks(raw: Path, processed: Path, classes: Classes, cfg: Convert) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (checks, per-FOV results, recomputed annotations). Each check: {'passed': bool, 'detail': ...}."""
    splits = pd.read_csv(processed / "splits.csv").sort_values("stem")
    found, stray = files_on_disk(processed)
    fovs, expected = [], []
    for stem, split in zip(splits["stem"], splits["split"]):
        rgb = cv2.imread(str(raw / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise OSError(f"Cannot read raw image: {stem}")
        h, w = rgb.shape[:2]
        rect, mask_shape = independent_fov_rect(raw / "mask" / f"{stem}.png", w, h)
        exp = expected_fov(raw / "csv" / f"{stem}.csv", rect, w, h, classes, cfg)
        expected.append(exp.assign(stem=stem, split=split))
        fovs.append({**check_fov(processed, stem, split, rgb, rect, exp, len(classes.names)),
                     "width": w, "height": h, "mask_mismatch": mask_shape != (h, w),
                     "n_duplicate": int((exp["status"] == "dropped_duplicate").sum())})
    fovs, ann = pd.DataFrame(fovs), pd.concat(expected, ignore_index=True)
    errors = fovs["max_err_px"].dropna()
    checks = {
        "files_complete": check_files(found, stray, splits),
        "no_leakage_in_files": check_leakage(found),
        "data_yaml_valid": check_data_yaml(processed, classes),
        "images_match_raw_crops": {"passed": bool((fovs["image"] == "ok").all()),
                                   "detail": {k: int(v) for k, v in fovs["image"].value_counts().items()}},
        "label_format": {
            "passed": bool(fovs["label_exists"].all() and fovs["layout_ok"].all() and not fovs["invalid"].any()),
            "detail": {"files_missing": int((~fovs["label_exists"]).sum()),
                       "no_trailing_newline": int((~fovs["layout_ok"]).sum()),
                       "lines": int(fovs["lines"].sum()), "invalid_lines": int(fovs["invalid"].sum())}},
        "labels_match_raw": {
            "passed": bool(fovs["labels_match"].all()),
            "detail": {"fovs_mismatched": int((~fovs["labels_match"]).sum()),
                       "count_mismatch": int((fovs["lines"] != fovs["n_expected"]).sum()),
                       "class_mismatch": int(fovs["class_mismatch"].sum()),
                       "coord_mismatch": int(fovs["coord_mismatch"].sum()),
                       "max_err_px": float(errors.max()) if len(errors) else None,
                       "tolerance_px": MATCH_TOL_PX}},
        "no_duplicate_label_lines": {"passed": not fovs["duplicates"].any(),
                                     "detail": int(fovs["duplicates"].sum())},
        "drop_accounting": check_drop_accounting(raw, ann, fovs, classes),
        "audit_files_agree": check_audit_files(processed, ann, fovs),
    }
    return checks, fovs, ann

def _hash_files(root: Path, files: list[Path]) -> dict:
    h = hashlib.sha256()
    for p in files:
        for part in (p.relative_to(root).as_posix().encode(), p.read_bytes()):
            h.update(len(part).to_bytes(8, "big"))
            h.update(part)
    return {"sha256": h.hexdigest(), "files": len(files)}

def dataset_fingerprint(processed: Path) -> dict:
    """SHA-256 of images/, labels/ (excluding Ultralytics' labels/<split>.cache) and data.yaml."""
    fp = {}
    for kind in KINDS:
        root = processed / kind
        files = sorted(p for p in root.rglob("*") if p.is_file() and not
                       (kind == "labels" and p.parent == root and p.suffix == ".cache"))
        fp[kind] = _hash_files(root, files)
    fp["data_yaml"] = {"sha256": hashlib.sha256((processed / "data.yaml").read_bytes()).hexdigest()}
    fp["dataset"] = hashlib.sha256(" ".join(fp[k]["sha256"] for k in (*KINDS, "data_yaml")).encode()).hexdigest()
    return fp

def sample_overlays(fovs: pd.DataFrame, n_random: int, seed: int) -> dict[str, str]:
    """Stem → reason: targeted edge cases first, then a seeded random sample spread over the splits."""
    targeted = {
        "most boxes clipped to the crop": fovs.loc[fovs["n_clipped"].idxmax(), "stem"],
        "most annotations dropped": fovs.loc[fovs["n_dropped"].idxmax(), "stem"],
        "most kept boxes": fovs.loc[fovs["n_expected"].idxmax(), "stem"],
        "largest crop": fovs.loc[fovs[["crop_w", "crop_h"]].max(axis=1).idxmax(), "stem"],
        "RGB/mask size mismatch": next(iter(fovs.loc[fovs["mask_mismatch"], "stem"]), None),
        "duplicate source box dropped": next(iter(fovs.loc[fovs["n_duplicate"] > 0, "stem"]), None),
    }
    reasons: dict[str, str] = {}
    for reason, stem in targeted.items():
        if stem is not None:
            reasons[stem] = f"{reasons[stem]}; {reason}" if stem in reasons else reason
    rng = random.Random(seed)
    for i, s in enumerate(SPLITS):
        pool = sorted(set(fovs.loc[fovs["split"] == s, "stem"]) - set(reasons))
        for stem in rng.sample(pool, min(len(pool), n_random // len(SPLITS) + (i < n_random % len(SPLITS)))):
            reasons[stem] = f"random ({s})"
    return reasons

def _scaled(boxes: list[dict]) -> list[dict]:
    return [{**b, **{k: b[k] * OVERLAY_SCALE for k in ("x1", "y1", "x2", "y2")}} for b in boxes]

def _upscale(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, None, fx=OVERLAY_SCALE, fy=OVERLAY_SCALE, interpolation=cv2.INTER_NEAREST)

def render_overlay(raw: Path, processed: Path, stem: str, split: str, exp: pd.DataFrame,
                   rect: tuple[int, int, int, int], classes: Classes) -> np.ndarray:
    """Raw image with CSV boxes (grey = dropped) and crop (black) | crop with YOLO label boxes | official overlay."""
    img = cv2.imread(str(raw / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
    vis = cv2.imread(str(raw / "visualization" / f"{stem}.png"), cv2.IMREAD_COLOR)
    crop = cv2.imread(str(processed / "images" / split / f"{stem}.png"), cv2.IMREAD_COLOR)
    if img is None or vis is None or crop is None:
        raise OSError(f"Cannot read overlay inputs: {stem}")
    boxes = exp[exp["status"] != "dropped_type"].sort_values("status", key=lambda s: s == "kept", kind="stable")  # kept on top
    left = draw_boxes(_upscale(img), _scaled([
        {"x1": r.xmin, "y1": r.ymin, "x2": r.xmax, "y2": r.ymax,
         "class_name": r.project_class if r.status == "kept" else "dropped"} for r in boxes.itertuples()]))
    for r in exp[exp["status"] == "dropped_type"].itertuples():
        cv2.drawMarker(left, (int(r.xmin * OVERLAY_SCALE), int(r.ymin * OVERLAY_SCALE)),
                       CLASS_COLORS.get(r.project_class, (200, 200, 200)), cv2.MARKER_CROSS, 10, 1)
    x0, y0, x1, y1 = (v * OVERLAY_SCALE for v in rect)
    cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 1)
    _, parsed, _ = parse_label_text((processed / "labels" / split / f"{stem}.txt").read_text(), len(classes.names))
    middle = draw_boxes(_upscale(crop), _scaled([
        {"x1": a, "y1": b, "x2": c, "y2": d, "class_name": classes.names[k]}
        for k, a, b, c, d in label_boxes_px(parsed, crop.shape[1], crop.shape[0])]))
    return side_by_side(left, middle, _upscale(vis))

def summarize(checks: dict, fovs: pd.DataFrame, ann: pd.DataFrame, fingerprint: dict, overlays: dict[str, str],
              cfg: Convert, classes: Classes) -> str:
    kept = ann[ann["status"] == "kept"]
    boxes = ann[~ann["type"].isin(classes.drop_types)]
    conflicting = int((boxes.groupby(["stem", *SRC])["class_id"].nunique() > 1).sum())
    split_of = dict(zip(fovs["stem"], fovs["split"]))
    checklist = "\n".join(f"- [ ] overlays/{split_of[s]}_{s}.png — {r}" for s, r in overlays.items()) or "none"
    sections = [
        _block("Config", f"min_visibility: {cfg.min_visibility}  min_size_px: {cfg.min_size_px}  "
                         f"edge_tolerance_px: {EDGE_TOLERANCE_PX}  match_tolerance_px: {MATCH_TOL_PX}"),
        _block("Checks", json.dumps(checks, indent=2)),
        _block("Dataset fingerprint (fingerprint.json)", json.dumps(fingerprint, indent=2)),
        _block("Recomputed kept boxes: split × class", pd.crosstab(kept["split"], kept["project_class"], margins=True)
               .reindex(index=[*SPLITS, "All"], columns=[*classes.names, "All"], fill_value=0)),
        _block("Recomputed status × split", pd.crosstab(ann["status"], ann["split"], margins=True)
               .reindex(columns=[*SPLITS, "All"], fill_value=0)),
        _block("Info: same source box annotated with different classes (kept)", conflicting),
        _block("Info: crop size (px)", fovs[["crop_w", "crop_h"]].describe([0.01, 0.5, 0.99]).round(1)),
        "### Overlay review checklist\n"
        "Panels: raw image with CSV boxes (grey = dropped, cross = point, black = crop) | "
        "crop with boxes rebuilt from the YOLO label text only | official NuCLS visualization.\n\n" + checklist + "\n",
    ]
    return "# Conversion verification\n\n" + "\n".join(sections)

def run_verification(raw: Path, processed: Path, report: Path, classes: Classes, cfg: Convert,
                     n_overlays: int = 12, seed: int = 0) -> bool:
    shutil.rmtree(report, ignore_errors=True)
    report.mkdir(parents=True)
    checks, fovs, ann = run_checks(raw, processed, classes, cfg)
    (report / "checks.json").write_text(json.dumps(checks, indent=2))
    for name, c in checks.items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")
    fingerprint = dataset_fingerprint(processed)
    (report / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    overlays = {}
    if checks["files_complete"]["passed"]:
        overlays = sample_overlays(fovs, n_overlays, seed)
        (report / "overlays").mkdir()
        by_stem = dict(tuple(ann.groupby("stem")))
        rects = fovs.set_index("stem")
        for stem in overlays:
            split = rects.loc[stem, "split"]
            rect = tuple(int(v) for v in rects.loc[stem, RECT])
            cv2.imwrite(str(report / "overlays" / f"{split}_{stem}.png"),
                        render_overlay(raw, processed, stem, split, by_stem.get(stem, ann.iloc[:0]), rect, classes))
    else:
        print("Files incomplete; overlays not rendered.")
    (report / "summary.md").write_text(summarize(checks, fovs, ann, fingerprint, overlays, cfg, classes))
    print(f"Report: {report / 'summary.md'}")
    return all(c["passed"] for c in checks.values())

def verify(n_overlays: int = 12, seed: int = 0) -> bool:
    paths = Paths.load()
    return run_verification(paths.raw_dir, paths.processed_dir, paths.artifacts_dir / "verify",
                            Classes.load(), Convert.load(), n_overlays, seed)