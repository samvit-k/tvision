"""Step 1: programmatic exploration and validation of the raw NuCLS dataset."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tvision import nucls_data as nd
from tvision.config import Classes, Paths
from tvision.draw import CLASS_COLORS, draw_boxes, side_by_side

OFFICIAL_FOLDS = [1, 2, 3, 4, 5]
COORDS = ["xmin", "ymin", "xmax", "ymax"]
FOV_COLS = ["fov_x0", "fov_y0", "fov_x1", "fov_y1"]
EDGE_TOLERANCE_PX = 2  # observed right/bottom rounding overshoot; clipped during conversion

def _block(title: str, obj) -> str:
    text = obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else str(obj)
    return f"### {title}\n```\n{text}\n```\n"

def check_coords_numeric(ann: pd.DataFrame) -> dict:
    """Coordinates must be present and numeric before any geometry is trusted."""
    bad = ann[COORDS].apply(pd.to_numeric, errors="coerce").isna()
    per_col = {c: int(n) for c, n in bad.sum().items() if n}
    return {"passed": not per_col, "detail": per_col or 0}

def check_geometry(ann: pd.DataFrame, index: pd.DataFrame) -> dict[str, dict]:
    a = ann.merge(index[["stem", "width", "height"]], on="stem")
    inverted = (a["xmax"] < a["xmin"]) | (a["ymax"] < a["ymin"])
    left_top = (a["xmin"] < 0) | (a["ymin"] < 0)
    right_bottom = pd.concat([a["xmax"] - a["width"], a["ymax"] - a["height"]], axis=1).max(axis=1)
    zero_area = (a["type"] != "point") & ((a["xmax"] <= a["xmin"]) | (a["ymax"] <= a["ymin"]))
    return {
        "no_inverted_boxes": {"passed": not inverted.any(), "detail": int(inverted.sum())},
        "boxes_within_image": {
            "passed": not left_top.any() and bool((right_bottom <= EDGE_TOLERANCE_PX).all()),
            "detail": {"left_top_overshoot": int(left_top.sum()),
                       "right_bottom_beyond_0px": int((right_bottom > 0).sum()),
                       "right_bottom_beyond_tolerance": int((right_bottom > EDGE_TOLERANCE_PX).sum()),
                       "max_right_bottom_px": float(right_bottom.max())},
        },
        "no_zero_area_non_point_boxes": {"passed": not zero_area.any(), "detail": int(zero_area.sum())},
    }

def check_fov_rects(index: pd.DataFrame) -> dict:
    """Every mask defines an FOV rectangle inside its RGB image, up to the right/bottom edge tolerance.

    Mask and RGB sizes differ by up to 2 px in some FOVs; rectangles are clipped during conversion.
    """
    missing = index[FOV_COLS].isna().any(axis=1)
    r = index[~missing]
    left_top = (r["fov_x0"] < 0) | (r["fov_y0"] < 0)
    right_bottom = pd.concat([r["fov_x1"] - r["width"], r["fov_y1"] - r["height"]], axis=1).max(axis=1)
    beyond_tol = right_bottom > EDGE_TOLERANCE_PX
    return {"passed": not missing.any() and not left_top.any() and not beyond_tol.any(),
            "detail": {"missing": int(missing.sum()), "left_top_overshoot": int(left_top.sum()),
                       "right_bottom_beyond_0px": int((right_bottom > 0).sum()),
                       "right_bottom_beyond_tolerance": int(beyond_tol.sum())}}

def fov_box_visibility(ann: pd.DataFrame, index: pd.DataFrame) -> pd.Series:
    """Fraction of each box's area that lies inside its FOV rectangle."""
    a = ann[["stem", *COORDS]].merge(index[["stem", *FOV_COLS]], on="stem", how="left")
    iw = (np.minimum(a["xmax"], a["fov_x1"]) - np.maximum(a["xmin"], a["fov_x0"])).clip(lower=0)
    ih = (np.minimum(a["ymax"], a["fov_y1"]) - np.maximum(a["ymin"], a["fov_y0"])).clip(lower=0)
    area = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    return (iw * ih / area).set_axis(ann.index)

def check_folds(index: pd.DataFrame, raw_dir: Path) -> dict:
    data_slides = set(index["slide"])
    detail, ok = {}, True
    for k in OFFICIAL_FOLDS:
        tr, te = nd.load_split_file(raw_dir, k, "train"), nd.load_split_file(raw_dir, k, "test")
        tr_s, te_s = set(tr["slide_name"]), set(te["slide_name"])
        d = {
            "slide_overlap": len(tr_s & te_s),
            "hospital_overlap": len(set(tr["hospital"]) & set(te["hospital"])),
            "slides_not_in_data": len((tr_s | te_s) - data_slides),
            "data_slides_unassigned": len(data_slides - (tr_s | te_s)),
        }
        detail[k] = d
        ok &= not any(d.values())
    return {"passed": ok, "detail": detail}

def run_checks(index: pd.DataFrame, ann: pd.DataFrame, raw_dir: Path) -> dict[str, dict]:
    """Dataset invariants. Each entry: {'passed': bool, 'detail': ...}."""
    missing = nd.check_file_pairing(raw_dir)
    hierarchy = ann.groupby("raw_classification")[["main_classification", "super_classification"]].nunique()
    inconsistent = hierarchy[(hierarchy > 1).any(axis=1)].index.tolist()
    no_ann = set(index["stem"]) - set(ann["stem"])
    checks = {
        "file_pairing": {"passed": not missing, "detail": {k: len(v) for k, v in missing.items()}},
        "raw_to_super_consistent": {"passed": not inconsistent, "detail": inconsistent},
        "every_fov_annotated": {"passed": not no_ann, "detail": len(no_ann)},
        "fov_rects_valid": check_fov_rects(index),
        "coords_numeric": check_coords_numeric(ann),
    }
    if checks["coords_numeric"]["passed"]:
        checks.update(check_geometry(ann, index))
    else:
        checks["geometry"] = {"passed": False, "detail": "skipped: non-numeric or missing coordinates"}
    checks["official_folds_leakage_free_and_complete"] = check_folds(index, raw_dir)
    return checks

def fold_summary(index: pd.DataFrame, ann: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for k in OFFICIAL_FOLDS:
        for part in ("train", "test"):
            slides = set(nd.load_split_file(raw_dir, k, part)["slide_name"])
            fovs = index[index["slide"].isin(slides)]
            counts = ann.loc[ann["stem"].isin(fovs["stem"]), "project_class"].value_counts()
            rows.append({"fold": k, "part": part, "hospitals": fovs["hospital"].nunique(),
                         "slides": len(slides), "fovs": len(fovs), **counts.to_dict()})
    return pd.DataFrame(rows).set_index(["fold", "part"]).fillna(0).astype(int)

def render_overlays(raw_dir: Path, ann: pd.DataFrame, index: pd.DataFrame, stems: list[str],
                    out_dir: Path) -> None:
    """CSV boxes and FOV rectangle (black) over the RGB image, next to the official NuCLS visualization."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rects = index.set_index("stem")[FOV_COLS]
    for stem in stems:
        img = cv2.imread(str(raw_dir / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
        vis = cv2.imread(str(raw_dir / "visualization" / f"{stem}.png"), cv2.IMREAD_COLOR)
        rows = ann[ann["stem"] == stem]
        boxes = rows[rows["type"] != "point"]
        drawn = draw_boxes(img, [{"x1": r.xmin, "y1": r.ymin, "x2": r.xmax, "y2": r.ymax,
                                  "class_name": r.project_class} for r in boxes.itertuples()])
        for r in rows[rows["type"] == "point"].itertuples():
            cv2.drawMarker(drawn, (int(r.xmin), int(r.ymin)), CLASS_COLORS[r.project_class],
                           cv2.MARKER_CROSS, 8, 1)
        if rects.loc[stem].notna().all():
            x0, y0, x1, y1 = (int(v) for v in rects.loc[stem])
            cv2.rectangle(drawn, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 1)
        cv2.imwrite(str(out_dir / f"{stem}.png"), side_by_side(drawn, vis))

def explore(n_overlays: int = 12, seed: int = 0) -> bool:
    paths, classes = Paths.load(), Classes.load()
    raw, out = paths.raw_dir, paths.artifacts_dir / "explore"
    out.mkdir(parents=True, exist_ok=True)

    index = nd.fov_index(raw)
    ann = nd.map_classes(nd.load_annotations(raw), classes)
    checks = run_checks(index, ann, raw)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    for name, c in checks.items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")
    if not checks["coords_numeric"]["passed"]:
        print("Coordinates invalid; geometry statistics and overlays not generated.")
        return False

    ann = ann.merge(index[["stem", "width", "height"]], on="stem")
    ann["box_w"], ann["box_h"] = ann["xmax"] - ann["xmin"], ann["ymax"] - ann["ymin"]
    boxes = ann[~ann["type"].isin(classes.drop_types)]
    pct = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    per_hospital = index.groupby("hospital").agg(slides=("slide", "nunique"), fovs=("stem", "size"))
    mask_mismatch = ((index["mask_height"] != index["height"]) | (index["mask_width"] != index["width"])).sum()
    if checks["fov_rects_valid"]["passed"]:
        fov_area = (index["fov_x1"] - index["fov_x0"]) * (index["fov_y1"] - index["fov_y0"])
        outside_frac = 1 - fov_area / (index["width"] * index["height"])
        vis_bins = pd.cut(fov_box_visibility(boxes, index), [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1],
                          include_lowest=True)
        fov_sections = [
            _block("Image area outside the annotated FOV rectangle (fraction)", outside_frac.describe(pct).round(3)),
            _block("Fraction of box area inside the FOV rectangle × project class",
                   pd.crosstab(vis_bins, boxes["project_class"], margins=True)),
        ]
    else:
        fov_sections = [_block("FOV rectangle analysis", "skipped: fov_rects_valid failed")]

    sections = [
        _block("Totals", f"FOVs: {len(index)}  slides: {index['slide'].nunique()}  "
                         f"hospitals: {index['hospital'].nunique()}  nuclei: {len(ann)}"),
        _block("Checks", json.dumps(checks, indent=2)),
        _block("Raw → main → super → project class", ann.groupby(
            ["raw_classification", "main_classification", "super_classification", "project_class"]).size()),
        _block("Project class counts", ann["project_class"].value_counts()),
        _block("Annotation type × project class", pd.crosstab(ann["type"], ann["project_class"], margins=True)),
        _block("Box width (px), excluding dropped types", boxes.groupby("project_class")["box_w"].describe(pct).round(1)),
        _block("Box height (px), excluding dropped types", boxes.groupby("project_class")["box_h"].describe(pct).round(1)),
        _block("Nuclei per FOV", ann.groupby("stem").size().describe(pct).round(1)),
        _block("Image size (px)", index[["height", "width"]].describe(pct).round(1)),
        _block("Masks with size different from RGB", int(mask_mismatch)),
        *fov_sections,
        _block("Hospitals", per_hospital.sort_values("fovs", ascending=False)),
        _block("Official folds (FOV and project-class counts)", fold_summary(index, ann, raw)),
    ]
    (out / "summary.md").write_text("# NuCLS exploration\n\n" + "\n".join(sections))
    index.to_csv(out / "fov_index.csv", index=False)

    sample = random.Random(seed).sample(sorted(index["stem"]), n_overlays)
    render_overlays(raw, ann, index, sample, out / "overlays")
    print(f"Report: {out / 'summary.md'}")
    return all(c["passed"] for c in checks.values())