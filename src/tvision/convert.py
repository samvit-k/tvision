"""Step 3: convert NuCLS FOVs to YOLO detection labels.

Each image is cropped to its annotated FOV rectangle. Per annotation, in order: drop `drop_types` (points),
enforce the Step 1 RGB edge rule and clip to the RGB image, drop exact duplicates (same class and source
box as a lower `ann_id` in the FOV), intersect with the crop, then drop boxes whose visible fraction
(crop intersection / RGB-clipped area) is below `min_visibility` or whose clipped width or height is
below `min_size_px`. Outputs are built in a temporary directory and published all-or-nothing
only if every check passes, so a failed run never leaves partial, mixed or stale outputs.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from tvision import nucls_data as nd
from tvision.config import Classes, Convert, Paths
from tvision.explore import EDGE_TOLERANCE_PX, _block

SPLITS = ["train", "val", "test"]
STATUSES = ["kept", "dropped_type", "dropped_duplicate", "dropped_visibility", "dropped_size"]
DUPLICATE_KEY = ["class_id", "xmin", "ymin", "xmax", "ymax"]
OUTPUTS = ["images", "labels", "manifest.csv", "annotations.csv", "data.yaml"]
TMP_DIR = ".convert_tmp"
BACKUP_DIR = ".convert_backup"
CLIPPED = ["cx0", "cy0", "cx1", "cy1"]
LABEL_DECIMALS = 6
EDGE_ROUNDING_TOL = 10 ** -LABEL_DECIMALS  # cx ± w/2 from rounded values can overshoot [0, 1] by ≤ 5e-7

def crop_rect(fov: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    """FOV rectangle (x0, y0, x1, y1), exclusive end, clipped to the RGB image."""
    if fov is None:
        raise ValueError("FOV rectangle missing")
    x0, y0, x1, y1 = max(fov[0], 0), max(fov[1], 0), min(fov[2], width), min(fov[3], height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty crop rectangle {fov} for image {width}x{height}")
    return x0, y0, x1, y1

def convert_boxes(ann: pd.DataFrame, rect: tuple[int, int, int, int], width: int, height: int,
                  classes: Classes, cfg: Convert) -> pd.DataFrame:
    """Add `visibility`, crop-frame `cx0, cy0, cx1, cy1` and `status` to one FOV's annotations.

    `ann` must not be filtered: duplicates are judged against all of the FOV's rows.
    Raises if a non-dropped box is degenerate, overshoots the RGB image left/top, or overshoots
    right/bottom beyond EDGE_TOLERANCE_PX.
    """
    out = ann.sort_values("ann_id").reset_index(drop=True)
    is_type = out["type"].isin(classes.drop_types).to_numpy()
    b = out[~is_type]
    degenerate = (b["xmax"] <= b["xmin"]) | (b["ymax"] <= b["ymin"])
    if degenerate.any():
        raise ValueError(f"Degenerate boxes: ann_id {b.loc[degenerate, 'ann_id'].tolist()}")
    left_top = (b["xmin"] < 0) | (b["ymin"] < 0)
    beyond = (b["xmax"] - width > EDGE_TOLERANCE_PX) | (b["ymax"] - height > EDGE_TOLERANCE_PX)
    if left_top.any() or beyond.any():
        raise ValueError(f"Boxes outside the RGB edge tolerance: ann_id {b.loc[left_top | beyond, 'ann_id'].tolist()}")
    is_duplicate = np.zeros(len(out), bool)
    is_duplicate[b.index] = b.duplicated(DUPLICATE_KEY, keep="first").to_numpy()

    rx0, rx1 = out["xmin"].clip(0, width).to_numpy(float), out["xmax"].clip(0, width).to_numpy(float)
    ry0, ry1 = out["ymin"].clip(0, height).to_numpy(float), out["ymax"].clip(0, height).to_numpy(float)
    x0, y0, x1, y1 = rect
    ix0, ix1 = np.maximum(rx0, x0), np.minimum(rx1, x1)
    iy0, iy1 = np.maximum(ry0, y0), np.minimum(ry1, y1)
    iw, ih = (ix1 - ix0).clip(min=0), (iy1 - iy0).clip(min=0)
    area = (rx1 - rx0) * (ry1 - ry0)
    visibility = np.divide(iw * ih, area, out=np.zeros(len(out)), where=area > 0)

    out["visibility"] = np.where(is_type, np.nan, visibility)
    overlaps = ~is_type & (iw > 0) & (ih > 0)
    for col, v in zip(CLIPPED, (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0)):
        out[col] = np.where(overlaps, v, np.nan)
    out["status"] = np.select(
        [is_type, is_duplicate, visibility < cfg.min_visibility, (iw < cfg.min_size_px) | (ih < cfg.min_size_px)],
        STATUSES[1:], STATUSES[0])
    return out

def yolo_lines(kept: pd.DataFrame, crop_w: int, crop_h: int) -> list[str]:
    """`class_id cx cy w h`, normalized by the crop size."""
    cx = (kept["cx0"] + kept["cx1"]) / 2 / crop_w
    cy = (kept["cy0"] + kept["cy1"]) / 2 / crop_h
    w = (kept["cx1"] - kept["cx0"]) / crop_w
    h = (kept["cy1"] - kept["cy0"]) / crop_h
    return [f"{int(c)} " + " ".join(f"{v:.{LABEL_DECIMALS}f}" for v in (a, b, d, e))
            for c, a, b, d, e in zip(kept["class_id"], cx, cy, w, h)]

def read_labels(label_dir: Path) -> pd.DataFrame:
    """All label lines under `label_dir/<split>/` as rows: split, stem, fields (list of str)."""
    rows = []
    for path in sorted(label_dir.glob("*/*.txt")):
        for line in path.read_text().splitlines():
            rows.append({"split": path.parent.name, "stem": path.stem, "fields": line.split()})
    return pd.DataFrame(rows, columns=["split", "stem", "fields"])

def _label_values_valid(labels: pd.DataFrame, n_classes: int) -> dict:
    bad = 0
    for fields in labels["fields"]:
        try:
            c, cx, cy, w, h = int(fields[0]), *(float(v) for v in fields[1:])
            ok = len(fields) == 5 and 0 <= c < n_classes and w > 0 and h > 0 and \
                all(0 <= v <= 1 for v in (cx, cy, w, h)) and \
                all(-EDGE_ROUNDING_TOL <= v <= 1 + EDGE_ROUNDING_TOL
                    for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        except (ValueError, IndexError, TypeError):
            ok = False
        bad += not ok
    return {"passed": not bad, "detail": {"lines": len(labels), "invalid": bad}}

def check_conversion(manifest: pd.DataFrame, annotations: pd.DataFrame, splits: pd.DataFrame, n_source: int,
                     root: Path, n_classes: int) -> dict[str, dict]:
    """Self-consistency of one conversion output under `root`. Each entry: {'passed': bool, 'detail': ...}."""
    expected, got = set(splits["stem"]), set(manifest["stem"])
    duplicated = int(manifest["stem"].duplicated().sum())
    pairing = {}
    for s in SPLITS:
        stems = set(manifest.loc[manifest["split"] == s, "stem"])
        images = {p.stem for p in (root / "images" / s).glob("*.png")}
        labels = {p.stem for p in (root / "labels" / s).glob("*.txt")}
        pairing[s] = {"fovs": len(stems), "images_mismatch": len(images ^ stems),
                      "labels_mismatch": len(labels ^ stems)}
    status_counts = annotations["status"].value_counts()
    unknown = sorted(set(status_counts.index) - set(STATUSES))
    duplicate_ids = int(annotations.duplicated(["stem", "ann_id"]).sum())
    labels = read_labels(root / "labels")
    lines = labels.groupby("stem").size().reindex(manifest["stem"], fill_value=0)
    kept = annotations[annotations["status"] == "kept"].groupby("stem").size().reindex(manifest["stem"], fill_value=0)
    n_labels = manifest.set_index("stem")["n_labels"]
    line_mismatch = int(((lines != kept) | (lines != n_labels)).sum())
    return {
        "every_fov_converted_once": {
            "passed": expected == got and not duplicated,
            "detail": {"missing": len(expected - got), "extra": len(got - expected), "duplicated": duplicated}},
        "image_label_pairing": {
            "passed": not any(v["images_mismatch"] or v["labels_mismatch"] for v in pairing.values()),
            "detail": pairing},
        "annotations_accounted": {
            "passed": len(annotations) == n_source and not unknown and not duplicate_ids,
            "detail": {"source": n_source, "recorded": len(annotations), "unknown_status": unknown,
                       "duplicate_ids": duplicate_ids, **status_counts.reindex(STATUSES, fill_value=0).to_dict()}},
        "label_lines_match_kept": {"passed": not line_mismatch, "detail": line_mismatch},
        "label_values_valid": _label_values_valid(labels, n_classes),
    }

def build(raw: Path, splits: pd.DataFrame, classes: Classes, cfg: Convert, root: Path,
          data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Write images, labels, manifest.csv, annotations.csv and data.yaml under `root`.

    Returns (manifest, annotations, number of source annotations).
    """
    for kind in ("images", "labels"):
        for s in SPLITS:
            (root / kind / s).mkdir(parents=True, exist_ok=True)
    splits = splits.sort_values("stem")
    ann = nd.map_classes(nd.load_annotations(raw, splits["stem"].tolist()), classes)
    by_stem = dict(tuple(ann.groupby("stem")))
    manifest, converted = [], []
    for row in splits.itertuples():
        img = cv2.imread(str(raw / "rgb" / f"{row.stem}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(raw / "mask" / f"{row.stem}.png"), cv2.IMREAD_COLOR)
        if img is None or mask is None:
            raise OSError(f"Cannot read image or mask: {row.stem}")
        h, w = img.shape[:2]
        x0, y0, x1, y1 = rect = crop_rect(nd.fov_rect(nd.mask_codes(mask)), w, h)
        boxes = convert_boxes(by_stem.get(row.stem, ann.iloc[:0]), rect, w, h, classes, cfg)
        lines = yolo_lines(boxes[boxes["status"] == "kept"], x1 - x0, y1 - y0)
        if not cv2.imwrite(str(root / "images" / row.split / f"{row.stem}.png"), img[y0:y1, x0:x1]):
            raise OSError(f"Cannot write image: {row.stem}")
        (root / "labels" / row.split / f"{row.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
        manifest.append({"stem": row.stem, "split": row.split, "width": w, "height": h,
                         "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                         "crop_w": x1 - x0, "crop_h": y1 - y0, "n_labels": len(lines)})
        converted.append(boxes.assign(split=row.split))
    manifest = pd.DataFrame(manifest)
    annotations = pd.concat(converted, ignore_index=True)[
        ["stem", "ann_id", "split", *nd.ANNOTATION_COLUMNS[:3], "project_class", "class_id", "type",
         "xmin", "ymin", "xmax", "ymax", "visibility", *CLIPPED, "status"]]
    manifest.to_csv(root / "manifest.csv", index=False)
    annotations.to_csv(root / "annotations.csv", index=False)
    data = {"path": str(data_path), **{s: f"images/{s}" for s in SPLITS}, "names": dict(enumerate(classes.names))}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return manifest, annotations, len(ann)

def publish(tmp: Path, out: Path) -> None:
    """Promote every output in `tmp` to `out`, or roll back so the previous outputs remain intact.

    Previous outputs are first moved into a backup directory and deleted only after all new outputs are
    in place. If a rollback itself fails, the backup is kept for manual recovery.
    """
    backup = out / BACKUP_DIR
    backup.mkdir()
    old, new = [], []
    try:
        for name in OUTPUTS:
            if (out / name).exists():
                (out / name).rename(backup / name)
                old.append(name)
        for name in OUTPUTS:
            (tmp / name).rename(out / name)
            new.append(name)
    except BaseException:
        for name in reversed(new):
            (out / name).rename(tmp / name)
        for name in reversed(old):
            (backup / name).rename(out / name)
        backup.rmdir()
        raise
    shutil.rmtree(backup)

def summarize(manifest: pd.DataFrame, annotations: pd.DataFrame, checks: dict, cfg: Convert,
              classes: Classes) -> str:
    kept = annotations[annotations["status"] == "kept"]
    dropped = annotations[annotations["status"] != "kept"]
    shifted = kept[["xmin", "ymin", "xmax", "ymax"]].to_numpy() - \
        kept.merge(manifest, on="stem")[["crop_x0", "crop_y0", "crop_x0", "crop_y0"]].to_numpy()
    clipped = (np.abs(kept[CLIPPED].to_numpy() - shifted) > 1e-9).any(axis=1)
    fovs = manifest.groupby("split").agg(fovs=("stem", "size"), empty_labels=("n_labels", lambda n: int((n == 0).sum())),
                                         labels=("n_labels", "sum")).reindex(SPLITS)
    sections = [
        _block("Config", f"min_visibility: {cfg.min_visibility}  min_size_px: {cfg.min_size_px}  "
                         f"edge_tolerance_px: {EDGE_TOLERANCE_PX}"),
        _block("Checks", json.dumps(checks, indent=2)),
        _block("FOVs and label lines per split", fovs),
        _block("Kept boxes: split × class", pd.crosstab(kept["split"], kept["project_class"], margins=True)
               .reindex(index=[*SPLITS, "All"], columns=[*classes.names, "All"], fill_value=0)),
        _block("Status × split", pd.crosstab(annotations["status"], annotations["split"], margins=True)
               .reindex(columns=[*SPLITS, "All"], fill_value=0)),
        _block("Dropped: status × split × class", pd.crosstab([dropped["status"], dropped["split"]],
                                                              dropped["project_class"], margins=True)),
        _block("Kept boxes clipped to the crop or RGB edge", pd.Series(clipped, index=kept.index)
               .groupby(kept["split"]).sum().reindex(SPLITS)),
        _block("Visibility of kept boxes clipped to the crop", kept.loc[kept["visibility"] < 1, "visibility"]
               .describe([0.01, 0.05, 0.25, 0.5]).round(3)),
        _block("Crop size (px)", manifest[["crop_w", "crop_h"]].describe([0.01, 0.5, 0.99]).round(1)),
    ]
    return "# NuCLS → YOLO conversion\n\n" + "\n".join(sections)

def run_conversion(raw: Path, out: Path, report: Path, classes: Classes, cfg: Convert) -> bool:
    splits_path = out / "splits.csv"
    if not splits_path.exists():
        print(f"Missing {splits_path}; run `python -m tvision splits` first.")
        return False
    if (out / BACKUP_DIR).exists():
        raise RuntimeError(f"{out / BACKUP_DIR} exists from an interrupted publish; restore or remove it first.")
    shutil.rmtree(report, ignore_errors=True)
    report.mkdir(parents=True)
    splits = pd.read_csv(splits_path)
    tmp = out / TMP_DIR
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        manifest, annotations, n_source = build(raw, splits, classes, cfg, tmp, out.resolve())
        checks = check_conversion(manifest, annotations, splits, n_source, tmp, len(classes.names))
        (report / "checks.json").write_text(json.dumps(checks, indent=2))
        for name, c in checks.items():
            print(f"[{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")
        (report / "summary.md").write_text(summarize(manifest, annotations, checks, cfg, classes))
        passed = all(c["passed"] for c in checks.values())
        if passed:
            publish(tmp, out)
            print(f"Dataset: {out / 'data.yaml'}  Report: {report / 'summary.md'}")
        else:
            print("Checks failed; previous outputs left unchanged.")
        return passed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def convert() -> bool:
    paths = Paths.load()
    return run_conversion(paths.raw_dir, paths.processed_dir, paths.artifacts_dir / "convert",
                          Classes.load(), Convert.load())