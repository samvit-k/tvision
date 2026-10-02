from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from tvision import convert as cv
from tvision import nucls_data as nd
from tvision.config import Classes, Convert
from tvision.explore import EDGE_TOLERANCE_PX

CLASSES = Classes.load()
CFG = Convert(min_visibility=0.1, min_size_px=2)
W, H = 100, 80
RECT = (10, 10, 90, 70)

def _ann(*boxes: dict) -> pd.DataFrame:
    base = {"type": "rectangle", "super_classification": "tumor_any"}
    df = pd.DataFrame([{**base, **b} for b in boxes]).rename_axis("ann_id").reset_index()
    return nd.map_classes(df, CLASSES)

def _box(x0, y0, x1, y1, **kw) -> dict:
    return {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, **kw}

def _convert(*boxes: dict, rect=RECT) -> pd.DataFrame:
    return cv.convert_boxes(_ann(*boxes), rect, W, H, CLASSES, CFG)

def test_convert_config_validation():
    assert Convert.load() == CFG
    with pytest.raises(ValueError, match="min_visibility"):
        Convert(min_visibility=1.5, min_size_px=2)
    with pytest.raises(ValueError, match="min_size_px"):
        Convert(min_visibility=0.1, min_size_px=0)

def test_crop_rect_clipped_to_image():
    assert cv.crop_rect((5, 6, W + 2, H + 1), W, H) == (5, 6, W, H)
    with pytest.raises(ValueError, match="missing"):
        cv.crop_rect(None, W, H)
    with pytest.raises(ValueError, match="Empty"):
        cv.crop_rect((W, 0, W + 2, H), W, H)

def test_inside_box_only_shifted():
    r = _convert(_box(20, 30, 40, 50)).iloc[0]
    assert r.status == "kept" and r.visibility == 1
    assert (r.cx0, r.cy0, r.cx1, r.cy1) == (10, 20, 30, 40)

def test_edge_box_clipped_with_visibility():
    r = _convert(_box(0, 20, 20, 40)).iloc[0]  # half inside the crop horizontally
    assert r.status == "kept" and r.visibility == pytest.approx(0.5)
    assert (r.cx0, r.cy0, r.cx1, r.cy1) == (0, 10, 10, 30)

def test_visibility_threshold_inclusive():
    at = _convert(_box(0, 20, 20, 30), rect=(18, 10, 90, 70)).iloc[0]  # exactly 2/20 = 0.1
    assert at.visibility == 0.1 and at.status == "kept"
    under = _convert(_box(0, 20, 20, 30), rect=(18.1, 10, 90, 70)).iloc[0]  # visibility checked before size
    assert under.visibility < 0.1 and under.status == "dropped_visibility"

def test_sliver_dropped_for_size():
    r = _convert(_box(15, 20, 16.5, 40)).iloc[0]
    assert r.visibility == 1 and r.status == "dropped_size"

def test_point_dropped_first_even_outside_fov():
    out = _convert(_box(5, 5, 5, 5, type="point"), _box(95, 75, 95, 75, type="point"))
    assert out.status.tolist() == ["dropped_type", "dropped_type"]
    assert out.visibility.isna().all() and out[cv.CLIPPED].isna().all().all()

def test_fully_outside_box_dropped_for_visibility():
    r = _convert(_box(0, 0, 8, 8)).iloc[0]
    assert r.visibility == 0 and r.status == "dropped_visibility" and np.isnan(r.cx0)

@pytest.mark.parametrize("box", [_box(20, 20, 20, 30), _box(20, 30, 30, 30), _box(30, 20, 20, 30)])
def test_degenerate_non_point_box_raises(box):
    with pytest.raises(ValueError, match="Degenerate"):
        _convert(box)

@pytest.mark.parametrize("box", [_box(-1, 20, 30, 30), _box(20, -1, 30, 30),
                                 _box(20, 20, W + EDGE_TOLERANCE_PX + 1, 30),
                                 _box(20, 20, 30, H + EDGE_TOLERANCE_PX + 1)])
def test_rgb_edge_rule_raises(box):
    with pytest.raises(ValueError, match="edge tolerance"):
        _convert(box)

def test_rgb_overshoot_clipped_before_visibility():
    r = _convert(_box(80, 60, W + EDGE_TOLERANCE_PX, H + EDGE_TOLERANCE_PX), rect=(0, 0, W, H)).iloc[0]
    assert r.status == "kept" and r.visibility == 1
    assert (r.cx1, r.cy1) == (W, H)

def test_exact_duplicate_dropped_keeping_lowest_ann_id():
    out = _convert(_box(20, 20, 30, 30), _box(40, 40, 50, 50), _box(20, 20, 30, 30))
    assert out.status.tolist() == ["kept", "kept", "dropped_duplicate"]

def test_duplicate_requires_same_class_and_box():
    out = _convert(_box(20, 20, 30, 30), _box(20, 20, 30, 30, super_classification="sTIL"),
                   _box(20, 20, 30, 31), _box(20, 20, 20, 20, type="point"), _box(20, 20, 20, 20, type="point"))
    assert out.status.tolist() == ["kept", "kept", "kept", "dropped_type", "dropped_type"]

def test_duplicate_checked_before_visibility():
    out = _convert(_box(0, 0, 8, 8), _box(0, 0, 8, 8))
    assert out.status.tolist() == ["dropped_visibility", "dropped_duplicate"]

def test_rows_ordered_by_ann_id():
    ann = _ann(_box(20, 20, 30, 30), _box(40, 40, 50, 50)).iloc[::-1]
    assert cv.convert_boxes(ann, RECT, W, H, CLASSES, CFG)["ann_id"].tolist() == [0, 1]

def test_yolo_round_trip_and_format():
    out = _convert(_box(0, 20, 20, 40, super_classification="sTIL"), _box(33, 41, 57, 69))
    lines = cv.yolo_lines(out, 80, 60)
    assert lines[0] == "1 0.062500 0.333333 0.125000 0.333333"
    for line, r in zip(lines, out.itertuples()):
        c, x, y, w, h = line.split()
        x, y, w, h = float(x) * 80, float(y) * 60, float(w) * 80, float(h) * 60
        assert int(c) == r.class_id
        assert (x - w / 2, y - h / 2, x + w / 2, y + h / 2) == pytest.approx((r.cx0, r.cy0, r.cx1, r.cy1), abs=1e-4)

def test_label_edge_rounding_tolerated_but_real_overshoot_fails():
    labels = pd.DataFrame({"fields": [["0", "0.942547", "0.5", "0.114907", "0.2"],  # right edge 1.0000005
                                      ["0", "0.5", "0.5", "0.2", "0.2"],
                                      ["0", "0.95", "0.5", "0.1001", "0.2"],  # right edge 1.00005
                                      ["0", "0.5", "0.5", "0", "0.2"],
                                      ["3", "0.5", "0.5", "0.2", "0.2"]]})
    assert cv._label_values_valid(labels, 3)["detail"] == {"lines": 5, "invalid": 3}

def test_no_kept_boxes_gives_no_lines():
    assert cv.yolo_lines(_convert(_box(0, 0, 8, 8)).query("status == 'kept'"), 80, 60) == []

def test_mask_codes_reads_red_channel_of_bgr():
    mask = np.zeros((10, 12, 3), np.uint8)
    mask[2:8, 3:11, 2] = nd.FOV_CODE  # red in OpenCV's BGR order
    mask[0:10, 0:12, 0] = nd.FOV_CODE  # blue must be ignored
    assert nd.fov_rect(nd.mask_codes(mask)) == (3, 2, 11, 8)
    with pytest.raises(ValueError, match="3-channel"):
        nd.mask_codes(mask[..., 2])

def test_mask_codes_after_png_round_trip(tmp_path):
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[2:8, 3:11, 0] = nd.FOV_CODE  # red in RGB order, as NuCLS stores it
    cv2.imwrite(str(tmp_path / "m.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert nd.fov_rect(nd.mask_codes(cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_COLOR))) == (3, 2, 11, 8)

STEMS = {"TCGA-AA-0001-DX1_id-a": "train", "TCGA-BB-0002-DX1_id-b": "val", "TCGA-CC-0003-DX1_id-c": "test"}

def _raw(root: Path) -> tuple[Path, Path]:
    raw, out = root / "raw", root / "out"
    for d in ("rgb", "mask", "csv"):
        (raw / d).mkdir(parents=True)
    out.mkdir()
    rng = np.random.default_rng(0)
    for i, stem in enumerate(STEMS):
        cv2.imwrite(str(raw / "rgb" / f"{stem}.png"), rng.integers(0, 255, (H, W, 3), np.uint8))
        mask = np.zeros((H, W, 3), np.uint8)
        mask[10:70, 10:90, 2] = nd.FOV_CODE
        cv2.imwrite(str(raw / "mask" / f"{stem}.png"), mask)
        rows = [{"raw_classification": "tumor", "main_classification": "tumor_nonMitotic",
                 "super_classification": "tumor_any", "type": "rectangle", **_box(20, 20, 40, 40 + i)},
                {"raw_classification": "tumor", "main_classification": "tumor_nonMitotic",
                 "super_classification": "tumor_any", "type": "polyline", **_box(20, 20, 40, 40 + i)},
                {"raw_classification": "lymphocyte", "main_classification": "lymphocyte",
                 "super_classification": "sTIL", "type": "point", **_box(5, 5, 5, 5)},
                {"raw_classification": "lymphocyte", "main_classification": "lymphocyte",
                 "super_classification": "sTIL", "type": "polyline", **_box(0, 0, 8, 8)}]
        pd.DataFrame(rows).to_csv(raw / "csv" / f"{stem}.csv")
    pd.DataFrame({"stem": list(STEMS), "split": list(STEMS.values())}).to_csv(out / "splits.csv", index=False)
    return raw, out

def _snapshot(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}

def test_end_to_end_and_rerun_identical(tmp_path):
    raw, out = _raw(tmp_path)
    assert cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    first = _snapshot(out)
    assert cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    assert _snapshot(out) == first
    assert not (out / cv.TMP_DIR).exists() and not (out / cv.BACKUP_DIR).exists()
    ann = pd.read_csv(out / "annotations.csv")
    assert ann["status"].value_counts().to_dict() == {"kept": 3, "dropped_type": 3, "dropped_duplicate": 3,
                                                      "dropped_visibility": 3}
    stem = "TCGA-CC-0003-DX1_id-c"
    assert (out / "labels" / "test" / f"{stem}.txt").read_text() == "0 0.250000 0.350000 0.250000 0.366667\n"
    assert cv2.imread(str(out / "images" / "test" / f"{stem}.png")).shape == (60, 80, 3)

def _first_run(tmp_path) -> tuple[Path, Path, dict[str, bytes]]:
    raw, out = _raw(tmp_path)
    assert cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    pd.DataFrame([{"raw_classification": "tumor", "main_classification": "tumor_nonMitotic",
                   "super_classification": "tumor_any", "type": "rectangle", **_box(30, 30, 50, 50)}]).to_csv(
        raw / "csv" / "TCGA-AA-0001-DX1_id-a.csv")
    return raw, out, _snapshot(out)

def test_failed_checks_keep_previous_outputs(tmp_path, monkeypatch):
    raw, out, before = _first_run(tmp_path)
    monkeypatch.setattr(cv, "check_conversion", lambda *a: {"forced": {"passed": False, "detail": 0}})
    assert not cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    assert _snapshot(out) == before and not (out / cv.TMP_DIR).exists()

def test_build_exception_keeps_previous_outputs(tmp_path, monkeypatch):
    raw, out, before = _first_run(tmp_path)
    def boom(*a):
        raise ValueError("boom")
    monkeypatch.setattr(cv, "convert_boxes", boom)
    with pytest.raises(ValueError, match="boom"):
        cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    assert _snapshot(out) == before and not (out / cv.TMP_DIR).exists()

def test_publish_failure_rolls_back(tmp_path, monkeypatch):
    raw, out, before = _first_run(tmp_path)
    real_rename = Path.rename
    def failing_rename(self, target):
        if self.parent.name == cv.TMP_DIR and self.name == "annotations.csv":
            raise OSError("simulated publish failure")
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated"):
        cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    assert _snapshot(out) == before
    assert not (out / cv.TMP_DIR).exists() and not (out / cv.BACKUP_DIR).exists()

def test_leftover_backup_blocks_run(tmp_path):
    raw, out = _raw(tmp_path)
    (out / cv.BACKUP_DIR).mkdir()
    with pytest.raises(RuntimeError, match="interrupted publish"):
        cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)

def test_check_conversion_detects_failures(tmp_path):
    raw, out = _raw(tmp_path)
    assert cv.run_conversion(raw, out, tmp_path / "report", CLASSES, CFG)
    manifest, ann = pd.read_csv(out / "manifest.csv"), pd.read_csv(out / "annotations.csv")
    splits = pd.read_csv(out / "splits.csv")
    args = (manifest, ann, splits, len(ann), out, len(CLASSES.names))
    assert all(c["passed"] for c in cv.check_conversion(*args).values())

    assert not cv.check_conversion(manifest, ann, splits, len(ann) + 1, out, 3)["annotations_accounted"]["passed"]
    label = out / "labels" / "train" / "TCGA-AA-0001-DX1_id-a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 1.2 0.2\n")
    checks = cv.check_conversion(*args)
    assert not checks["label_lines_match_kept"]["passed"]
    assert checks["label_values_valid"]["detail"]["invalid"] == 1
    label.unlink()
    assert not cv.check_conversion(*args)["image_label_pairing"]["passed"]