import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from tvision import convert as cv
from tvision import verify as vf
from tvision.config import Classes, Convert

from conftest import H, STEMS, W, _box, _raw

CLASSES = Classes.load()
CFG = Convert(min_visibility=0.1, min_size_px=2)
TRAIN = "TCGA-AA-0001-DX1_id-a"
CROP_W, CROP_H = 80, 60

def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic raw set converted by Step 3; the train FOV has two kept boxes of different classes."""
    raw, out = _raw(tmp_path)
    csv = raw / "csv" / f"{TRAIN}.csv"
    df = pd.read_csv(csv, index_col=0)
    extra = {"raw_classification": "lymphocyte", "main_classification": "lymphocyte",
             "super_classification": "sTIL", "type": "rectangle", **_box(50, 30, 70, 50)}
    pd.concat([df, pd.DataFrame([extra])], ignore_index=True).to_csv(csv)
    assert cv.run_conversion(raw, out, tmp_path / "convert_report", CLASSES, CFG)
    return raw, out

def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

def _label(out: Path, stem: str = TRAIN, split: str = "train") -> Path:
    return out / "labels" / split / f"{stem}.txt"

def _image(out: Path, stem: str = TRAIN, split: str = "train") -> Path:
    return out / "images" / split / f"{stem}.png"

def _edit_lines(out: Path, edit) -> None:
    lines = _label(out).read_text().splitlines()
    _label(out).write_text("".join(f"{l}\n" for l in edit(lines)))

def _shift_1px(lines):
    c, cx, *rest = lines[0].split()
    return [" ".join([c, f"{float(cx) + 1 / CROP_W:.6f}", *rest]), *lines[1:]]

def _swap_class(lines):
    return [("1" if lines[0][0] == "0" else "0") + lines[0][1:], *lines[1:]]

def _alter_pixel(out):
    img = cv2.imread(str(_image(out)))
    img[0, 0] = 255 - img[0, 0]
    cv2.imwrite(str(_image(out)), img)

def _uncropped(raw, out):
    shutil.copy(raw / "rgb" / f"{TRAIN}.png", _image(out))

def _copy_to_val(out):
    shutil.copy(_image(out), _image(out, split="val"))
    shutil.copy(_label(out), _label(out, split="val"))

def _reorder_names(out):
    data = yaml.safe_load((out / "data.yaml").read_text())
    data["names"] = {0: "non_tumor", 1: "tumor", 2: "ambiguous"}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

def _tamper_status(out):
    ann = pd.read_csv(out / "annotations.csv")
    ann.loc[ann["status"] == "dropped_type", "status"] = "kept"
    ann.to_csv(out / "annotations.csv", index=False)

MUTATIONS = {
    "shift_label_1px": (lambda raw, out: _edit_lines(out, _shift_1px), "labels_match_raw"),
    "swap_class": (lambda raw, out: _edit_lines(out, _swap_class), "labels_match_raw"),
    "delete_line": (lambda raw, out: _edit_lines(out, lambda l: l[1:]), "labels_match_raw"),
    "reorder_lines": (lambda raw, out: _edit_lines(out, lambda l: l[::-1]), "labels_match_raw"),
    "duplicate_line": (lambda raw, out: _edit_lines(out, lambda l: [*l, l[0]]), "no_duplicate_label_lines"),
    "invalid_line": (lambda raw, out: _edit_lines(out, lambda l: [*l, "0 0.5 0.5 0.2"]), "label_format"),
    "no_trailing_newline": (lambda raw, out: _label(out).write_text(_label(out).read_text().rstrip("\n")),
                            "label_format"),
    "alter_pixel": (lambda raw, out: _alter_pixel(out), "images_match_raw_crops"),
    "uncropped_image": (_uncropped, "images_match_raw_crops"),
    "stray_file": (lambda raw, out: (out / "images" / "train" / "notes.txt").write_text("x"), "files_complete"),
    "missing_label": (lambda raw, out: _label(out).unlink(), "files_complete"),
    "stem_in_two_splits": (lambda raw, out: _copy_to_val(out), "no_leakage_in_files"),
    "reordered_names": (lambda raw, out: _reorder_names(out), "data_yaml_valid"),
    "tampered_audit_status": (lambda raw, out: _tamper_status(out), "audit_files_agree"),
}

def test_clean_dataset_passes_and_stays_unmodified(tmp_path):
    raw, out = _dataset(tmp_path)
    before_raw, before_out = _snapshot(raw), _snapshot(out)
    report = tmp_path / "report"
    assert vf.run_verification(raw, out, report, CLASSES, CFG, n_overlays=3, seed=0)
    assert _snapshot(raw) == before_raw and _snapshot(out) == before_out
    checks = json.loads((report / "checks.json").read_text())
    assert all(c["passed"] for c in checks.values())
    assert checks["drop_accounting"]["detail"]["kept"] == 4
    assert checks["drop_accounting"]["detail"]["dropped_duplicate"] == 3
    overlays = sorted((report / "overlays").glob("*.png"))
    assert len(overlays) == len(STEMS)
    panel_gap = 6
    expected_w = vf.OVERLAY_SCALE * (W + CROP_W + W) + 2 * panel_gap
    assert all(cv2.imread(str(p)).shape == (vf.OVERLAY_SCALE * H, expected_w, 3) for p in overlays)
    assert json.loads((report / "fingerprint.json").read_text()) == vf.dataset_fingerprint(out)

@pytest.mark.parametrize("name", MUTATIONS)
def test_mutation_caught_by_intended_check(tmp_path, name):
    raw, out = _dataset(tmp_path)
    mutate, check = MUTATIONS[name]
    mutate(raw, out)
    checks, _, _ = vf.run_checks(raw, out, CLASSES, CFG)
    assert not checks[check]["passed"]

def test_label_match_reports_error_size(tmp_path):
    raw, out = _dataset(tmp_path)
    checks, _, _ = vf.run_checks(raw, out, CLASSES, CFG)
    assert checks["labels_match_raw"]["detail"]["max_err_px"] < 1e-3
    _edit_lines(out, _shift_1px)
    detail = vf.run_checks(raw, out, CLASSES, CFG)[0]["labels_match_raw"]["detail"]
    assert detail["coord_mismatch"] == 1 and detail["max_err_px"] == pytest.approx(1, abs=1e-3)

def test_drop_accounting_counts_every_raw_row(tmp_path):
    raw, out = _dataset(tmp_path)
    extra = raw / "csv" / "TCGA-DD-0004-DX1_id-d.csv"
    shutil.copy(raw / "csv" / f"{TRAIN}.csv", extra)
    detail = vf.run_checks(raw, out, CLASSES, CFG)[0]["drop_accounting"]
    assert not detail["passed"] and detail["detail"]["source"] > detail["detail"]["recomputed"]

def test_fingerprint_tracks_dataset_bytes_but_not_ultralytics_cache(tmp_path):
    _, out = _dataset(tmp_path)
    first = vf.dataset_fingerprint(out)
    assert first["images"]["files"] == first["labels"]["files"] == len(STEMS)
    (out / "labels" / "train.cache").write_bytes(b"cache")
    assert vf.dataset_fingerprint(out) == first
    _label(out).write_text(_label(out).read_text().replace("0.", "1.", 1))
    changed = vf.dataset_fingerprint(out)
    assert changed["labels"] != first["labels"] and changed["dataset"] != first["dataset"]
    assert changed["images"] == first["images"]

def test_ultralytics_cache_is_not_stray(tmp_path):
    raw, out = _dataset(tmp_path)
    (out / "labels" / "train.cache").write_bytes(b"cache")
    assert vf.run_checks(raw, out, CLASSES, CFG)[0]["files_complete"]["passed"]

def test_independent_fov_rect_reads_red_channel(tmp_path):
    mask = np.zeros((10, 12, 3), np.uint8)
    mask[2:8, 3:11, 2] = 253
    mask[:, :, 0] = 253
    cv2.imwrite(str(tmp_path / "m.png"), mask)
    assert vf.independent_fov_rect(tmp_path / "m.png", 12, 10) == ((3, 2, 11, 8), (10, 12))
    assert vf.independent_fov_rect(tmp_path / "m.png", 10, 7)[0] == (3, 2, 10, 7)

@pytest.mark.parametrize("line,ok", [("0 0.5 0.5 0.2 0.2", True), ("2 1.0 1.0 0.1 0.1", False),
                                     ("0 0.942547 0.5 0.114907 0.2", True), ("0 0.95 0.5 0.1001 0.2", False),
                                     ("3 0.5 0.5 0.2 0.2", False), ("0 0.5 0.5 0 0.2", False),
                                     ("0 nan 0.5 0.2 0.2", False), ("0.0 0.5 0.5 0.2 0.2", False),
                                     ("0  0.5 0.5 0.2 0.2", False), ("", False), ("0 0.5 0.5 0.2", False)])
def test_parse_line(line, ok):
    assert (vf.parse_line(line, 3) is not None) == ok

def test_expected_fov_rules(tmp_path):
    rows = [{"super_classification": "tumor_any", "type": "rectangle", **_box(20, 20, 40, 40)},
            {"super_classification": "tumor_any", "type": "polyline", **_box(20, 20, 40, 40)},
            {"super_classification": "sTIL", "type": "rectangle", **_box(20, 20, 40, 40)},
            {"super_classification": "sTIL", "type": "point", **_box(5, 5, 5, 5)},
            {"super_classification": "sTIL", "type": "rectangle", **_box(0, 20, 10.5, 40)},
            {"super_classification": "sTIL", "type": "rectangle", **_box(15, 20, 16.5, 40)},
            {"super_classification": "AMBIGUOUS", "type": "rectangle", **_box(80, 60, W + 2, H + 2)}]
    pd.DataFrame(rows).to_csv(tmp_path / "a.csv")
    exp = vf.expected_fov(tmp_path / "a.csv", (10, 10, 90, 70), W, H, CLASSES, CFG)
    assert exp["status"].tolist() == ["kept", "dropped_duplicate", "kept", "dropped_type",
                                      "dropped_visibility", "dropped_size", "kept"]
    assert exp.loc[0, vf.BOX].tolist() == [10, 10, 30, 30]
    assert exp.loc[6, vf.BOX].tolist() == [70, 50, 80, 60]
    pd.DataFrame([{"super_classification": "tumor_any", "type": "rectangle", **_box(-1, 20, 40, 40)}]).to_csv(
        tmp_path / "b.csv")
    with pytest.raises(ValueError, match="edge rule"):
        vf.expected_fov(tmp_path / "b.csv", (10, 10, 90, 70), W, H, CLASSES, CFG)

def test_verify_does_not_import_convert():
    tree = ast.parse(Path(vf.__file__).read_text())
    modules = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    modules |= {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not any("convert" in m.split(".")[1:2] or m.endswith(".convert") for m in modules if m.startswith("tvision"))