import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tvision import two_class as tc
from tvision.config import Classes, TwoClass

from test_train import _verified

CLASSES = Classes.load()
KEPT = tc.kept_classes(CLASSES, ["ambiguous"])
SPLITS = ["train", "val"]
LABELS = {
    "train": {"a": [(0, 0.5), (2, 0.4), (1, 0.3)], "b": [(2, 0.5)], "c": [(1, 0.2), (1, 0.6)]},
    "val": {"d": [(2, 0.1), (0, 0.2), (0, 0.7)], "e": [(2, 0.3), (2, 0.6)]},
    "test": {"f": [(2, 0.5), (0, 0.5)]},
}

def _source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    rng = np.random.default_rng(0)
    for split, files in LABELS.items():
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for stem, lines in files.items():
            cv2.imwrite(str(src / "images" / split / f"{stem}.png"), rng.integers(0, 255, (20, 30, 3), np.uint8))
            (src / "labels" / split / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.6f} 0.500000 0.100000 0.100000\n" for c, x in lines))
    return src

def _derived(tmp_path: Path) -> tuple[Path, Path]:
    src, out = _source(tmp_path), tmp_path / "out"
    tc.derive(src, out, out, KEPT, SPLITS)
    return src, out

def test_kept_classes_drops_trailing_ambiguous_only():
    assert KEPT.names == ["tumor", "non_tumor"] and KEPT.headline == CLASSES.headline
    assert "ambiguous" not in KEPT.super_to_class.values()

@pytest.mark.parametrize("exclude, match", [(["tumor"], "headline|trailing"), (["mitosis"], "unknown"),
                                            (["non_tumor"], "trailing")])
def test_kept_classes_refuses_bad_exclusions(exclude, match):
    with pytest.raises(ValueError, match=match):
        tc.kept_classes(CLASSES, exclude)

def test_two_class_config_refuses_test_split():
    with pytest.raises(ValueError, match="test"):
        TwoClass(Path("x"), ["ambiguous"], ["train", "test"])

def test_derive_removes_only_ambiguous_lines_in_order(tmp_path):
    src, out = _derived(tmp_path)
    assert (out / "labels" / "train" / "a.txt").read_text().splitlines() == [
        line for line in (src / "labels" / "train" / "a.txt").read_text().splitlines() if not line.startswith("2 ")]
    assert (out / "labels" / "train" / "b.txt").read_text() == ""
    assert not (out / "images" / "test").exists() and not (out / "labels" / "test").exists()
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "tumor", 1: "non_tumor"} and "test" not in data

def test_checks_pass_on_derived_dataset(tmp_path):
    src, out = _derived(tmp_path)
    checks = tc.run_checks(src, out, out, CLASSES, KEPT, SPLITS)
    assert all(c["passed"] for c in checks.values()), {k: c for k, c in checks.items() if not c["passed"]}
    assert checks["empty_label_files"]["detail"] == {"train": {"derived": 1, "expected": 1, "passed": True},
                                                     "val": {"derived": 1, "expected": 1, "passed": True}}
    assert checks["counts_match"]["detail"]["train"]["removed"] == {"ambiguous": 2}
    assert checks["counts_match"]["detail"]["val"]["derived"] == {"tumor": 2, "non_tumor": 0}

def _rewrite(path: Path, fn) -> None:
    path.write_text(fn(path.read_text()))

def _reorder(out: Path) -> None:
    _rewrite(out / "labels" / "val" / "d.txt", lambda t: "".join(f"{l}\n" for l in reversed(t.splitlines())))

def _keep_ambiguous(out: Path) -> None:
    _rewrite(out / "labels" / "train" / "a.txt", lambda t: t + "2 0.400000 0.500000 0.100000 0.100000\n")

def _change_image(out: Path) -> None:
    img = out / "images" / "train" / "a.png"
    cv2.imwrite(str(img), 255 - cv2.imread(str(img)))

def _delete_image(out: Path) -> None:
    (out / "images" / "val" / "d.png").unlink()

def _add_test_split(out: Path) -> None:
    (out / "images" / "test").mkdir()

def _test_in_yaml(out: Path) -> None:
    _rewrite(out / "data.yaml", lambda t: t + "test: images/test\n")

def _fill_expected_empty(out: Path) -> None:
    (out / "labels" / "train" / "b.txt").write_text("0 0.500000 0.500000 0.100000 0.100000\n")

def _empty_kept_file(out: Path) -> None:
    (out / "labels" / "train" / "c.txt").write_text("")

@pytest.mark.parametrize("mutate, check", [
    (_reorder, "labels_derived"),
    (_keep_ambiguous, "class_ids_valid"),
    (_change_image, "images_identical"),
    (_delete_image, "files_complete"),
    (_add_test_split, "no_test_split"),
    (_test_in_yaml, "data_yaml_valid"),
    (_fill_expected_empty, "empty_label_files"),
    (_empty_kept_file, "empty_label_files"),
])
def test_each_mutation_is_caught(tmp_path, mutate, check):
    src, out = _derived(tmp_path)
    mutate(out)
    checks = tc.run_checks(src, out, out, CLASSES, KEPT, SPLITS)
    assert not checks[check]["passed"]

def test_run_two_class_publishes_then_rechecks(tmp_path):
    paths = _verified(tmp_path)
    cfg = TwoClass(tmp_path / "two", ["ambiguous"], SPLITS)
    assert tc.run_two_class(paths, CLASSES, cfg)
    assert not (tmp_path / "two.tmp").exists()
    assert sorted(p.name for p in (cfg.processed_dir / "images").iterdir()) == SPLITS
    derived, source = tc.gate_two_class(paths, cfg)
    assert derived == json.loads((paths.artifacts_dir / "two_class" / "fingerprint.json").read_text())
    assert tc.run_two_class(paths, CLASSES, cfg)
    checks = json.loads((paths.artifacts_dir / "two_class" / "checks.json").read_text())
    assert checks["fingerprint_unchanged"]["passed"]

def test_gate_two_class_refuses_changed_label(tmp_path):
    paths = _verified(tmp_path)
    cfg = TwoClass(tmp_path / "two", ["ambiguous"], SPLITS)
    assert tc.run_two_class(paths, CLASSES, cfg)
    label = next((cfg.processed_dir / "labels" / "train").glob("*.txt"))
    label.write_text(label.read_text() + "0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(RuntimeError, match="fingerprint"):
        tc.gate_two_class(paths, cfg)
    assert not tc.run_two_class(paths, CLASSES, cfg)

def test_gate_two_class_refuses_unbuilt(tmp_path):
    paths = _verified(tmp_path)
    with pytest.raises(RuntimeError, match="not built"):
        tc.gate_two_class(paths, TwoClass(tmp_path / "two", ["ambiguous"], SPLITS))