import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Albumentations, RandomFlip, RandomHSV, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import DetMetrics

from tvision import convert as cv
from tvision import train as tr
from tvision import verify as vf
from tvision.config import AUGMENTATION_KEYS, Classes, Convert, Paths, Train, load_yaml

from conftest import _raw

CLASSES = Classes.load()
CFG = Convert(min_visibility=0.1, min_size_px=2)
TRAIN_YAML = load_yaml("train.yaml")
FLIPS = {"fliplr": 0.5, "flipud": 0.5}

def _verified(tmp_path: Path) -> Paths:
    """Synthetic raw set converted by Step 3 and verified by Step 4, laid out like the real project."""
    raw, out = _raw(tmp_path)
    assert cv.run_conversion(raw, out, tmp_path / "convert_report", CLASSES, CFG)
    paths = Paths(raw, out, tmp_path / "artifacts", tmp_path / "runs")
    assert vf.run_verification(raw, out, paths.artifacts_dir / "verify", CLASSES, CFG, 0, 0)
    return paths

def _cfg(**overrides) -> Train:
    return Train.from_dict(TRAIN_YAML | overrides)

def test_config_is_flips_only_on_mps():
    cfg = Train.load()
    assert cfg.device == "mps"
    assert cfg.augmentation == {k: FLIPS.get(k, 0) for k in AUGMENTATION_KEYS}

def test_ultralytics_args_pass_complete_flips_only_policy():
    args = Train.load().ultralytics_args()
    for k in AUGMENTATION_KEYS:
        assert args[k] == FLIPS.get(k, 0), k
    assert args["augmentations"] == []
    assert {"imgsz": 640, "max_det": 1000, "device": "mps"}.items() <= args.items()
    assert not {"model", "name", "data", "project", "augmentation"} & args.keys()
    get_cfg(overrides=args)

def test_smoke_overrides_merge():
    full, smoke = Train.load(), Train.load(smoke=True)
    assert smoke.name != full.name and smoke.fraction < 1 and smoke.epochs <= full.epochs
    assert smoke.augmentation == full.augmentation

@pytest.mark.parametrize("change", [
    lambda c: c | {"lr0": 0.01},
    lambda c: c | {"smoke": {"lr0": 0.01}},
    lambda c: c | {"smoke": {"augmentation": {}}},
    lambda c: c | {"augmentation": c["augmentation"] | {"erasing": 0.4}},
    lambda c: c | {"augmentation": {k: v for k, v in c["augmentation"].items() if k != "bgr"}},
    lambda c: c | {"augmentation": c["augmentation"] | {"hsv_s": 1.5}},
    lambda c: c | {"augmentation": c["augmentation"] | {"degrees": -1}},
    lambda c: c | {"augmentation": c["augmentation"] | {"fliplr": True}},
    lambda c: c | {"device": None},
    lambda c: c | {"cache": "disk"},
    lambda c: c | {"imgsz": 600},
    lambda c: c | {"epochs": 0},
])
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        Train.from_dict(change(dict(TRAIN_YAML)))

def _train_dataset(paths: Paths, args: dict) -> YOLODataset:
    data = check_det_dataset(str(paths.processed_dir / "data.yaml"), autodownload=False)
    return YOLODataset(img_path=data["train"], data=data, task="detect", augment=True,
                       hyp=get_cfg(overrides=args), imgsz=args["imgsz"])

def test_training_pipeline_has_no_albumentations_or_colour(tmp_path):
    ds = _train_dataset(_verified(tmp_path), Train.load().ultralytics_args())
    transforms = ds.transforms.transforms
    album = next(t for t in transforms if isinstance(t, Albumentations))
    assert album.transform is None or len(album.transform.transforms) == 0
    hsv = next(t for t in transforms if isinstance(t, RandomHSV))
    assert (hsv.hgain, hsv.sgain, hsv.vgain) == (0, 0, 0)
    assert sorted(t.p for t in transforms if isinstance(t, RandomFlip)) == [0.5, 0.5]

def test_flips_off_training_boxes_equal_letterboxed_labels(tmp_path):
    """With flips disabled the remaining pipeline only resizes and centres: boxes must be exact."""
    paths = _verified(tmp_path)
    ds = _train_dataset(paths, Train.load().ultralytics_args() | {"fliplr": 0.0, "flipud": 0.0})
    assert any(isinstance(t, RandomPerspective) for t in ds.transforms.transforms[0].transforms)
    item = ds[0]
    label = paths.processed_dir / "labels" / "train" / f"{Path(item['im_file']).stem}.txt"
    expected = np.array([[float(v) for v in line.split()] for line in label.read_text().splitlines()])
    h0, w0 = item["ori_shape"]
    r = 640 / max(h0, w0)
    w, h = min(np.ceil(w0 * r), 640), min(np.ceil(h0 * r), 640)
    cx = (expected[:, 1] * w + (640 - w) / 2) / 640
    cy = (expected[:, 2] * h + (640 - h) / 2) / 640
    want = np.stack([cx, cy, expected[:, 3] * w / 640, expected[:, 4] * h / 640], 1)
    assert item["img"].shape[1:] == (640, 640)
    np.testing.assert_allclose(item["bboxes"].numpy(), want, atol=1e-4)
    np.testing.assert_array_equal(item["cls"].numpy().ravel(), expected[:, 0])

def test_per_class_val_reads_ultralytics_summary_columns():
    metrics = DetMetrics(names=dict(enumerate(CLASSES.names)))
    metrics.update_stats({"tp": np.ones((3, 10), bool), "conf": np.array([0.9, 0.8, 0.7]),
                          "pred_cls": np.array([0, 1, 2]), "target_cls": np.array([0, 1, 2]),
                          "target_img": np.array([0, 1, 2]), "im_name": "a"})
    metrics.process()
    assert list(metrics.summary()[0]) == ["Class", "Images", "Instances", *tr.METRIC_COLUMNS]
    table, headline = tr.per_class_val(metrics, CLASSES)
    assert list(table.index) == CLASSES.names
    assert table.loc["tumor", "Instances"] == 1
    assert set(headline) == set(tr.METRIC_COLUMNS) and all(np.isfinite(list(headline.values())))
    json.dumps({"per_class": table.reset_index().to_dict("records"), "headline_mean": headline}, allow_nan=False)

def test_gate_passes_on_verified_dataset(tmp_path):
    paths = _verified(tmp_path)
    fp = tr.gate(paths.processed_dir, paths.artifacts_dir / "verify")
    assert fp == json.loads((paths.artifacts_dir / "verify" / "fingerprint.json").read_text())

def test_gate_refuses_changed_label(tmp_path):
    paths = _verified(tmp_path)
    label = next((paths.processed_dir / "labels" / "train").glob("*.txt"))
    label.write_text(label.read_text().replace("0.", "0.0", 1))
    with pytest.raises(RuntimeError, match="fingerprint"):
        tr.gate(paths.processed_dir, paths.artifacts_dir / "verify")

def test_gate_refuses_failed_check(tmp_path):
    paths = _verified(tmp_path)
    checks_file = paths.artifacts_dir / "verify" / "checks.json"
    checks = json.loads(checks_file.read_text())
    checks["labels_match_raw"]["passed"] = False
    checks_file.write_text(json.dumps(checks))
    with pytest.raises(RuntimeError, match="labels_match_raw"):
        tr.gate(paths.processed_dir, paths.artifacts_dir / "verify")

def test_gate_refuses_albumentations(tmp_path, monkeypatch):
    paths = _verified(tmp_path)
    monkeypatch.setattr(tr.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="albumentations"):
        tr.gate(paths.processed_dir, paths.artifacts_dir / "verify")

def test_loader_check_matches_label_files(tmp_path):
    paths = _verified(tmp_path)
    result = tr.loader_check(paths.processed_dir, CLASSES)
    assert all(s["passed"] for s in result.values())
    assert result["train"]["instances"] == result["train"]["expected"] == {"tumor": 1, "non_tumor": 0, "ambiguous": 0}

def test_loader_check_catches_duplicate_line(tmp_path):
    paths = _verified(tmp_path)
    label = paths.processed_dir / "labels" / "val" / next(iter(
        p.name for p in (paths.processed_dir / "labels" / "val").glob("*.txt")))
    first = label.read_text().splitlines()[0]
    label.write_text(label.read_text() + first + "\n")
    result = tr.loader_check(paths.processed_dir, CLASSES)
    assert not result["val"]["passed"] and result["val"]["images_mismatched"] == 1
    assert result["train"]["passed"] and result["test"]["passed"]

def test_check_device_refuses_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        tr.check_device("mps")
    tr.check_device("cpu")

class _NoTraining(Exception):
    pass

def _refuse_yolo(*args, **kwargs):
    raise _NoTraining

@pytest.fixture
def no_training(monkeypatch):
    monkeypatch.setattr(tr, "check_device", lambda device: None)
    monkeypatch.setattr(tr, "YOLO", _refuse_yolo)

def test_full_run_refused_on_dirty_tree(tmp_path, monkeypatch, no_training):
    paths = _verified(tmp_path)
    monkeypatch.setattr(tr, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(RuntimeError, match="dirty"):
        tr.run_training(_cfg(), paths, CLASSES, smoke=False)
    with pytest.raises(_NoTraining):
        tr.run_training(Train.from_dict(TRAIN_YAML, smoke=True), paths, CLASSES, smoke=True)

def test_existing_run_dir_refused(tmp_path, monkeypatch, no_training):
    paths = _verified(tmp_path)
    monkeypatch.setattr(tr, "git_state", lambda: {"commit": "abc", "dirty": False})
    (paths.runs_dir / _cfg().name).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        tr.run_training(_cfg(), paths, CLASSES)

def test_check_only_does_not_train(tmp_path, no_training):
    assert tr.run_training(_cfg(), _verified(tmp_path), CLASSES, check_only=True)