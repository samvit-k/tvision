import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from ultralytics.engine.results import Results

from tvision import __main__ as cli
from tvision import evaluate as evl
from tvision import inference as inf
from tvision import inference_check as ic
from tvision.config import PROJECT_ROOT, Classes, Inference, Paths, load_yaml
from tvision.contract import Box, Detection, DetectionResult, ImageInfo, ModelInfo
from tvision.draw import CLASS_COLORS

from test_evaluate import _fake_frozen, _images

CLASSES = Classes.load()
NAMES = CLASSES.names
CFG = load_yaml("inference.yaml")
REAL = Inference.load()
PROCESSED = Paths.load().processed_dir
VAL_DIR = PROCESSED / "images" / "val"
VAL_IMAGES = sorted(VAL_DIR.glob("*.png"))
needs_checkpoint = pytest.mark.skipif(not REAL.weights.is_file(), reason="no selected checkpoint")

def _info(**kw) -> ModelInfo:
    return ModelInfo(**{"name": "stub", "weights_sha256": "0" * 64, "imgsz": 640, "max_det": 1000, "iou": 0.7,
                        "classes": NAMES, "headline_classes": CLASSES.headline, "default_threshold": 0.24} | kw)

def _det(cid: int = 0, conf: float = 0.9, box=(10, 10, 40, 40)) -> Detection:
    return Detection(class_id=cid, class_name=NAMES[cid], confidence=conf, box=Box(x1=box[0], y1=box[1], x2=box[2],
                                                                                  y2=box[3]))

def _result(dets=None, counts=None, w: int = 300, h: int = 200, threshold: float = 0.24) -> DetectionResult:
    dets = [_det()] if dets is None else dets
    counts = counts or {n: sum(d.class_name == n for d in dets) for n in NAMES}
    return DetectionResult(image=ImageInfo(width=w, height=h, scale=640 / max(w, h)), model=_info(),
                           threshold=threshold, detections=dets, counts=counts)

def _stub_result(w: int = 300, h: int = 200) -> DetectionResult:
    info = ModelInfo(name="stub", weights_sha256="0" * 64, imgsz=640, max_det=1000, iou=0.7,
                     classes=CLASSES.names, headline_classes=CLASSES.headline, default_threshold=0.24)
    det = Detection(class_id=0, class_name="tumor", confidence=0.9, box=Box(x1=10, y1=10, x2=40, y2=40))
    return DetectionResult(image=ImageInfo(width=w, height=h, scale=640 / w), model=info, threshold=0.24,
                           detections=[det], counts={"tumor": 1, "non_tumor": 0, "ambiguous": 0})

def _encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()

def _rgb(h: int = 200, w: int = 300) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), np.uint8)

class FakeYOLO:
    names = dict(enumerate(NAMES))
    end2end = False

    def __init__(self, weights, task=None):
        self.model = SimpleNamespace(end2end=self.end2end)

    def predict(self, img, **kwargs):
        return [Results(img, path="", names=self.names, boxes=torch.zeros((0, 6)))]

def _cfg(tmp_path, data: bytes = b"weights", **kw) -> Inference:
    weights = tmp_path / "best.pt"
    weights.write_bytes(data)
    return replace(REAL, **{"weights": weights, "sha256": hashlib.sha256(data).hexdigest(), "device": "cpu"} | kw)

# Config

def test_inference_config_loads_and_validates():
    assert REAL.weights.is_absolute() and REAL.threshold == 0.24 and REAL.imgsz == 640 and REAL.max_det == 1000
    for bad in ({"sha256": "abc"}, {"device": "tpu"}, {"imgsz": 650}, {"max_det": 0}, {"iou": 1.0},
                {"threshold": 0.005}, {"threshold": 1.0}, {"min_threshold": 0.3}, {"min_long_side": 0},
                {"max_long_side": 100}):
        with pytest.raises(ValueError):
            Inference.from_dict(CFG | bad)
    with pytest.raises(ValueError, match="exactly"):
        Inference.from_dict(CFG | {"extra": 1})
    with pytest.raises(ValueError, match="exactly"):
        Inference.from_dict({k: v for k, v in CFG.items() if k != "iou"})

@pytest.mark.skipif(not (PROCESSED / "splits.csv").is_file(), reason="no processed dataset")
def test_val_images_resolve_to_the_processed_val_pngs():
    assert VAL_DIR == PROJECT_ROOT / "data" / "processed" / "nucls_yolo" / "images" / "val"
    splits = pd.read_csv(PROCESSED / "splits.csv")
    expected = sorted(splits.loc[splits["split"] == "val", "stem"])
    assert [p.stem for p in VAL_IMAGES] == expected and expected
    assert all(p.suffix == ".png" and p.parent == VAL_DIR for p in VAL_IMAGES)

# Contract

def test_contract_round_trips_through_json():
    r = _result([_det(0, 0.9), _det(1, 0.5, (50, 50, 80, 90)), _det(2, 0.3, (0, 0, 300, 200))])
    assert DetectionResult.model_validate_json(r.model_dump_json()) == r
    assert r.counts == {"tumor": 1, "non_tumor": 1, "ambiguous": 1}

@pytest.mark.parametrize("make", [
    lambda: Box(x1=5, y1=5, x2=5, y2=9),
    lambda: Box(x1=-1, y1=5, x2=5, y2=9),
    lambda: Detection(**_det().model_dump(), extra=1),
    lambda: _det(conf=0.0),
    lambda: _det(conf=1.1),
    lambda: _result([Detection(class_id=1, class_name="tumor", confidence=0.9, box=Box(x1=0, y1=0, x2=1, y2=1))]),
    lambda: _result(counts={"tumor": 2, "non_tumor": 0, "ambiguous": 0}),
    lambda: _result(counts={"non_tumor": 0, "tumor": 1, "ambiguous": 0}),
    lambda: _result([_det(0, 0.5), _det(0, 0.9)]),
    lambda: _result([_det(conf=0.2)]),
    lambda: _result([_det(box=(10, 10, 400, 40))]),
    lambda: _info(headline_classes=["mitosis"]),
])
def test_contract_rejects_invalid(make):
    with pytest.raises(ValidationError):
        make()

def test_contract_is_frozen():
    with pytest.raises(ValidationError):
        _result().threshold = 0.5

# load_image

def test_load_image_accepts_encoded_files_and_arrays(tmp_path):
    rgb = _rgb()
    assert np.array_equal(inf.load_image(_encode(rgb), 192, 1024), rgb)
    assert inf.load_image(_encode(rgb, ".jpg"), 192, 1024).shape == (200, 300, 3)
    rgba = np.dstack([rgb, np.full((200, 300), 128, np.uint8)])
    assert np.array_equal(inf.load_image(_encode(rgba), 192, 1024), rgb)
    grey = inf.load_image(_encode(rgb[..., 0]), 192, 1024)
    assert grey.shape == (200, 300, 3) and all(np.array_equal(grey[..., c], rgb[..., 0]) for c in range(3))
    path = tmp_path / "fov.png"
    path.write_bytes(_encode(rgb))
    assert np.array_equal(inf.load_image(path, 192, 1024), rgb)
    assert np.array_equal(inf.load_image(str(path), 192, 1024), rgb)
    assert inf.load_image(rgb, 192, 1024) is rgb

@pytest.mark.parametrize("source, match", [
    (b"not an image", "decoded"),
    (b"", "decoded"),
    (_rgb().astype(np.uint16), "8-bit 3-channel"),
    (_rgb()[..., 0], "8-bit 3-channel"),
    (np.dstack([_rgb(), _rgb()[..., :1]]), "8-bit 3-channel"),
    (_encode(_rgb(100, 150)), "long side"),
    (_encode(_rgb(500, 1100)), "long side"),
])
def test_load_image_rejects(source, match):
    with pytest.raises(inf.InputError, match=match):
        inf.load_image(source, 192, 1024)

def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(inf.InputError, match="no such file"):
        inf.load_image(tmp_path / "missing.png", 192, 1024)

# Predictor

def test_predictor_refuses_missing_or_altered_weights(tmp_path):
    with pytest.raises(inf.ModelError, match="not found"):
        inf.Predictor(replace(REAL, weights=tmp_path / "missing.pt"), CLASSES)
    with pytest.raises(inf.ModelError, match="SHA-256"):
        inf.Predictor(replace(_cfg(tmp_path), sha256="0" * 64), CLASSES)

def test_predictor_refuses_unavailable_device(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(inf, "YOLO", lambda *a, **k: pytest.fail("model must not load"))
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        inf.Predictor(_cfg(tmp_path, device="mps"), CLASSES)

def test_predictor_refuses_wrong_classes_and_end2end(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "YOLO", type("Coco", (FakeYOLO,), {"names": {0: "person", 1: "car"}}))
    with pytest.raises(inf.ModelError, match="classes"):
        inf.Predictor(_cfg(tmp_path), CLASSES)
    monkeypatch.setattr(inf, "YOLO", type("E2E", (FakeYOLO,), {"end2end": True}))
    with pytest.raises(inf.ModelError, match="end-to-end"):
        inf.Predictor(_cfg(tmp_path), CLASSES)

def test_predictor_info_and_conf_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "YOLO", FakeYOLO)
    cfg = _cfg(tmp_path)
    p = inf.Predictor(cfg, CLASSES)
    assert p.info.weights_sha256 == cfg.sha256 and p.info.classes == NAMES and p.info.default_threshold == 0.24
    for conf in (0.005, 1.0):
        with pytest.raises(inf.InputError, match="conf"):
            p.predict(_rgb(), conf=conf)
    r = p.predict(_rgb(), conf=0.5)
    assert r.threshold == 0.5 and r.detections == [] and r.counts == dict.fromkeys(NAMES, 0)
    assert r.image == ImageInfo(width=300, height=200, scale=640 / 300)

def test_to_result_sorts_clips_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "YOLO", FakeYOLO)
    p = inf.Predictor(_cfg(tmp_path), CLASSES)
    xyxy = np.array([[10, 10, 40, 40], [280, 150, 310, 210], [0, 0, 5, 5]], np.float32)
    r = p.to_result(xyxy, np.array([0.3, 0.9, 0.5], np.float32), np.array([0.0, 1.0, 2.0]), (200, 300), 0.24)
    assert [d.class_name for d in r.detections] == ["non_tumor", "ambiguous", "tumor"]
    assert r.detections[0].box == Box(x1=280, y1=150, x2=300, y2=200)
    assert r.counts == {"tumor": 1, "non_tumor": 1, "ambiguous": 1}

def test_predict_passes_explicit_square_letterbox_and_config(tmp_path):
    calls = {}

    class StubModel:
        def predict(self, img, **kwargs):
            calls.update(kwargs)
            return [Results(img, path="", names=dict(enumerate(CLASSES.names)), boxes=torch.zeros((0, 6)))]

    cfg = Inference.load()
    predictor = object.__new__(inf.Predictor)  # skip checkpoint loading
    predictor.cfg, predictor.model = cfg, StubModel()
    predictor.info = _stub_result().model
    predictor.predict(np.zeros((300, 450, 3), np.uint8))
    assert calls["rect"] is False
    assert {k: calls[k] for k in ("imgsz", "conf", "iou", "max_det", "device")} == {
        "imgsz": cfg.imgsz, "conf": cfg.threshold, "iou": cfg.iou, "max_det": cfg.max_det, "device": cfg.device}

@needs_checkpoint
def test_real_predict_letterboxes_to_a_square(tmp_path):
    predictor = inf.Predictor(Inference.load(), CLASSES, device="cpu")
    img = np.full((300, 450, 3), 200, np.uint8)
    predictor.predict(img)
    assert tuple(predictor.model.predictor.preprocess([img]).shape) == (1, 3, 640, 640)

@needs_checkpoint
@pytest.mark.skipif(not VAL_IMAGES, reason="no processed val images")
def test_real_predict_is_valid_deterministic_and_thresholds_nest():
    predictor = inf.Predictor(REAL, CLASSES, device="cpu")
    img = predictor.load(VAL_IMAGES[0])
    r = predictor.predict(img)
    assert r.detections and DetectionResult.model_validate_json(r.model_dump_json()) == r
    assert predictor.predict(img) == r
    low = predictor.predict(img, conf=0.1)
    kept = [d for d in low.detections if d.confidence > r.threshold]
    assert [(d.class_id, d.confidence, d.box) for d in kept] == [(d.class_id, d.confidence, d.box)
                                                                 for d in r.detections]

def test_annotate_draws_class_colours_without_modifying_input():
    img = np.zeros((200, 300, 3), np.uint8)
    out = inf.annotate(img, _result([_det(0, 0.9, (10, 10, 40, 40))]), labels=False)
    assert tuple(out[25, 10]) == CLASS_COLORS["tumor"] and tuple(out[25, 25]) == (0, 0, 0)
    assert not img.any()
    assert inf.annotate(img, _result(), labels=True).any()

# CLI

class StubPredictor:
    def __init__(self, cfg, classes, device=None):
        pass

    def load(self, source):
        return np.zeros((200, 300, 3), np.uint8)

    def predict(self, img, conf=None):
        return _stub_result()

def test_cli_fails_when_annotated_png_cannot_be_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inf, "Predictor", StubPredictor)
    monkeypatch.setattr(inf.cv2, "imwrite", lambda path, img: False)
    image = tmp_path / "fov.png"
    image.write_bytes(b"unused by the stub")
    with pytest.raises(RuntimeError, match="failed to write annotated image"):
        inf.predict_images([image], out=tmp_path / "out", annotate_images=True)
    assert "Wrote" not in capsys.readouterr().out
    assert not (tmp_path / "out" / "fov.json").exists()

def test_cli_writes_outputs_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inf, "Predictor", StubPredictor)
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    out = tmp_path / "out"
    assert inf.predict_images(images, out=out, annotate_images=True) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a.png", "b.json", "b.png"]
    assert DetectionResult.model_validate_json((out / "a.json").read_text()) == _stub_result()
    assert inf.predict_images(images[:1], out=out) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert inf.predict_images([tmp_path / "x" / "a.png", tmp_path / "a.png"], out=tmp_path / "o2") == 1
    assert "distinct" in capsys.readouterr().err

def test_cli_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(inf, "Predictor", StubPredictor)
    assert inf.predict_images([tmp_path / "a.png"]) == 0
    assert DetectionResult.model_validate(json.loads(capsys.readouterr().out)) == _stub_result()
    assert inf.predict_images([tmp_path / "a.png", tmp_path / "b.png"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2

def test_cli_bad_image_exits_1_without_outputs(tmp_path, monkeypatch, capsys):
    class Validating(StubPredictor):
        def load(self, source):
            return inf.load_image(source, 192, 1024)

    monkeypatch.setattr(inf, "Predictor", Validating)
    good, bad = tmp_path / "good.png", tmp_path / "bad.png"
    good.write_bytes(_encode(_rgb()))
    bad.write_bytes(b"broken")
    assert inf.predict_images([good, bad], out=tmp_path / "out") == 1
    assert "bad.png" in capsys.readouterr().err and not (tmp_path / "out").exists()

def test_cli_refused_model_exits_1(tmp_path, monkeypatch, capsys):
    def refuse(*a, **k):
        raise inf.ModelError("SHA-256 mismatch")
    monkeypatch.setattr(inf, "Predictor", refuse)
    assert inf.predict_images([tmp_path / "a.png"]) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err

@pytest.mark.parametrize("argv", [["predict"], ["predict", "--check", "a.png"], ["predict", "a.png", "--annotate"]])
def test_cli_argument_validation(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["tvision", *argv])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2

# Parity check

def test_metric_parity_is_zero_against_itself_and_detects_differences():
    records, gt = _images()
    ids = [CLASSES.class_id(n) for n in CLASSES.headline]
    prf = evl.operating_prf(records, gt, 0.3, 0.5, len(NAMES))
    op = {"val_per_class": {NAMES[c]: v for c, v in prf.items()},
          "val_headline_mean": {k: float(np.mean([prf[c][k] for c in ids])) for k in ic.PRF}}
    assert ic.metric_parity(records, gt, op, 0.3, 0.5, NAMES, ids)["max_abs_diff"] == 0
    op["val_per_class"]["tumor"] = op["val_per_class"]["tumor"] | {"f1": op["val_per_class"]["tumor"]["f1"] + 0.02}
    m = ic.metric_parity(records, gt, op, 0.3, 0.5, NAMES, ids)
    assert m["max_abs_diff"] == pytest.approx(0.02) and m["abs_diff"]["tumor"]["f1"] == pytest.approx(0.02)

def test_one_class_per_box_keeps_most_confident_duplicate_and_clips():
    box = [10, 10, 40, 40]
    rec = {"boxes": np.array([box, box, [290, 0, 320, 30], [0, 0, 5, 5]], float),
           "conf": np.array([0.4, 0.8, 0.5, 0.1], np.float32), "cls": np.array([0, 1, 2, 0])}
    boxes, conf, cls = ic.one_class_per_box(rec, 0.24, (200, 300))
    assert cls.tolist() == [1, 2] and conf.tolist() == pytest.approx([0.8, 0.5])
    np.testing.assert_array_equal(boxes, [box, [290, 0, 300, 30]])

def test_box_agreement_matches_identical_boxes_and_reports_the_rest():
    pred = {"stem": "s", "boxes": np.array([[10, 10, 40, 40], [50, 50, 90, 90]], float),
            "conf": np.array([0.9, 0.5]), "cls": np.array([0, 1])}
    ref = {"stem": "s", "boxes": pred["boxes"].copy(), "conf": pred["conf"].astype(np.float32), "cls": pred["cls"]}
    a = ic.box_agreement([pred], [ref], {"s": (200, 300)}, 0.24, NAMES)
    assert a["matched"] == 2 and a["matched_fraction_of_predictor"] == 1 and a["max_abs_coordinate_diff_px"] == 0
    ref["cls"] = np.array([0, 0])
    a = ic.box_agreement([pred], [ref], {"s": (200, 300)}, 0.24, NAMES)
    assert a["matched"] == 1 and a["predictor_only"]["non_tumor"] == 1 and a["val_path_only"]["tumor"] == 1

GT_BOX = (120, 80, 180, 120)  # label "0 0.5 0.5 0.2 0.2" in a 300x200 crop

@pytest.fixture
def mocked_check(tmp_path, monkeypatch):
    """run_check on a fake frozen selection and synthetic val/test images, with the heavy gates replaced."""
    ctx, ev = _fake_frozen(tmp_path)
    final = ctx.runs / "a2_geometry" / "seed0" / "weights" / "best.pt"
    cfg = replace(REAL, weights=final, sha256=evl.ex._sha256(final))
    for split in ("val", "test"):
        for stem in (f"TCGA-AA-0001-DX1_id-{split}1", f"TCGA-AA-0001-DX1_id-{split}2"):
            for d in ("images", "labels"):
                (ctx.paths.processed_dir / d / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ctx.paths.processed_dir / "images" / split / f"{stem}.png"), _rgb())
            (ctx.paths.processed_dir / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    seen = []

    class Stub:
        def __init__(self, c, classes, device=None):
            assert c == cfg

        def predict(self, source, conf=None):
            seen.append(source)
            return _result([_det(0, 0.9, GT_BOX)])

    def run_val(model, data_yaml, split, out):
        assert split == "val" and model.name == "final"
        out.mkdir(parents=True)
        stems = sorted(p.stem for p in (ctx.paths.processed_dir / "images" / "val").glob("*.png"))
        return None, {"images": [{"stem": s, "boxes": np.array([GT_BOX], float), "conf": np.array([0.9], np.float32),
                                  "cls": np.array([0])} for s in stems]}

    monkeypatch.setattr(ic, "Predictor", Stub)
    monkeypatch.setattr(ic.evl, "run_val", run_val)
    monkeypatch.setattr(ic.tr, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(ic.tr, "gate", lambda *a: {"dataset": "fp"})
    monkeypatch.setattr(ic, "dataset_fingerprint", lambda *a: {"dataset": "fp"})
    monkeypatch.setattr(ic.ex, "require_baseline", lambda c: {"detail": {"manifest_sha256": "m", "files": 1}})
    monkeypatch.setattr(ic.ex, "baseline_check", lambda c: {"passed": True, "detail": {}})
    monkeypatch.setattr(ic.evl, "raw_hash", lambda raw: "raw")
    perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    op = {"final": {"threshold": 0.24, "val_per_class": {"tumor": perfect, "non_tumor": zero, "ambiguous": zero},
                    "val_headline_mean": {"precision": 0.5, "recall": 0.5, "f1": 0.5}}}
    (ctx.paths.artifacts_dir / "evaluate").mkdir(parents=True)
    (ctx.paths.artifacts_dir / "evaluate" / "operating_points.json").write_text(json.dumps(op))
    return ctx, ev, cfg, seen, op

def test_check_reads_only_val_and_publishes(mocked_check):
    ctx, ev, cfg, seen, _ = mocked_check
    assert ic.run_check(ctx, ev, cfg) is True
    assert len(seen) == 2 and all(p.parent == ctx.paths.processed_dir / "images" / "val" for p in seen)
    root = ctx.paths.artifacts_dir / ic.STEP
    assert sorted(p.name for p in root.iterdir()) == sorted(ic.OUTPUTS)
    checks = json.loads((root / "checks.json").read_text())
    assert all(c["passed"] for c in checks.values()) and checks["metric_parity"]["detail"]["max_abs_diff"] == 0
    parity = json.loads((root / "parity.json").read_text())
    assert parity["box_agreement"]["matched_fraction_of_predictor"] == 1 and len(parity["examples"]) == 2
    assert ic.run_check(ctx, ev, cfg) is True  # re-runnable: outputs are replaced

def test_check_fails_when_parity_exceeds_tolerance(mocked_check):
    ctx, ev, cfg, _, op = mocked_check
    op["final"]["val_headline_mean"]["f1"] = 0.52
    (ctx.paths.artifacts_dir / "evaluate" / "operating_points.json").write_text(json.dumps(op))
    assert ic.run_check(ctx, ev, cfg) is False
    checks = json.loads((ctx.paths.artifacts_dir / ic.STEP / "checks.json").read_text())
    assert not checks["metric_parity"]["passed"]

@pytest.mark.parametrize("change, match", [
    ({"threshold": 0.29}, "threshold_matches_val_choice"),
    ({"sha256": "0" * 64}, "weights_match_selection"),
    ({"max_det": 300}, "settings_match_training"),
])
def test_check_refuses_config_that_disagrees_with_frozen_record(mocked_check, change, match):
    ctx, ev, cfg, seen, _ = mocked_check
    with pytest.raises(RuntimeError, match=match):
        ic.run_check(ctx, ev, replace(cfg, **change))
    assert not seen

def test_check_refuses_changed_selection_and_dirty_tree(mocked_check, monkeypatch):
    ctx, ev, cfg, seen, _ = mocked_check
    with pytest.raises(RuntimeError, match="selection_sha256"):
        ic.run_check(ctx, replace(ev, selection_sha256="0" * 64), cfg)
    monkeypatch.setattr(ic.tr, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(RuntimeError, match="dirty"):
        ic.run_check(ctx, ev, cfg)
    assert not seen