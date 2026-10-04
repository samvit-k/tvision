import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou

from tvision import evaluate as evl
from tvision import experiments as ex
from tvision import train as tr
from tvision.config import CONFIG_DIR, PROJECT_ROOT, Classes, Evaluate, TwoClass, load_yaml

from test_train import _verified

CLASSES = Classes.load()
NAMES = CLASSES.names
NC = len(NAMES)
HEADLINE = [CLASSES.class_id(n) for n in CLASSES.headline]
EV = load_yaml("evaluate.yaml")

def _random_image(rng, n_gt: int, n_extra: int, nc: int = NC) -> tuple[dict, tuple]:
    """GT boxes, jittered copies of them as predictions plus random extras, with random classes and confidences."""
    xy = rng.uniform(0, 200, (n_gt, 2))
    gt_boxes = np.hstack([xy, xy + rng.uniform(10, 40, (n_gt, 2))])
    jitter = gt_boxes + rng.normal(0, 4, gt_boxes.shape)
    exy = rng.uniform(0, 200, (n_extra, 2))
    extra = np.hstack([exy, exy + rng.uniform(10, 40, (n_extra, 2))])
    boxes = np.vstack([jitter, extra])
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)
    record = {"boxes": boxes, "conf": rng.uniform(0, 1, len(boxes)).astype(np.float32),
              "cls": rng.integers(0, nc, len(boxes))}
    return record, (rng.integers(0, nc, n_gt), gt_boxes)

def _images(seed: int = 0, n: int = 6) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    records, gt = [], {}
    for i in range(n):
        rec, g = _random_image(rng, int(rng.integers(0, 12)), int(rng.integers(0, 8)))
        records.append(rec | {"stem": f"img{i}"})
        gt[f"img{i}"] = g
    return records, gt

def test_box_iou_equals_ultralytics():
    rec, (_, gb) = _random_image(np.random.default_rng(1), 10, 5)
    theirs = box_iou(torch.tensor(gb), torch.tensor(rec["boxes"])).numpy()
    np.testing.assert_allclose(evl.box_iou(gb, rec["boxes"]), theirs, rtol=1e-5, atol=1e-6)

def test_match_class_aware_equals_ultralytics(tmp_path):
    validator = DetectionValidator(save_dir=tmp_path)
    rng = np.random.default_rng(2)
    for _ in range(20):
        rec, (gc, gb) = _random_image(rng, int(rng.integers(1, 15)), int(rng.integers(0, 10)))
        iou = evl.box_iou(gb, rec["boxes"]).astype(np.float32)
        theirs = validator.match_predictions(torch.tensor(rec["cls"]), torch.tensor(gc), torch.tensor(iou)).numpy()
        np.testing.assert_array_equal(evl.match_class_aware(rec["cls"], gc, iou, validator.iouv.tolist()), theirs)

def test_ap_table_equals_ultralytics_ap_per_class():
    records, gt = _images(3, 10)
    stats = evl.match_stats(records, gt)
    ours = evl.ap_table(stats, NC)
    cat = {k: np.concatenate([s[k] for s in stats]) for k in ("tp", "conf", "cls", "gt_cls")}
    out = ap_per_class(cat["tp"], cat["conf"], cat["cls"], cat["gt_cls"])
    ap, classes = out[5], out[6]
    np.testing.assert_allclose(ours[classes], ap, atol=1e-12)
    assert set(classes) == set(range(NC))

def test_ap_table_marks_classes_without_gt_and_without_predictions():
    gt = {"a": (np.array([0]), np.array([[0, 0, 10, 10.0]]))}
    rec = [{"stem": "a", "boxes": np.array([[0, 0, 10, 10.0]]), "conf": np.array([0.9], np.float32), "cls": np.array([1])}]
    ap = evl.ap_table(evl.match_stats(rec, gt), NC)
    assert (ap[0] == 0).all() and np.isnan(ap[1]).all() and np.isnan(ap[2]).all()

def test_confusion_equals_ultralytics():
    records, gt = _images(4, 12)
    records.append({"stem": "empty_gt", "boxes": np.array([[0, 0, 5, 5.0]]), "conf": np.array([0.9], np.float32),
                    "cls": np.array([2])})
    gt["empty_gt"] = (np.zeros(0, int), np.zeros((0, 4)))
    records.append({"stem": "no_pred", "boxes": np.zeros((0, 4)), "conf": np.zeros(0, np.float32),
                    "cls": np.zeros(0, int)})
    gt["no_pred"] = (np.array([1]), np.array([[0, 0, 5, 5.0]]))
    for conf, iou in ((0.001, evl.CM_IOU), (0.4, 0.5)):
        cm = ConfusionMatrix(names=dict(enumerate(NAMES)))
        for r in records:
            gc, gb = gt[r["stem"]]
            cm.process_batch({"cls": torch.tensor(r["cls"]), "conf": torch.tensor(r["conf"]),
                              "bboxes": torch.tensor(r["boxes"], dtype=torch.float32)},
                             {"cls": torch.tensor(gc), "bboxes": torch.tensor(gb, dtype=torch.float32)},
                             conf=conf, iou_thres=iou)
        np.testing.assert_array_equal(evl.confusion(records, gt, NC, conf, iou), cm.matrix.astype(int))

def test_has_ties_detects_equal_competing_ious_only():
    assert not evl.has_ties(np.array([[0.9, 0.6], [0.2, 0.7]]), 0.45)
    assert evl.has_ties(np.array([[0.8, 0.8], [0.1, 0.2]]), 0.45)
    assert evl.has_ties(np.array([[0.7, 0.1], [0.7, 0.2]]), 0.45)
    assert not evl.has_ties(np.array([[0.3, 0.3]]), 0.45)

def _duplicate_image():
    """One GT tumor and the same box predicted as tumor and non_tumor (multi-label NMS), plus a second GT."""
    t, n = HEADLINE
    gt = {"x": (np.array([t, n]), np.array([[0, 0, 10, 10], [50, 0, 60, 10.0]]))}
    rec = {"stem": "x", "boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 0, 60, 10.0]]),
           "conf": np.array([0.6, 0.9, 0.8], np.float32), "cls": np.array([t, n, n])}
    return rec, gt

def test_confusion_check_is_exact_except_for_tie_order():
    rec, gt = _duplicate_image()
    ours = evl.confusion([rec], gt, NC, 0.001, evl.CM_IOU)
    store = {"images": [rec], "cm_deltas": [ours], "conf": 0.001}
    assert evl.confusion_check(store, gt, ours, NC)["passed"]
    t, n = HEADLINE
    swapped = ours.copy()
    swapped[t, t], swapped[n, t], swapped[t, NC], swapped[n, NC] = (ours[n, t], ours[t, t], ours[n, NC], ours[t, NC])
    check = evl.confusion_check(store | {"cm_deltas": [swapped]}, gt, swapped, NC)
    assert check["passed"] and check["detail"]["tie_order_differences"] == {"x": 4}
    missed = ours.copy()
    missed[NC, t] += 1
    missed[t, t] -= 1
    assert not evl.confusion_check(store | {"cm_deltas": [missed]}, gt, missed, NC)["passed"]
    assert not evl.confusion_check(store, gt, ours + 1, NC)["detail"]["increments_sum_to_matrix"]

def test_confusion_check_requires_exact_match_without_ties():
    rec, gt = _hand_built()[:2]
    rec = {k: v[:5] if k != "stem" else v for k, v in rec[0].items()}
    ours = evl.confusion([rec], gt, NC, 0.001, evl.CM_IOU)
    assert not evl.has_ties(evl.box_iou(gt["x"][1], rec["boxes"]), evl.CM_IOU)
    wrong = ours.copy()
    wrong[0, 0] -= 1
    wrong[1, 0] += 1
    check = evl.confusion_check({"images": [rec], "cm_deltas": [wrong], "conf": 0.001}, gt, wrong, NC)
    assert not check["passed"] and check["detail"]["mismatched_without_ties"] == ["x"]

def test_operating_point_ties_go_to_the_most_confident_duplicate():
    rec, gt = _duplicate_image()
    t, n = HEADLINE
    err = evl.errors([rec], gt, 0.5, 0.5).set_index("pred_index")
    assert err.loc[1, "kind"] == "misclassified" and err.loc[0, "kind"] == "fp_duplicate"
    rec["conf"] = np.array([0.9, 0.6, 0.8], np.float32)
    err = evl.errors([rec], gt, 0.5, 0.5).set_index("pred_index")
    assert err.loc[0, "kind"] == "tp" and err.loc[1, "kind"] == "fp_duplicate"
    cm = evl.cm_from_errors(err.reset_index(), NC)
    assert cm[t, t] == 1 and cm[n, NC] == 1 and cm[n, n] == 1

def test_unscale_inverts_letterbox_without_clipping():
    pbatch = {"imgsz": (640, 640), "ori_shape": (300, 400), "ratio_pad": ((1.6, 1.6), (0, 80))}
    inside = torch.tensor([[16.0, 96.0, 160.0, 400.0]])
    np.testing.assert_allclose(evl.unscale(inside, pbatch), [[10, 10, 100, 200]], atol=1e-5)
    clipped = ops.scale_boxes((640, 640), inside.clone(), (300, 400), ratio_pad=pbatch["ratio_pad"])
    np.testing.assert_allclose(evl.unscale(inside, pbatch), clipped.numpy(), atol=1e-5)
    outside = evl.unscale(torch.tensor([[-16.0, 96.0, 700.0, 400.0]]), pbatch)
    assert outside[0, 0] == pytest.approx(-10) and outside[0, 2] > 400

def _hand_built():
    t, n, a = (CLASSES.class_id(c) for c in ("tumor", "non_tumor", "ambiguous"))
    gt = {"x": (np.array([t, n, a, t]), np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10],
                                                   [60, 0, 70, 10.0]]))}
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [42, 0, 52, 10], [64, 0, 74, 10], [100, 100, 110, 110],
                      [0, 0, 10, 10.0]])
    rec = [{"stem": "x", "boxes": boxes, "conf": np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1], np.float32),
            "cls": np.array([t, t, t, t, n, n])}]
    return rec, gt, (t, n, a)

def test_errors_classify_every_box_once():
    rec, gt, (t, n, a) = _hand_built()
    err = evl.errors(rec, gt, 0.5, 0.5)
    preds = err[err["kind"] != "fn_missed"].set_index("pred_index")
    assert preds["kind"].to_dict() == {0: "tp", 1: "misclassified", 2: "misclassified", 3: "fp_localization",
                                       4: "fp_background"}
    fn = err[err["kind"] == "fn_missed"]
    assert len(fn) == 1 and fn.iloc[0]["gt_index"] == 3 and fn.iloc[0]["pred_index"] == 3
    assert fn.iloc[0]["iou"] == pytest.approx(60 / 140, abs=1e-6) and preds.loc[3, "gt_index"] == 3
    assert preds.loc[2, "gt_cls"] == a and preds.loc[4, "gt_index"] == -1
    cm = evl.cm_from_errors(err, NC)
    assert cm[t, t] == 1 and cm[t, n] == 1 and cm[t, a] == 1 and cm[t, NC] == 1 and cm[n, NC] == 1 and cm[NC, t] == 1
    assert cm.sum() == len(err)
    counts = evl.error_counts(err, NAMES, a)
    assert counts["involving_ambiguous"] == {"misclassified": 1}
    assert counts["misclassified_true_to_pred"] == {"ambiguous -> tumor": 1, "non_tumor -> tumor": 1}
    assert counts["by_kind_and_class"]["fn_missed"]["tumor"] == 1

def test_operating_prf_counts_class_aware_matches():
    rec, gt, (t, n, a) = _hand_built()
    prf = evl.operating_prf(rec, gt, 0.5, 0.5, NC)
    assert prf[t] == pytest.approx({"tp": 1, "fp": 3, "fn": 1, "precision": 0.25, "recall": 0.5, "f1": 1 / 3})
    assert prf[n]["tp"] == 0 and prf[n]["fp"] == 1 and prf[n]["fn"] == 1 and prf[a]["fn"] == 1

def test_operating_threshold_maximises_headline_f1_and_breaks_ties_low():
    t, n = HEADLINE
    gt = {"x": (np.array([t, n]), np.array([[0, 0, 10, 10], [20, 0, 30, 10.0]]))}
    rec = [{"stem": "x", "boxes": np.array([[0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60.0]]),
            "conf": np.array([0.9, 0.6, 0.305], np.float32), "cls": np.array([t, n, t])}]
    op = evl.operating_threshold(rec, gt, HEADLINE, 0.5, NC)
    assert op["threshold"] == 0.31 and op["mean_headline_f1"] == 1.0
    curve = dict(zip(op["grid"], op["curve"]))
    assert curve[0.3] == pytest.approx((2 / 3 + 1) / 2) and curve[0.6] == 1.0 and curve[0.61] == 0.5

def _bootstrap_stats(shift: float = 0.0) -> tuple[list[dict], dict]:
    t, n = HEADLINE
    gt = {"s1a": (np.array([t]), np.array([[0, 0, 10, 10.0]])), "s1b": (np.array([n]), np.array([[0, 0, 10, 10.0]])),
          "s2a": (np.array([t, n]), np.array([[0, 0, 10, 10], [20, 0, 30, 10.0]]))}
    records = [{"stem": s, "boxes": gb + np.array([shift, 0, shift, 0]), "conf": np.full(len(gc), 0.9, np.float32),
                "cls": gc} for s, (gc, gb) in gt.items()]
    return evl.match_stats(records, gt), {"s1a": "S1", "s1b": "S1", "s2a": "S2"}

def test_bootstrap_identical_models_have_zero_difference_and_slides_move_together():
    stats, slides = _bootstrap_stats()
    out = evl.slide_bootstrap({"baseline": stats, "final": stats}, slides, HEADLINE, NC, 200, 0, 0.95)
    assert out["used"] == 200, "a slide's images must be drawn together, so every resample has both classes"
    assert out["difference"]["mAP50"]["ci"] == [0.0, 0.0] and out["difference"]["mAP50"]["fraction_positive"] == 0.0
    assert out["slides"] == 2 and out["scope"] == evl.CI_SCOPE

def test_bootstrap_is_paired_and_seeded():
    good, slides = _bootstrap_stats()
    worse, _ = _bootstrap_stats(shift=4.0)
    a = evl.slide_bootstrap({"baseline": worse, "final": good}, slides, HEADLINE, NC, 100, 7, 0.9)
    b = evl.slide_bootstrap({"baseline": worse, "final": good}, slides, HEADLINE, NC, 100, 7, 0.9)
    assert a == b
    assert a["difference"]["mAP50-95"]["fraction_positive"] == 1.0 and a["difference"]["mAP50-95"]["point"] > 0
    with pytest.raises(RuntimeError, match="same images"):
        evl.slide_bootstrap({"baseline": good[:2], "final": good}, slides, HEADLINE, NC, 10, 0, 0.9)

def test_ann_ids_follow_kept_rows_and_detect_mismatch(tmp_path):
    t, n = HEADLINE
    pd.DataFrame({"stem": ["a", "a", "a", "b"], "ann_id": [0, 1, 2, 0], "split": "test",
                  "class_id": [t, n, n, t], "status": ["kept", "dropped_type", "kept", "kept"]}).to_csv(
        tmp_path / "annotations.csv", index=False)
    gt = {"a": (np.array([t, n]), np.zeros((2, 4))), "b": (np.array([t]), np.zeros((1, 4)))}
    ids, check = evl.ann_ids(tmp_path, "test", gt)
    assert ids == {"a": [0, 2], "b": [0]} and check["passed"]
    _, check = evl.ann_ids(tmp_path, "test", gt | {"b": (np.array([n]), np.zeros((1, 4)))})
    assert not check["passed"] and check["detail"]["mismatched"] == ["b"]

def test_evaluate_config_validation():
    Evaluate(**EV)
    for change in ({"selection_sha256": "abc"}, {"match_iou": 1.0}, {"gallery": {**EV["gallery"], "extra": 1}},
                   {"gallery": {**EV["gallery"], "per_kind": 0}}, {"bootstrap": {**EV["bootstrap"], "level": 1.0}},
                   {"bootstrap": {**EV["bootstrap"], "resamples": True}}):
        with pytest.raises(ValueError):
            Evaluate(**(EV | change))

@pytest.mark.skipif(not (PROJECT_ROOT / "artifacts" / "step5a" / "selection.json").is_file(), reason="no selection")
def test_committed_anchor_matches_frozen_selection():
    assert ex._sha256(PROJECT_ROOT / "artifacts" / "step5a" / "selection.json") == EV["selection_sha256"]

def test_ledger_logs_second_model_failure_after_first_completed(tmp_path):
    models = [evl.Model(n, tmp_path, tmp_path / f"{n}.pt", f"sha-{n}", 640, 1000, 16, "cpu") for n in ("baseline", "final")]
    evl.ledger(tmp_path, "evaluation_started", models={m.name: m.sha256 for m in models})

    def run_one(m):
        if m.name == "final":
            raise RuntimeError("crash in final")
        return ("ok",)

    with pytest.raises(RuntimeError, match="crash in final"):
        evl.logged_test_passes(tmp_path, models, run_one)
    events = [json.loads(l) for l in (tmp_path / evl.LEDGER).read_text().splitlines()]
    assert [(e["event"], e.get("model")) for e in events] == [
        ("evaluation_started", None), ("test_pass_started", "baseline"), ("test_pass_completed", "baseline"),
        ("test_pass_started", "final"), ("test_pass_failed", "final")]
    assert events[3]["sha256"] == "sha-final" and "crash in final" in events[4]["error"]
    assert evl.test_passes(tmp_path) == {"baseline": 1, "final": 1}
    assert evl.logged_test_passes(tmp_path, models[:1], run_one) == {"baseline": ("ok",)}
    assert evl.test_passes(tmp_path) == {"baseline": 2, "final": 1}

def _fake_frozen(tmp_path: Path) -> tuple[ex.Context, Evaluate]:
    paths = tr.Paths(tmp_path / "raw", tmp_path / "processed", tmp_path / "artifacts", tmp_path / "runs")
    ctx = ex.Context(paths, CLASSES, copy.deepcopy(load_yaml("experiments.yaml")),
                     TwoClass(tmp_path / "two", ["ambiguous"], ["train", "val"]), CONFIG_DIR / "train.yaml")
    config = {"imgsz": 640, "max_det": 1000, "batch": 16, "device": "cpu", "seed": 0}
    final = ctx.runs / "a2_geometry" / "seed0"
    for run_dir, data in ((ctx.baseline_dir, b"baseline"), (final, b"final")):
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_bytes(data)
        (run_dir / "run_meta.json").write_text(json.dumps({"best": {"sha256": ex._sha256(run_dir / "weights" / "best.pt")},
                                                           "config": config}))
    ctx.artifacts.mkdir(parents=True)
    (ctx.artifacts / "selection.json").write_text(json.dumps({
        "experiment": "a2_geometry", "seed": 0, "run_dir": str(final), "best_pt": str(final / "weights" / "best.pt"),
        "best_pt_sha256": ex._sha256(final / "weights" / "best.pt"), "dataset": "three_class", "config": config}))
    return ctx, Evaluate(**(EV | {"selection_sha256": ex._sha256(ctx.artifacts / "selection.json")}))

def test_frozen_models_verify_every_hash(tmp_path):
    ctx, ev = _fake_frozen(tmp_path)
    models, frozen = evl.frozen_models(ctx, ev)
    assert [m.name for m in models] == ["baseline", "final"] and frozen["experiment"] == "a2_geometry"
    with pytest.raises(RuntimeError, match="selection_sha256"):
        evl.frozen_models(ctx, replace(ev, selection_sha256="0" * 64))
    (ctx.runs / "a2_geometry" / "seed0" / "weights" / "best.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="run_meta"):
        evl.frozen_models(ctx, ev)

def test_frozen_models_refuse_two_class_selection(tmp_path):
    ctx, ev = _fake_frozen(tmp_path)
    sel = ctx.artifacts / "selection.json"
    sel.write_text(json.dumps(json.loads(sel.read_text()) | {"dataset": "two_class"}))
    with pytest.raises(RuntimeError, match="two-class"):
        evl.frozen_models(ctx, replace(ev, selection_sha256=ex._sha256(sel)))

def test_existing_result_refused(tmp_path):
    ctx, ev = _fake_frozen(tmp_path)
    root = ctx.paths.artifacts_dir / evl.STEP
    root.mkdir(parents=True)
    (root / "result.json").write_text("{}")
    with pytest.raises(RuntimeError, match="made once"):
        evl.run_evaluation(ctx, ev)

@pytest.mark.parametrize("check_only", [False, True])
def test_dirty_tree_refused_in_both_modes(tmp_path, monkeypatch, check_only):
    ctx, ev = _fake_frozen(tmp_path)
    monkeypatch.setattr(evl.tr, "git_state", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(evl.tr, "gate", lambda *a: pytest.fail("gates must not run on a dirty tree"))
    with pytest.raises(RuntimeError, match="dirty"):
        evl.run_evaluation(ctx, ev, check_only=check_only)

@pytest.fixture
def mocked_pipeline(tmp_path, monkeypatch):
    """run_evaluation with every model run and heavy gate replaced; records which phases ran."""
    ctx, ev = _fake_frozen(tmp_path)
    ctx.paths.processed_dir.mkdir(parents=True)
    pd.DataFrame({"stem": ["s"], "slide": ["S"], "hospital": ["H"]}).to_csv(ctx.paths.processed_dir / "splits.csv",
                                                                            index=False)
    calls = {"val": [], "test": [], "val_passes": True, "fail_test": None}
    monkeypatch.setattr(evl.tr, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(evl.tr, "gate", lambda *a: {"dataset": "fp"})
    monkeypatch.setattr(evl, "dataset_fingerprint", lambda *a: {"dataset": "fp"})
    monkeypatch.setattr(evl.ex, "require_baseline", lambda c: {"detail": {"manifest_sha256": "m", "files": 1}})
    monkeypatch.setattr(evl.ex, "baseline_check", lambda c: {"passed": True, "detail": {}})
    monkeypatch.setattr(evl.tr, "check_device", lambda d: None)
    monkeypatch.setattr(evl, "raw_hash", lambda raw: "raw")
    monkeypatch.setattr(evl, "ground_truth", lambda *a: {"s": (np.zeros(0, int), np.zeros((0, 4)))})
    monkeypatch.setattr(evl, "ann_ids", lambda *a: ({"s": []}, {"passed": True, "detail": {}}))

    def val_phase(m, *a):
        calls["val"].append(m.name)
        return {f"{m.name}/val/x": {"passed": calls["val_passes"], "detail": None}}, {"threshold": 0.5,
                                                                                      "val_headline_mean": {}}

    def test_phase(m, *a):
        calls["test"].append(m.name)
        if m.name == calls["fail_test"]:
            raise RuntimeError(f"crash in {m.name}")
        return {}, {}, [], [], pd.DataFrame()

    monkeypatch.setattr(evl, "val_phase", val_phase)
    monkeypatch.setattr(evl, "test_phase", test_phase)
    return ctx, ev, calls

def test_failed_val_phase_never_reaches_test(mocked_pipeline):
    ctx, ev, calls = mocked_pipeline
    calls["val_passes"] = False
    assert evl.run_evaluation(ctx, ev) is False
    assert calls["val"] == ["baseline", "final"] and calls["test"] == []
    assert not (ctx.paths.artifacts_dir / evl.STEP / evl.LEDGER).exists()

def test_check_only_never_reaches_test(mocked_pipeline):
    ctx, ev, calls = mocked_pipeline
    assert evl.run_evaluation(ctx, ev, check_only=True) is True
    assert calls["test"] == [] and not (ctx.paths.artifacts_dir / evl.STEP / evl.LEDGER).exists()
    assert (ctx.paths.artifacts_dir / evl.STEP / "check" / "summary.md").is_file()

def test_second_model_crash_leaves_audit_trail_and_no_result(mocked_pipeline):
    ctx, ev, calls = mocked_pipeline
    calls["fail_test"] = "final"
    with pytest.raises(RuntimeError, match="crash in final"):
        evl.run_evaluation(ctx, ev)
    root = ctx.paths.artifacts_dir / evl.STEP
    events = [json.loads(l) for l in (root / evl.LEDGER).read_text().splitlines()]
    assert [(e["event"], e.get("model")) for e in events] == [
        ("evaluation_started", None), ("test_pass_started", "baseline"), ("test_pass_completed", "baseline"),
        ("test_pass_started", "final"), ("test_pass_failed", "final"), ("evaluation_failed", None)]
    assert calls["test"] == ["baseline", "final"] and not (root / "result.json").exists()
    assert evl.test_passes(root) == {"baseline": 1, "final": 1}

@pytest.mark.skipif(not (PROJECT_ROOT / "yolo26n.pt").is_file(), reason="no yolo26n.pt")
def test_recording_validator_reproduces_ultralytics_on_cpu(tmp_path):
    paths = _verified(tmp_path)
    model = evl.Model("final", tmp_path, PROJECT_ROOT / "yolo26n.pt", "", 320, 300, 2, "cpu")
    metrics, store = evl.run_val(model, paths.processed_dir / "data.yaml", "val", tmp_path / "out" / "val")
    nc = len(metrics.names)
    gt = evl.ground_truth(paths.processed_dir, "val", nc)
    assert len(store["images"]) == store["seen"] == len(gt) == len(store["cm_deltas"]) and store["conf"] == 0.001
    np.testing.assert_array_equal(sum(store["cm_deltas"]), metrics.confusion_matrix.matrix)
    checks = evl.scoring_checks("final/val", metrics, store, gt, evl.match_stats(store["images"], gt), nc)
    assert all(c["passed"] for c in checks.values()), checks
    with pytest.raises(RuntimeError, match="already exists"):
        evl.run_val(model, paths.processed_dir / "data.yaml", "val", tmp_path / "out" / "val")