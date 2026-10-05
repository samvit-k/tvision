"""Step 7 diagnostic (val only): does letterbox geometry or NMS mode explain the Predictor-vs-val parity gap?

Scores the frozen final checkpoint through `Model.predict` under four conditions, rect in {False, True} x NMS in
{single-label, multi-label}, holding the checkpoint, threshold, imgsz, max_det, IoU and input decoding fixed. Each is
scored with Step 6's operating-point code against the frozen val operating point and compared box by box with a fresh
Step 6 val-path run. Diagnostic only: it changes no deployment code, config or gate, and writes only to
artifacts/predict_diagnostic/. No test images or labels are used for model inference, scoring, model selection, or
diagnostic comparisons; test-split files may participate only in frozen whole-dataset integrity checks.
Run with `python -m tvision.predict_diagnostic`.
"""

from tvision import evaluate as evl  # imports tvision.train first, which configures Ultralytics

import json
import shutil
import sys

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import nms, ops

from tvision import experiments as ex
from tvision import inference_check as ic
from tvision import train as tr
from tvision.config import Evaluate, Inference
from tvision.explore import _block
from tvision.inference import load_image
from tvision.verify import dataset_fingerprint

STEP = "predict_diagnostic"
SPLIT = "val"
CONDITIONS = {  # name: (rect, multi_label)
    "deployed": (False, False),
    "geometry_only": (True, False),
    "nms_only": (False, True),
    "both": (True, True),
}
OUTPUTS = ("checks.json", "diagnostic.json", "summary.md", "val_reference")
GEOMETRY_NOTE = (
    "rect=True in Model.predict with one image at a time pads each image only to its own minimal stride-32 "
    "rectangle. Step 6's validator resizes with load_image (long side to imgsz, ceil rounding of the short side, where "
    "predict's LetterBox rounds) and pads every image of a batch of 16 to a shared rect batch shape "
    "ceil(shape * imgsz / stride + 0.5) * stride. rect=True is therefore only closer to the val geometry; it is not "
    "guaranteed to reproduce it.")
SCORING_NOTE = (
    "Metrics score each condition's raw predictions exactly as Step 6 scored the val path (multi-label conditions keep "
    "one row per class of a box). Box agreement reduces both sides to the most confident class per identical box.")

class MultiLabelPredictor(DetectionPredictor):
    """DetectionPredictor with the validator's multi_label=True NMS; every other NMS argument as in the parent."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms,
                                        multi_label=True, max_det=self.args.max_det, nc=0,
                                        end2end=getattr(self.model, "end2end", False))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(preds, img, orig_imgs, **kwargs)

def predict_condition(cfg: Inference, images: dict[str, np.ndarray], rect: bool, multi_label: bool) -> list[dict]:
    """Records for every image from a fresh model, with the deployed Predictor's settings except rect and NMS mode."""
    model = YOLO(cfg.weights, task="detect")
    records = []
    for stem, img in images.items():
        r = model.predict(img, imgsz=cfg.imgsz, conf=cfg.threshold, iou=cfg.iou, max_det=cfg.max_det,
                          device=cfg.device, rect=rect, verbose=False,
                          predictor=MultiLabelPredictor if multi_label else None)[0]
        h, w = r.orig_shape
        records.append({"stem": stem, "boxes": np.clip(r.boxes.xyxy.cpu().numpy().astype(np.float64), 0, [w, h, w, h]),
                        "conf": r.boxes.conf.cpu().numpy().astype(np.float64),
                        "cls": r.boxes.cls.cpu().numpy().astype(int)})
    return records

def reduced(records: list[dict], threshold: float, shapes: dict) -> list[dict]:
    return [dict(zip(("boxes", "conf", "cls"), ic.one_class_per_box(r, threshold, shapes[r["stem"]])), stem=r["stem"])
            for r in records]

def deltas(metrics: dict, deployed: dict) -> dict:
    return {g: {k: metrics["predictor"][g][k] - deployed["predictor"][g][k] for k in ic.PRF}
            for g in metrics["predictor"]}

def summarize(checks: dict, result: dict) -> str:
    groups = list(result["conditions"]["deployed"]["metrics"]["predictor"])
    head = "| condition | rect | NMS | boxes (raw / one per box) | " + " | ".join(
        f"{g} P/R/F1" for g in groups) + " | max abs diff vs Step 6 |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    s6 = result["conditions"]["deployed"]["metrics"]["step6_val"]
    rows.append("| Step 6 val path (frozen) | batch rect | multi | - | " + " | ".join(
        "/".join(f"{s6[g][k]:.4f}" for k in ic.PRF) for g in groups) + " | - |")
    for name, c in result["conditions"].items():
        m = c["metrics"]
        rows.append(f"| {name} | {c['rect']} | {'multi' if c['multi_label'] else 'single'} | "
                    f"{c['boxes_raw']} / {c['boxes_one_per_box']} | " + " | ".join(
                        "/".join(f"{m['predictor'][g][k]:.4f}" for k in ic.PRF) for g in groups)
                    + f" | {m['max_abs_diff']:.4f} |")
    agree = ["| condition | matched | of predictor | of val path | predictor only | val path only | max coord px | "
             "max conf |", "|---|---|---|---|---|---|---|---|"]
    for name, c in result["conditions"].items():
        a = c["box_agreement"]
        agree.append(f"| {name} | {a['matched']} | {a['matched_fraction_of_predictor']:.4f} | "
                     f"{a['matched_fraction_of_val_path']:.4f} | {a['predictor_only']} | {a['val_path_only']} | "
                     f"{a['max_abs_coordinate_diff_px']:.2f} | {a['max_abs_confidence_diff']:.4f} |")
    return ("# Step 7 diagnostic: rect x NMS mode (validation-only inference/scoring)\n\n"
            f"Geometry caveat: {GEOMETRY_NOTE}\n\nScoring: {SCORING_NOTE}\n\n"
            + _block("Checks", json.dumps(checks, indent=2))
            + "\n## Operating point on val (conf > threshold, IoU 0.5)\n\n" + "\n".join(rows)
            + "\n\n## Box agreement with the Step 6 val path (IoU >= 0.9, same class)\n\n" + "\n".join(agree) + "\n\n"
            + _block("Change vs the deployed condition (condition - deployed)",
                     json.dumps({n: c["delta_vs_deployed"] for n, c in result["conditions"].items()}, indent=2)))

def run_diagnostic(ctx: ex.Context, ev: Evaluate, cfg: Inference) -> bool:
    root = ctx.paths.artifacts_dir / STEP
    git = tr.git_state()
    if git["dirty"]:
        raise RuntimeError("git working tree is dirty; commit before the diagnostic so it is reproducible")
    processed = ctx.paths.processed_dir
    fingerprint = tr.gate(processed, ctx.paths.artifacts_dir / "verify")
    ex.require_baseline(ctx)
    models, frozen = evl.frozen_models(ctx, ev)
    final = next(m for m in models if m.name == "final")
    operating = json.loads((ctx.paths.artifacts_dir / "evaluate" / "operating_points.json").read_text())
    gates = ic.config_gates(cfg, final, operating)
    if failed := [k for k, c in gates.items() if not c["passed"]]:
        raise RuntimeError(f"configs/inference.yaml disagrees with the frozen Step 5A/6 record: {failed}")
    tr.check_device(cfg.device)
    raw_before = evl.raw_hash(ctx.paths.raw_dir)
    print(f"[PASS] gates: dataset {fingerprint['dataset'][:16]}…, final {final.sha256[:16]}…, threshold {cfg.threshold}")
    work = root / ".tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    names, nc = ctx.classes.names, len(ctx.classes.names)
    ids = [ctx.classes.class_id(n) for n in ctx.classes.headline]
    gt = evl.ground_truth(processed, SPLIT, nc)
    folder = processed / "images" / SPLIT
    images = {s: load_image(folder / f"{s}.png", cfg.min_long_side, cfg.max_long_side) for s in sorted(gt)}
    shapes = {s: img.shape[:2] for s, img in images.items()}
    _, store = evl.run_val(final, processed / "data.yaml", SPLIT, work / "val_reference")
    t = cfg.threshold
    reference = store["images"]
    conditions = {}
    for name, (rect, multi) in CONDITIONS.items():
        print(f"condition {name}: rect={rect}, multi_label={multi}")
        records = predict_condition(cfg, images, rect, multi)
        conditions[name] = {
            "rect": rect, "multi_label": multi, "images": len(records),
            "boxes_raw": int(sum(len(r["cls"]) for r in records)),
            "boxes_one_per_box": int(sum(len(r["cls"]) for r in reduced(records, t, shapes))),
            "metrics": ic.metric_parity(records, gt, operating["final"], t, ev.match_iou, names, ids),
            "box_agreement": ic.box_agreement(reduced(records, t, shapes), reference, shapes, t, names)}
    for c in conditions.values():
        c["delta_vs_deployed"] = deltas(c["metrics"], conditions["deployed"]["metrics"]) | {
            "boxes_raw": c["boxes_raw"] - conditions["deployed"]["boxes_raw"],
            "max_abs_diff_vs_step6": c["metrics"]["max_abs_diff"] - conditions["deployed"]["metrics"]["max_abs_diff"]}
    recorded = json.loads((ctx.paths.artifacts_dir / ic.STEP / "parity.json").read_text())["metric_parity"]
    dep = conditions["deployed"]["metrics"]
    checks = {"gates": evl._check(True, {"dataset": fingerprint["dataset"], "git": git, **frozen}), **gates,
              "every_condition_scored_every_val_image": evl._check(
                  all(c["images"] == len(gt) for c in conditions.values())
                  and sorted(r["stem"] for r in reference) == sorted(gt), {"val_images": len(gt)}),
              "deployed_reproduces_predict_check": evl._check(
                  all(abs(dep["predictor"][g][k] - recorded["predictor"][g][k]) < 1e-12
                      for g in dep["predictor"] for k in ic.PRF),
                  {"max_abs_diff_vs_step6": [dep["max_abs_diff"], recorded["max_abs_diff"]]}),
              "baseline_untouched_after": ex.baseline_check(ctx),
              "fingerprint_unchanged_after": evl._check(dataset_fingerprint(processed) == fingerprint,
                                                        fingerprint["dataset"]),
              "raw_unchanged_after": evl._check(evl.raw_hash(ctx.paths.raw_dir) == raw_before, raw_before)}
    result = {"split": SPLIT, "threshold": t, "match_iou": ev.match_iou, "geometry_note": GEOMETRY_NOTE,
              "scoring_note": SCORING_NOTE, "conditions": conditions, "frozen": frozen, "git": git,
              "versions": tr.versions(), "finished": tr._now()}
    tr._write_json(work / "checks.json", checks)
    tr._write_json(work / "diagnostic.json", result)
    (work / "summary.md").write_text(summarize(checks, result))
    for name in OUTPUTS:
        old = root / name
        if old.is_dir():
            shutil.rmtree(old)
        elif old.exists():
            old.unlink()
        (work / name).rename(old)
    work.rmdir()
    for k, c in checks.items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {k}")
    print(f"Report: {root / 'summary.md'}")
    return all(c["passed"] for c in checks.values())

if __name__ == "__main__":
    sys.exit(0 if run_diagnostic(ex.Context.load(), Evaluate.load(), Inference.load()) else 1)