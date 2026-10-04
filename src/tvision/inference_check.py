"""Step 7 check: the deployed Predictor reproduces Step 6's validation operating point. Val only; never reads test.

Ultralytics' predict path differs from the val path Step 6 scored: one class per box (multi_label=False NMS) and a
square letterbox instead of rect batches. The Predictor is therefore run on every val image, scored with Step 6's own
operating-point code, and must reproduce the per-class P/R/F1 recorded in operating_points.json within PARITY_TOL.
Box-level agreement with the val path is reported, not gated. This module imports the evaluation code and is not
part of the deployment path.
"""

from tvision import evaluate as evl  # imports tvision.train first, which configures Ultralytics

import json
import shutil
from dataclasses import asdict

import cv2
import numpy as np

from tvision import experiments as ex
from tvision import train as tr
from tvision.config import Evaluate, Inference
from tvision.contract import DetectionResult
from tvision.draw import draw_boxes, side_by_side
from tvision.explore import _block
from tvision.inference import Predictor, annotate
from tvision.verify import dataset_fingerprint

STEP = "predict"
SPLIT = "val"
PARITY_TOL = 0.01
AGREEMENT_IOU = 0.9
EXAMPLES = 6
EXAMPLE_SEED = 0
EXAMPLE_SCALE = 2
PRF = ("precision", "recall", "f1")
OUTPUTS = ("checks.json", "parity.json", "summary.md", "examples", "val_reference")

_check = evl._check

def config_gates(cfg: Inference, final: evl.Model, operating: dict) -> dict:
    """configs/inference.yaml must name the frozen checkpoint, its val-chosen threshold and its trained settings."""
    return {
        "weights_match_selection": _check(cfg.weights.resolve() == final.weights.resolve()
                                          and cfg.sha256 == final.sha256,
                                          {"config": [str(cfg.weights), cfg.sha256],
                                           "selection": [str(final.weights), final.sha256]}),
        "threshold_matches_val_choice": _check(cfg.threshold == operating["final"]["threshold"],
                                               {"config": cfg.threshold, "val_chosen": operating["final"]["threshold"]}),
        "settings_match_training": _check(cfg.imgsz == final.imgsz and cfg.max_det == final.max_det,
                                          {"config": [cfg.imgsz, cfg.max_det], "trained": [final.imgsz, final.max_det]}),
    }

def records_of(stem: str, result: DetectionResult) -> dict:
    """A result in the record layout Step 6's scoring functions take."""
    d = result.detections
    return {"stem": stem, "boxes": np.array([[x.box.x1, x.box.y1, x.box.x2, x.box.y2] for x in d]).reshape(-1, 4),
            "conf": np.array([x.confidence for x in d]), "cls": np.array([x.class_id for x in d], dtype=int)}

def metric_parity(records: list[dict], gt: dict, operating: dict, threshold: float, iou: float, names: list[str],
                  ids: list[int]) -> dict:
    """Per-class and headline-mean P/R/F1 of the Predictor's records against Step 6's recorded val values."""
    prf = evl.operating_prf(records, gt, threshold, iou, len(names))
    ours = {names[c]: {k: prf[c][k] for k in PRF} for c in range(len(names))}
    ours["headline_mean"] = {k: float(np.mean([prf[c][k] for c in ids])) for k in PRF}
    recorded = {n: {k: operating["val_per_class"][n][k] for k in PRF} for n in names}
    recorded["headline_mean"] = {k: operating["val_headline_mean"][k] for k in PRF}
    diff = {g: {k: abs(ours[g][k] - recorded[g][k]) for k in PRF} for g in ours}
    return {"predictor": ours, "step6_val": recorded, "abs_diff": diff,
            "max_abs_diff": max(v for d in diff.values() for v in d.values()), "tolerance": PARITY_TOL,
            "counts": {names[c]: {k: prf[c][k] for k in ("tp", "fp", "fn")} for c in range(len(names))}}

def one_class_per_box(rec: dict, threshold: float, shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Val-path records above `threshold`, keeping the most confident class among exactly identical boxes (multi-label
    NMS emits a box once per class), clipped to the image like predict output."""
    keep = rec["conf"] > threshold
    boxes, conf, cls = rec["boxes"][keep], rec["conf"][keep], rec["cls"][keep]
    order = np.argsort(-conf, kind="stable")
    _, first = np.unique(boxes[order], axis=0, return_index=True)
    idx = order[np.sort(first)]
    h, w = shape
    return np.clip(boxes[idx], 0, [w, h, w, h]), conf[idx], cls[idx]

def box_agreement(predicted: list[dict], reference: list[dict], shapes: dict, threshold: float,
                  names: list[str]) -> dict:
    """One-to-one same-class matching at IoU >= AGREEMENT_IOU between Predictor and val-path boxes."""
    nc = len(names)
    ref_by_stem = {r["stem"]: r for r in reference}
    n_pred = n_ref = matched = 0
    max_coord = max_conf = 0.0
    only_pred, only_ref = np.zeros(nc, int), np.zeros(nc, int)
    for p in predicted:
        rb, rconf, rcls = one_class_per_box(ref_by_stem[p["stem"]], threshold, shapes[p["stem"]])
        iou = evl.box_iou(rb, p["boxes"]) * (rcls[:, None] == p["cls"][None, :])
        gm = evl.agnostic_match(iou, AGREEMENT_IOU, strict=False)
        hit = gm >= 0
        n_pred, n_ref, matched = n_pred + len(p["cls"]), n_ref + len(rcls), matched + int(hit.sum())
        if hit.any():
            max_coord = max(max_coord, float(np.abs(rb[hit] - p["boxes"][gm[hit]]).max()))
            max_conf = max(max_conf, float(np.abs(rconf[hit] - p["conf"][gm[hit]]).max()))
        only_ref += np.bincount(rcls[~hit], minlength=nc)
        unmatched = np.setdiff1d(np.arange(len(p["cls"])), gm[hit])
        only_pred += np.bincount(p["cls"][unmatched], minlength=nc)
    return {"iou": AGREEMENT_IOU, "predictor_boxes": n_pred, "val_path_boxes": n_ref, "matched": matched,
            "matched_fraction_of_predictor": matched / n_pred if n_pred else None,
            "matched_fraction_of_val_path": matched / n_ref if n_ref else None,
            "predictor_only": dict(zip(names, only_pred.tolist())), "val_path_only": dict(zip(names, only_ref.tolist())),
            "max_abs_coordinate_diff_px": max_coord, "max_abs_confidence_diff": max_conf}

def examples(out, images, stems: list[str], results: dict[str, DetectionResult], gt: dict, names: list[str]) -> list:
    """GT | Predictor panels for seeded val images, for visual review."""
    out.mkdir(parents=True)
    rng = np.random.default_rng(EXAMPLE_SEED)
    chosen = sorted(rng.choice(stems, size=min(EXAMPLES, len(stems)), replace=False).tolist())
    for k, stem in enumerate(chosen):
        img = cv2.imread(str(images / f"{stem}.png"))
        gc, gb = gt[stem]
        r = results[stem]
        up = lambda im: cv2.resize(im, None, fx=EXAMPLE_SCALE, fy=EXAMPLE_SCALE, interpolation=cv2.INTER_NEAREST)
        panels = [evl.captioned(up(draw_boxes(img, evl._boxes(gb, gc, names))), ["GT", evl._short(stem)]),
                  evl.captioned(up(annotate(img, r, labels=False)),
                                [f"Predictor conf > {r.threshold:.2f}",
                                 ", ".join(f"{n} {c}" for n, c in r.counts.items())])]
        if not cv2.imwrite(str(out / f"{k}_{stem}.png"), side_by_side(*panels)):
            raise RuntimeError(f"failed to write example {out / f'{k}_{stem}.png'}")
    return chosen

def summarize(checks: dict, parity: dict) -> str:
    rows = ["| | P | R | F1 | Step 6 P | Step 6 R | Step 6 F1 |", "|---|---|---|---|---|---|---|"]
    m = parity["metric_parity"]
    for g in m["predictor"]:
        rows.append(f"| {g} | " + " | ".join(f"{m[s][g][k]:.4f}" for s in ("predictor", "step6_val") for k in PRF)
                    + " |")
    return ("# Step 7: Predictor parity on val (no test images read)\n\n"
            + _block("Checks", json.dumps(checks, indent=2))
            + f"\n## Operating point on val (IoU {parity['match_iou']}, conf > {parity['threshold']})\n\n"
            + "\n".join(rows) + f"\n\nmax |diff| {m['max_abs_diff']:.4f} (tolerance {PARITY_TOL})\n\n"
            + _block("Box agreement with the val path (reported, not gated)",
                     json.dumps(parity["box_agreement"], indent=2))
            + _block("Examples (GT | Predictor)", "\n".join(parity["examples"])))

def run_check(ctx: ex.Context, ev: Evaluate, cfg: Inference) -> bool:
    root = ctx.paths.artifacts_dir / STEP
    git = tr.git_state()
    if git["dirty"]:
        raise RuntimeError("git working tree is dirty; commit before the check so it is reproducible")
    processed = ctx.paths.processed_dir
    fingerprint = tr.gate(processed, ctx.paths.artifacts_dir / "verify")
    ex.require_baseline(ctx)
    models, frozen = evl.frozen_models(ctx, ev)
    final = next(m for m in models if m.name == "final")
    op_path = ctx.paths.artifacts_dir / "evaluate" / "operating_points.json"
    if not op_path.is_file():
        raise RuntimeError(f"{op_path} not found; Step 6 must have chosen the operating threshold")
    operating = json.loads(op_path.read_text())
    gates = config_gates(cfg, final, operating)
    if failed := [k for k, c in gates.items() if not c["passed"]]:
        raise RuntimeError(f"configs/inference.yaml disagrees with the frozen Step 5A/6 record: "
                           f"{ {k: gates[k]['detail'] for k in failed} }")
    raw_before = evl.raw_hash(ctx.paths.raw_dir)
    print(f"[PASS] gates: dataset {fingerprint['dataset'][:16]}…, final {final.sha256[:16]}…, "
          f"threshold {cfg.threshold}")
    work = root / ".tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    names, nc = ctx.classes.names, len(ctx.classes.names)
    ids = [ctx.classes.class_id(n) for n in ctx.classes.headline]
    gt = evl.ground_truth(processed, SPLIT, nc)
    images = processed / "images" / SPLIT
    stems = sorted(gt)
    predictor = Predictor(cfg, ctx.classes)
    results = {s: predictor.predict(images / f"{s}.png") for s in stems}
    predicted = [records_of(s, results[s]) for s in stems]
    _, store = evl.run_val(final, processed / "data.yaml", SPLIT, work / "val_reference")
    shapes = {s: (r.image.height, r.image.width) for s, r in results.items()}
    parity = {"split": SPLIT, "threshold": cfg.threshold, "match_iou": ev.match_iou,
              "metric_parity": metric_parity(predicted, gt, operating["final"], cfg.threshold, ev.match_iou, names, ids),
              "box_agreement": box_agreement(predicted, store["images"], shapes, cfg.threshold, names),
              "examples": examples(work / "examples", images, stems, results, gt, names),
              "config": {k: str(v) if k == "weights" else v for k, v in asdict(cfg).items()},
              "frozen": frozen, "git": git, "versions": tr.versions(), "finished": tr._now()}
    m = parity["metric_parity"]
    checks = {"gates": _check(True, {"dataset": fingerprint["dataset"], "git": git, **frozen}), **gates,
              "predicted_every_val_image": _check(sorted(results) == stems == sorted(r["stem"] for r in store["images"]),
                                                  {"images": len(results), "label_files": len(stems)}),
              "metric_parity": _check(m["max_abs_diff"] <= PARITY_TOL,
                                      {"max_abs_diff": m["max_abs_diff"], "tolerance": PARITY_TOL}),
              "baseline_untouched_after": ex.baseline_check(ctx),
              "fingerprint_unchanged_after": _check(dataset_fingerprint(processed) == fingerprint,
                                                    fingerprint["dataset"]),
              "raw_unchanged_after": _check(evl.raw_hash(ctx.paths.raw_dir) == raw_before, raw_before)}
    tr._write_json(work / "checks.json", checks)
    tr._write_json(work / "parity.json", parity)
    (work / "summary.md").write_text(summarize(checks, parity))
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

def inference_check() -> bool:
    return run_check(ex.Context.load(), Evaluate.load(), Inference.load())