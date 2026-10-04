"""Step 6: first evaluation of the frozen baseline and the validation-selected detector on the test split.

Gates run first: the verified dataset, the untouched baseline, and the frozen selection and checkpoint hashes. The val
phase must then reproduce each model's recorded validation metrics, our own matching, AP and confusion matrix must
agree with Ultralytics, and each model's operating threshold is fixed on val. Only after all that passes are the
models run on test. Each model's test pass is logged immediately before it runs and again when it completes or fails.
Galleries, the bootstrap and the error analysis use exactly the predictions Ultralytics scored.
"""

from tvision import experiments as ex  # imports tvision.train first, which configures Ultralytics

import inspect
import json
import math
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix

from tvision import nucls_data as nd
from tvision import train as tr
from tvision.config import Evaluate
from tvision.draw import draw_boxes, side_by_side
from tvision.explore import _block
from tvision.verify import _hash_files, dataset_fingerprint, label_boxes_px, parse_label_text

STEP = "evaluate"
IOUV = np.linspace(0.5, 0.95, 10)
IOU_EPS = 1e-7
CONF_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
AP_TOL = 1e-4
LOCALIZATION_IOU = 0.1
CM_IOU = inspect.signature(ConfusionMatrix.process_batch).parameters["iou_thres"].default
KINDS = ("misclassified", "fp_duplicate", "fp_localization", "fp_background", "fn_missed")
PRED_KINDS = ("tp", "misclassified", "fp_duplicate", "fp_localization", "fp_background")
IMAGE_SCALE = 2
GRID_COLUMNS = 4
OUTPUTS = ("checks.json", "result.json", "summary.md", "operating_points.json", "baseline", "final", "galleries")
LEDGER = "test_access.jsonl"
CI_SCOPE = ("Slide-level sampling uncertainty conditional on the four observed test hospitals; these are not "
            "hospital-level generalization CIs.")

@dataclass(frozen=True)
class Model:
    name: str
    run_dir: Path
    weights: Path
    sha256: str
    imgsz: int
    max_det: int
    batch: int
    device: str

def _check(passed: bool, detail) -> dict:
    return {"passed": bool(passed), "detail": detail}

def _num(x) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None

def frozen_models(ctx: ex.Context, ev: Evaluate) -> tuple[list[Model], dict]:
    """The frozen baseline and the selected checkpoint, each verified against every hash recorded for it."""
    path = ctx.artifacts / "selection.json"
    if not path.is_file():
        raise RuntimeError(f"{path} not found; Step 5A selection must be frozen before Step 6")
    sha = ex._sha256(path)
    if sha != ev.selection_sha256:
        raise RuntimeError(f"{path} SHA-256 {sha} != selection_sha256 {ev.selection_sha256} in configs/evaluate.yaml")
    selection = json.loads(path.read_text())
    if selection["dataset"] != "three_class":
        raise RuntimeError(f"selected dataset {selection['dataset']!r}: two-class test labels are not derived")
    models = []
    for name, run_dir in (("baseline", ctx.baseline_dir), ("final", Path(selection["run_dir"]))):
        meta = json.loads((run_dir / "run_meta.json").read_text())
        weights = run_dir / "weights" / "best.pt"
        actual = ex._sha256(weights)
        if actual != meta["best"]["sha256"]:
            raise RuntimeError(f"{weights} SHA-256 {actual} != its run_meta.json {meta['best']['sha256']}")
        if name == "final":
            if actual != selection["best_pt_sha256"] or Path(selection["best_pt"]).resolve() != weights.resolve():
                raise RuntimeError(f"{weights} is not the checkpoint frozen in selection.json")
            if meta["config"] != selection["config"]:
                raise RuntimeError(f"{run_dir} run_meta config differs from selection.json")
        c = meta["config"]
        models.append(Model(name, run_dir, weights, actual, c["imgsz"], c["max_det"], c["batch"], c["device"]))
    return models, {"selection_sha256": sha, "experiment": selection["experiment"], "seed": selection["seed"]}

def unscale(boxes: "torch.Tensor", pbatch: dict) -> np.ndarray:
    """Letterboxed xyxy boxes to original image pixels with Ultralytics' own mapping, without clipping."""
    xywh = ops.scale_boxes(pbatch["imgsz"], ops.xyxy2xywh(boxes.clone()), pbatch["ori_shape"],
                           ratio_pad=pbatch["ratio_pad"], xywh=True)
    return ops.xywh2xyxy(xywh).cpu().numpy().astype(np.float64)

def recording_validator(store: dict) -> type[DetectionValidator]:
    """A DetectionValidator that also records every image's post-NMS predictions in original pixels and its
    confusion-matrix increment."""
    class RecordingValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                store["images"].append({"stem": Path(pbatch["im_file"]).stem, "boxes": unscale(pred["bboxes"], pbatch),
                                        "conf": pred["conf"].cpu().numpy().astype(np.float32, copy=False),
                                        "cls": pred["cls"].cpu().numpy().astype(int)})
            super().update_metrics(preds, batch)

        def init_metrics(self, model):
            super().init_metrics(model)
            cm, process = self.confusion_matrix, self.confusion_matrix.process_batch

            def recorded(*args, **kwargs):
                before = cm.matrix.copy()
                process(*args, **kwargs)
                store["cm_deltas"].append((cm.matrix - before).astype(int))
            cm.process_batch = recorded

        def finalize_metrics(self):
            super().finalize_metrics()
            store.update(seen=self.seen, conf=self.args.conf)
    return RecordingValidator

def run_val(model: Model, data_yaml: Path, split: str, out: Path):
    """One Ultralytics val pass of `model` on `split` with its recorded settings; returns metrics and predictions."""
    if out.exists():
        raise RuntimeError(f"{out} already exists")
    store = {"images": [], "cm_deltas": []}
    metrics = YOLO(model.weights).val(validator=recording_validator(store), data=str(data_yaml), split=split,
                                      imgsz=model.imgsz, max_det=model.max_det, batch=model.batch, device=model.device,
                                      project=str(out.parent), name=out.name, exist_ok=False, plots=True)
    if not out.is_dir():
        raise RuntimeError(f"Ultralytics did not write to {out}")
    return metrics, store

def ground_truth(processed: Path, split: str, nc: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(class ids, xyxy boxes in image pixels) per stem, read from the label files and each PNG's own size."""
    gt = {}
    for label in sorted((processed / "labels" / split).glob("*.txt")):
        h, w = nd.image_size(processed / "images" / split / f"{label.stem}.png")
        _, parsed, layout_ok = parse_label_text(label.read_text(), nc)
        if not layout_ok or any(p is None for p in parsed):
            raise RuntimeError(f"invalid label file {label}")
        boxes = label_boxes_px(parsed, w, h)
        gt[label.stem] = (np.array([b[0] for b in boxes], dtype=int),
                          np.array([b[1:] for b in boxes], dtype=np.float64).reshape(-1, 4))
    return gt

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + IOU_EPS)

def match_class_aware(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray, thresholds=IOUV) -> np.ndarray:
    """(N, T) correct matrix: per threshold, one-to-one greedy matching by IoU among same-class pairs."""
    correct = np.zeros((len(pred_cls), len(thresholds)), dtype=bool)
    if not len(pred_cls) or not len(gt_cls):
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(thresholds):
        m = np.array(np.nonzero(iou >= t)).T
        if len(m) > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        if len(m):
            correct[m[:, 1], i] = True
    return correct

def agnostic_match(iou: np.ndarray, thr: float, strict: bool, conf: np.ndarray | None = None) -> np.ndarray:
    """Matched prediction index per GT (-1 if none), one-to-one greedy by IoU regardless of class.

    Without `conf` the order is Ultralytics' (exact IoU ties fall to argsort order); with `conf`, exactly tied
    candidates go to the most confident prediction.
    """
    gi, pi = np.nonzero(iou > thr if strict else iou >= thr)
    gt_match = np.full(iou.shape[0], -1)
    if not len(gi):
        return gt_match
    m = np.stack([gi, pi, iou[gi, pi]], 1)
    if len(m) > 1:
        def order(m):
            return m[:, 2].argsort()[::-1] if conf is None else np.lexsort((-conf[m[:, 1].astype(int)], -m[:, 2]))
        m = m[order(m)]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[order(m)]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    gt_match[m[:, 0].astype(int)] = m[:, 1].astype(int)
    return gt_match

def has_ties(iou: np.ndarray, thr: float) -> bool:
    """Whether any GT or prediction has two competing candidates with exactly equal IoU above `thr`."""
    for lines in (iou, iou.T):
        for line in lines:
            v = line[line > thr]
            if len(v) != len(np.unique(v)):
                return True
    return False

def match_stats(records: list[dict], gt: dict) -> list[dict]:
    out = []
    for r in records:
        gc, gb = gt[r["stem"]]
        out.append({"stem": r["stem"], "tp": match_class_aware(r["cls"], gc, box_iou(gb, r["boxes"])),
                    "conf": r["conf"], "cls": r["cls"], "gt_cls": gc})
    return out

def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP of the precision envelope (COCO-style)."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))

def ap_table(stats: list[dict], nc: int) -> np.ndarray:
    """(nc, 10) AP at IoU 0.50:0.95 over the pooled images; NaN for classes without GT."""
    tp = np.concatenate([s["tp"] for s in stats])
    conf = np.concatenate([s["conf"] for s in stats])
    pred = np.concatenate([s["cls"] for s in stats])
    target = np.concatenate([s["gt_cls"] for s in stats])
    order = np.argsort(-conf)
    tp, pred = tp[order], pred[order]
    ap = np.full((nc, len(IOUV)), np.nan)
    for c in range(nc):
        n_l = int((target == c).sum())
        if not n_l:
            continue
        hit = tp[pred == c]
        if not len(hit):
            ap[c] = 0.0
            continue
        tpc, fpc = hit.cumsum(0), (1 - hit).cumsum(0)
        recall, precision = tpc / (n_l + 1e-16), tpc / (tpc + fpc)
        ap[c] = [average_precision(recall[:, j], precision[:, j]) for j in range(len(IOUV))]
    return ap

def ap_summary(ap: np.ndarray, names: list[str], headline_ids: list[int]) -> dict:
    per = {n: {"mAP50": _num(ap[i, 0]), "mAP75": _num(ap[i, 5]), "mAP50-95": _num(ap[i].mean())}
           for i, n in enumerate(names)}
    return per | {"headline": {"mAP50": _num(ap[headline_ids, 0].mean()), "mAP75": _num(ap[headline_ids, 5].mean()),
                               "mAP50-95": _num(ap[headline_ids].mean())}}

def confusion(records: list[dict], gt: dict, nc: int, conf: float, iou_thres: float) -> np.ndarray:
    """[predicted, true] counts with background last, as Ultralytics' ConfusionMatrix builds them."""
    m = np.zeros((nc + 1, nc + 1), dtype=int)
    for r in records:
        gc, gb = gt[r["stem"]]
        keep = r["conf"] > conf
        pc = r["cls"][keep]
        gm = agnostic_match(box_iou(gb, r["boxes"][keep]), iou_thres, strict=True)
        for g, d in enumerate(gm):
            m[pc[d] if d >= 0 else nc, gc[g]] += 1
        for d in sorted(set(range(len(pc))) - set(gm[gm >= 0].tolist())):
            m[pc[d], nc] += 1
    return m

def scoring_checks(prefix: str, metrics, store: dict, gt: dict, stats: list[dict], nc: int) -> dict:
    """Recorded predictions are complete, and our AP and confusion matrix reproduce Ultralytics' from them."""
    stems = [r["stem"] for r in store["images"]]
    complete = sorted(stems) == sorted(gt) and store["seen"] == len(stems)
    box = metrics.box
    ours = ap_table(stats, nc)
    present = [int(c) for c in box.ap_class_index]
    instances = np.bincount(np.concatenate([g[0] for g in gt.values()]), minlength=nc)
    same_classes = present == [c for c in range(nc) if instances[c]]
    diff = float(np.abs(ours[present] - box.all_ap).max()) if same_classes else None
    same_instances = np.array_equal(instances, metrics.nt_per_class)
    return {
        f"{prefix}/predictions_complete": _check(complete, {"images": len(stems), "label_files": len(gt),
                                                             "seen": store["seen"]}),
        f"{prefix}/ap_matches_ultralytics": _check(same_classes and same_instances and diff is not None
                                                   and diff <= AP_TOL,
                                                   {"max_abs_ap_diff": diff, "tolerance": AP_TOL,
                                                    "instances": instances.tolist(), "same_classes": same_classes,
                                                    "same_instances": same_instances}),
        f"{prefix}/confusion_matches_ultralytics": confusion_check(store, gt, np.asarray(metrics.confusion_matrix.matrix),
                                                                   nc),
    }

def confusion_check(store: dict, gt: dict, total: np.ndarray, nc: int) -> dict:
    """Per image, our confusion matrix must equal Ultralytics' increment exactly, unless the image has exact IoU
    ties among competing candidates (multi-label NMS emits the same box once per class), where Ultralytics'
    assignment depends on argsort tie order. Tied images must still agree on missed GTs per true class (the
    background row), i.e. on every GT's matched/missed status. The increments must sum to Ultralytics' matrix."""
    deltas, conf = store["cm_deltas"], store["conf"]
    exact, fn_row, moved, tied = [], [], {}, 0
    for r, d in zip(store["images"], deltas):
        gc, gb = gt[r["stem"]]
        ours = confusion([r], gt, nc, conf, CM_IOU)
        ties = has_ties(box_iou(gb, r["boxes"][r["conf"] > conf]), CM_IOU)
        tied += ties
        if np.array_equal(ours, d):
            continue
        if not ties:
            exact.append(r["stem"])
        elif not np.array_equal(ours[nc], d[nc]):
            fn_row.append(r["stem"])
        else:
            moved[r["stem"]] = int(np.abs(ours - d).sum())
    summed = len(deltas) == len(store["images"]) and np.array_equal(sum(deltas), total.astype(int))
    return _check(summed and not exact and not fn_row,
                  {"conf": conf, "iou": CM_IOU, "images_with_ties": tied, "increments_sum_to_matrix": summed,
                   "mismatched_without_ties": exact, "mismatched_missed_gts": fn_row,
                   "tie_order_differences": moved})

def operating_prf(records: list[dict], gt: dict, conf: float, iou: float, nc: int) -> dict:
    """Per-class TP/FP/FN, precision, recall and F1 after keeping predictions with confidence > `conf`."""
    tp, n_pred, n_gt = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    for r in records:
        gc, gb = gt[r["stem"]]
        keep = r["conf"] > conf
        pc = r["cls"][keep]
        correct = match_class_aware(pc, gc, box_iou(gb, r["boxes"][keep]), [iou])[:, 0]
        tp += np.bincount(pc[correct], minlength=nc)
        n_pred += np.bincount(pc, minlength=nc)
        n_gt += np.bincount(gc, minlength=nc)
    return {c: _prf(int(tp[c]), int(n_pred[c]), int(n_gt[c])) for c in range(nc)}

def _prf(tp: int, n_pred: int, n_gt: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}

def operating_threshold(records: list[dict], gt: dict, headline_ids: list[int], iou: float, nc: int) -> dict:
    """The confidence on CONF_GRID maximising the mean headline F1 at IoU `iou`; ties go to the lowest."""
    pre = []
    for r in records:
        gc, gb = gt[r["stem"]]
        keep = r["conf"] > CONF_GRID[0]
        pre.append((gc, r["cls"][keep], r["conf"][keep], box_iou(gb, r["boxes"][keep])))
    curve = []
    for t in CONF_GRID:
        tp, n_pred, n_gt = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
        for gc, pc, conf, m in pre:
            k = conf > t
            correct = match_class_aware(pc[k], gc, m[:, k], [iou])[:, 0]
            tp += np.bincount(pc[k][correct], minlength=nc)
            n_pred += np.bincount(pc[k], minlength=nc)
            n_gt += np.bincount(gc, minlength=nc)
        curve.append(float(np.mean([_prf(int(tp[c]), int(n_pred[c]), int(n_gt[c]))["f1"] for c in headline_ids])))
    best = int(np.argmax(curve))
    return {"threshold": float(CONF_GRID[best]), "mean_headline_f1": curve[best], "iou": iou,
            "grid": CONF_GRID.tolist(), "curve": curve}

def errors(records: list[dict], gt: dict, conf: float, iou: float) -> pd.DataFrame:
    """Every kept prediction and every GT box at the operating point, by kind (class-agnostic matching, IoU >= iou;
    exactly tied candidates go to the most confident prediction).

    An unmatched prediction is fp_duplicate if its best IoU is >= iou (that GT went to another prediction),
    fp_localization if it is in [LOCALIZATION_IOU, iou), else fp_background. `gt_index` is the matched GT, the nearest
    GT for unmatched predictions, or the missed GT itself; `pred_index` is the prediction, or the nearest kept
    prediction for fn_missed. `iou` is between those two (0 when there is none).
    """
    rows = []
    for r in records:
        gc, gb = gt[r["stem"]]
        keep = np.flatnonzero(r["conf"] > conf)
        pc, pb, pconf = r["cls"][keep], r["boxes"][keep], r["conf"][keep]
        m = box_iou(gb, pb)
        gm = agnostic_match(m, iou, strict=False, conf=pconf)
        matched = {int(d): g for g, d in enumerate(gm) if d >= 0}
        for d in range(len(pc)):
            if d in matched:
                g = matched[d]
                kind = "tp" if pc[d] == gc[g] else "misclassified"
            else:
                g = int(m[:, d].argmax()) if len(gc) and m[:, d].max() > 0 else -1
                best = m[g, d] if g >= 0 else 0.0
                kind = ("fp_duplicate" if best >= iou else "fp_localization" if best >= LOCALIZATION_IOU
                        else "fp_background")
            rows.append(_row(r["stem"], kind, int(keep[d]), g, pc[d], gc, pconf[d], m, d, pb[d], gb))
        for g in np.flatnonzero(gm < 0):
            d = int(m[g].argmax()) if len(pc) and m[g].max() > 0 else -1
            rows.append(_row(r["stem"], "fn_missed", int(keep[d]) if d >= 0 else -1, int(g), pc[d] if d >= 0 else -1,
                             gc, pconf[d] if d >= 0 else np.nan, m, d, pb[d] if d >= 0 else None, gb))
    return pd.DataFrame(rows)

def _row(stem, kind, pred_index, g, pred_cls, gc, conf, m, d, pbox, gb) -> dict:
    pbox = [np.nan] * 4 if pbox is None else list(pbox)
    gbox = list(gb[g]) if g >= 0 else [np.nan] * 4
    return {"stem": stem, "kind": kind, "pred_index": pred_index, "gt_index": g, "pred_cls": int(pred_cls),
            "gt_cls": int(gc[g]) if g >= 0 else -1, "conf": float(conf),
            "iou": float(m[g, d]) if g >= 0 and d >= 0 else 0.0,
            **dict(zip(("px1", "py1", "px2", "py2"), map(float, pbox))),
            **dict(zip(("gx1", "gy1", "gx2", "gy2"), map(float, gbox)))}

def cm_from_errors(err: pd.DataFrame, nc: int) -> np.ndarray:
    m = np.zeros((nc + 1, nc + 1), dtype=int)
    for r in err.itertuples():
        pred = r.pred_cls if r.kind in PRED_KINDS else nc
        true = r.gt_cls if r.kind in ("tp", "misclassified", "fn_missed") else nc
        m[pred, true] += 1
    return m

def error_counts(err: pd.DataFrame, names: list[str], ambiguous: int) -> dict:
    """Counts by kind and class; anything involving an ambiguous prediction or GT is counted separately."""
    is_pred = err["kind"].isin(PRED_KINDS)
    has_gt = err["kind"].isin(["tp", "misclassified", "fn_missed"])
    amb = (is_pred & (err["pred_cls"] == ambiguous)) | (has_gt & (err["gt_cls"] == ambiguous))
    cls = np.where(err["kind"] == "fn_missed", err["gt_cls"], err["pred_cls"])
    table = pd.crosstab(err["kind"], pd.Series(cls, index=err.index).map(dict(enumerate(names))))
    mis = err[err["kind"] == "misclassified"]
    pairs = (mis["gt_cls"].map(dict(enumerate(names))) + " -> " + mis["pred_cls"].map(dict(enumerate(names))))
    return {"by_kind_and_class": {k: {n: int(v) for n, v in row.items()} for k, row in table.iterrows()},
            "class_is": "predicted class, except the GT class for fn_missed",
            "misclassified_true_to_pred": {k: int(v) for k, v in pairs.value_counts().sort_index().items()},
            "involving_ambiguous": {k: int(v) for k, v in err.loc[amb, "kind"].value_counts().sort_index().items()},
            "headline_only": {k: int(v) for k, v in err.loc[~amb, "kind"].value_counts().sort_index().items()}}

def headline_maps(ap: np.ndarray, ids: list[int]) -> dict:
    return {"mAP50": float(ap[ids, 0].mean()), "mAP50-95": float(ap[ids].mean())}

def slide_bootstrap(stats: dict[str, list[dict]], slide_of: dict[str, str], ids: list[int], nc: int,
                    resamples: int, seed: int, level: float) -> dict:
    """Paired bootstrap over whole slides: every resample draws the same slides for every model."""
    names = list(stats)
    by_stem = {n: {s["stem"]: s for s in stats[n]} for n in names}
    stems = sorted(by_stem[names[0]])
    if any(sorted(by_stem[n]) != stems for n in names):
        raise RuntimeError("models were not scored on the same images")
    slides = sorted({slide_of[s] for s in stems})
    groups = {sl: [s for s in stems if slide_of[s] == sl] for sl in slides}
    rng = np.random.default_rng(seed)
    draws = {n: {"mAP50": [], "mAP50-95": []} for n in names}
    skipped = 0
    for _ in range(resamples):
        chosen = [s for k in rng.choice(len(slides), size=len(slides), replace=True) for s in groups[slides[k]]]
        values = {n: headline_maps(ap_table([by_stem[n][s] for s in chosen], nc), ids) for n in names}
        if not all(math.isfinite(v) for d in values.values() for v in d.values()):
            skipped += 1
            continue
        for n in names:
            for k, v in values[n].items():
                draws[n][k].append(v)
    q = [50 * (1 - level), 50 * (1 + level)]
    point = {n: headline_maps(ap_table(stats[n], nc), ids) for n in names}
    out = {n: {k: {"point": point[n][k], "ci": np.percentile(v, q).tolist()} for k, v in draws[n].items()}
           for n in names}
    a, b = names
    out["difference"] = {}
    for k in ("mAP50", "mAP50-95"):
        d = np.array(draws[b][k]) - np.array(draws[a][k])
        out["difference"][k] = {"point": point[b][k] - point[a][k], "ci": np.percentile(d, q).tolist(),
                                "fraction_positive": float((d > 0).mean())}
    return out | {"difference_is": f"{b} - {a}", "resamples": resamples, "used": resamples - skipped, "seed": seed,
                  "level": level, "slides": len(slides), "images": len(stems), "scope": CI_SCOPE}

def per_group(stats: list[dict], group_of: dict[str, str], gt: dict, names: list[str], ids: list[int]) -> dict:
    out = {}
    for g in sorted(set(group_of[s["stem"]] for s in stats)):
        sub = [s for s in stats if group_of[s["stem"]] == g]
        counts = np.bincount(np.concatenate([gt[s["stem"]][0] for s in sub]), minlength=len(names))
        out[g] = {"images": len(sub), "instances": dict(zip(names, counts.tolist())),
                  **ap_summary(ap_table(sub, len(names)), names, ids)}
    return out

def ann_ids(processed: Path, split: str, gt: dict) -> tuple[dict[str, list[int]], dict]:
    """Source ann_id per label line: line order equals CSV row order (Step 4 invariant), checked per FOV by class."""
    ann = pd.read_csv(processed / "annotations.csv", usecols=["stem", "ann_id", "split", "class_id", "status"])
    kept = ann[(ann["split"] == split) & (ann["status"] == "kept")].sort_values(["stem", "ann_id"])
    groups = {s: g for s, g in kept.groupby("stem")}
    mismatched = [s for s, (gc, _) in gt.items()
                  if (groups[s]["class_id"].tolist() if s in groups else []) != gc.tolist()]
    ids = {s: groups[s]["ann_id"].tolist() if s in groups else [] for s in gt}
    return ids, _check(not mismatched and set(groups) <= set(gt), {"fovs": len(gt), "mismatched": mismatched})

def _short(stem: str) -> str:
    slide, _, rest = stem.partition("_id-")
    return f"{slide.removeprefix('TCGA-')} {rest[:6]}"

def captioned(img: np.ndarray, lines: list[str], min_width: int = 200) -> np.ndarray:
    if img.shape[1] < min_width:
        img = cv2.copyMakeBorder(img, 0, 0, 0, min_width - img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    strip = np.full((14 * len(lines) + 6, img.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (3, 14 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])

def grid(tiles: list[np.ndarray], columns: int = GRID_COLUMNS) -> np.ndarray:
    rows = [side_by_side(*tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    return np.vstack(rows)

def _boxes(boxes, classes, names, offset=(0, 0), scale=1) -> list[dict]:
    return [{"x1": (b[0] - offset[0]) * scale, "y1": (b[1] - offset[1]) * scale, "x2": (b[2] - offset[0]) * scale,
             "y2": (b[3] - offset[1]) * scale, "class_name": names[int(c)]} for b, c in zip(boxes, classes)]

def error_tile(img: np.ndarray, row, gt_entry, names: list[str], ids: list[int], context: int, scale: int) -> np.ndarray:
    """Crop around the error with every GT box thin, the focus GT 2 px and the focus prediction 3 px."""
    h, w = img.shape[:2]
    pbox = [row.px1, row.py1, row.px2, row.py2]
    gbox = [row.gx1, row.gy1, row.gx2, row.gy2]
    pts = np.array([b for b in (pbox, gbox) if not np.isnan(b[0])])
    x0, y0 = max(int(pts[:, 0].min()) - context, 0), max(int(pts[:, 1].min()) - context, 0)
    x1, y1 = min(int(math.ceil(pts[:, 2].max())) + context, w), min(int(math.ceil(pts[:, 3].max())) + context, h)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    gc, gb = gt_entry
    crop = draw_boxes(crop, _boxes(gb, gc, names, (x0, y0), scale), thickness=1)
    if row.gt_index >= 0:
        crop = draw_boxes(crop, _boxes([gbox], [row.gt_cls], names, (x0, y0), scale), thickness=2)
    if row.pred_cls >= 0:
        crop = draw_boxes(crop, _boxes([pbox], [row.pred_cls], names, (x0, y0), scale), thickness=3)
    gt_text = f"GT {names[row.gt_cls]} ann {ids[row.gt_index]}" if row.gt_index >= 0 else "GT -"
    pred_text = f"pred {names[row.pred_cls]} {row.conf:.2f}" if row.pred_cls >= 0 else "pred -"
    return captioned(crop, [row.kind, _short(row.stem), gt_text, pred_text, f"IoU {row.iou:.2f}"])

def error_galleries(out: Path, err: pd.DataFrame, images: Path, gt: dict, ids: dict, names: list[str],
                    gallery: dict) -> dict:
    counts = {}
    for kind in KINDS:
        rows = err[err["kind"] == kind].reset_index(drop=True)
        rng = np.random.default_rng(gallery["seed"])
        pick = sorted(rng.choice(len(rows), size=min(gallery["per_kind"], len(rows)), replace=False).tolist())
        folder = out / kind
        folder.mkdir(parents=True)
        tiles = []
        for k, i in enumerate(pick):
            row = rows.iloc[i]
            img = cv2.imread(str(images / f"{row.stem}.png"))
            tile = error_tile(img, row, gt[row.stem], names, ids[row.stem], gallery["context_px"], gallery["scale"])
            cv2.imwrite(str(folder / f"{k:02d}_{row.stem}.png"), tile)
            tiles.append(tile)
        if tiles:
            cv2.imwrite(str(folder / "grid.png"), grid(tiles))
        rows.iloc[pick].to_csv(folder / "items.csv", index=False)
        counts[kind] = len(pick)
    return counts

def image_f1(err: pd.DataFrame, gt: dict, ids: list[int]) -> pd.Series:
    """Per-image headline F1 at the operating point: 2 TP / (headline predictions + headline GT)."""
    f1 = {}
    for stem, (gc, _) in gt.items():
        e = err[err["stem"] == stem]
        tp = int(((e["kind"] == "tp") & e["pred_cls"].isin(ids)).sum())
        n_pred = int((e["kind"].isin(PRED_KINDS) & e["pred_cls"].isin(ids)).sum())
        n_gt = int(np.isin(gc, ids).sum())
        if n_pred + n_gt:
            f1[stem] = 2 * tp / (n_pred + n_gt)
    return pd.Series(f1).sort_index(kind="stable")

def image_galleries(out: Path, images: Path, gt: dict, records: dict, thresholds: dict, f1: dict[str, pd.Series],
                    names: list[str], n: int) -> dict:
    """GT | baseline | final for the best, median and worst FOVs by the final model's per-image headline F1."""
    out.mkdir(parents=True)
    ranked = f1["final"].sort_values(ascending=False, kind="stable")
    mid = len(ranked) // 2 - n // 2
    chosen = {"best": ranked.index[:n], "median": ranked.index[mid:mid + n], "worst": ranked.index[-n:]}
    by_stem = {m: {r["stem"]: r for r in recs} for m, recs in records.items()}
    for group, stems in chosen.items():
        for k, stem in enumerate(stems):
            raw = cv2.imread(str(images / f"{stem}.png"))
            h, w = raw.shape[:2]
            img = cv2.resize(raw, None, fx=IMAGE_SCALE, fy=IMAGE_SCALE, interpolation=cv2.INTER_NEAREST)
            gc, gb = gt[stem]
            panels = [captioned(draw_boxes(img, _boxes(gb, gc, names, scale=IMAGE_SCALE)), ["GT", _short(stem)])]
            for m in ("baseline", "final"):
                r = by_stem[m][stem]
                keep = r["conf"] > thresholds[m]
                boxes = np.clip(r["boxes"][keep], 0, [w, h, w, h])
                drawn = draw_boxes(img, _boxes(boxes, r["cls"][keep], names, scale=IMAGE_SCALE), thickness=2)
                score = f1[m].get(stem, np.nan)
                panels.append(captioned(drawn, [f"{m} conf > {thresholds[m]:.2f}", f"headline F1 {score:.3f}"]))
            cv2.imwrite(str(out / f"{group}_{k}_{stem}.png"), side_by_side(*panels))
    return {g: list(s) for g, s in chosen.items()}

def predictions_csv(path: Path, records: list[dict], shapes: dict, names: list[str], threshold: float) -> None:
    frames = []
    for r in records:
        h, w = shapes[r["stem"]]
        b = np.clip(r["boxes"], 0, [w, h, w, h])
        frames.append(pd.DataFrame({"stem": r["stem"], "x1": b[:, 0], "y1": b[:, 1], "x2": b[:, 2], "y2": b[:, 3],
                                    "conf": r["conf"], "class_name": [names[c] for c in r["cls"]],
                                    "above_operating_threshold": r["conf"] > threshold}))
    pd.concat(frames).to_csv(path, index=False, float_format="%.4f")

def raw_hash(raw: Path) -> str:
    return _hash_files(raw, sorted(p for p in raw.rglob("*") if p.is_file()))["sha256"]

def ledger(root: Path, event: str, **fields) -> None:
    """Append one event to the append-only test-access ledger."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / LEDGER).open("a") as f:
        f.write(json.dumps({"time": tr._now(), "event": event, **fields}) + "\n")

def test_passes(root: Path) -> dict[str, int]:
    """Model test passes started so far, per model; overall evaluation events are not counted."""
    path = root / LEDGER
    events = [json.loads(l) for l in path.read_text().splitlines()] if path.is_file() else []
    return dict(Counter(e["model"] for e in events if e["event"] == "test_pass_started"))

def logged_test_passes(root: Path, models: list[Model], run_one: Callable[[Model], tuple]) -> dict[str, tuple]:
    """Run each model on test, logging it immediately before it runs and again when it completes or fails."""
    out = {}
    for m in models:
        ledger(root, "test_pass_started", model=m.name, sha256=m.sha256)
        try:
            out[m.name] = run_one(m)
        except BaseException as e:
            ledger(root, "test_pass_failed", model=m.name, sha256=m.sha256, error=repr(e))
            raise
        ledger(root, "test_pass_completed", model=m.name, sha256=m.sha256)
    return out

def val_phase(model: Model, ctx: ex.Context, work: Path, gt: dict, ev: Evaluate) -> tuple[dict, dict]:
    classes, nc = ctx.classes, len(ctx.classes.names)
    ids = [classes.class_id(n) for n in classes.headline]
    metrics, store = run_val(model, ctx.paths.processed_dir / "data.yaml", "val", work / model.name / "val")
    table, headline = tr.per_class_val(metrics, classes)
    recorded = pd.DataFrame(json.loads((model.run_dir / "val_best.json").read_text())["per_class"]).set_index("Class")
    ours, theirs = table[tr.METRIC_COLUMNS].to_numpy(float), recorded.loc[table.index, tr.METRIC_COLUMNS].to_numpy(float)
    prefix = f"{model.name}/val"
    checks = {f"{prefix}/reproduces_val_best": _check(np.allclose(ours, theirs),
                                                      {"max_abs_diff": float(np.abs(ours - theirs).max()),
                                                       "recorded": str(model.run_dir / "val_best.json")})}
    checks |= scoring_checks(prefix, metrics, store, gt, match_stats(store["images"], gt), nc)
    op = operating_threshold(store["images"], gt, ids, ev.match_iou, nc)
    prf = operating_prf(store["images"], gt, op["threshold"], ev.match_iou, nc)
    op["val_per_class"] = {classes.names[c]: v for c, v in prf.items()}
    op["val_headline_mean"] = {k: float(np.mean([prf[c][k] for c in ids])) for k in ("precision", "recall", "f1")}
    return checks, op | {"val_table": table.reset_index().to_dict("records"), "val_headline": headline}

def test_phase(model: Model, ctx: ex.Context, work: Path, gt: dict, op: dict, ev: Evaluate, groups: dict) -> tuple:
    classes, nc = ctx.classes, len(ctx.classes.names)
    names, ids = classes.names, [classes.class_id(n) for n in classes.headline]
    metrics, store = run_val(model, ctx.paths.processed_dir / "data.yaml", "test", work / model.name / "test")
    table, headline = tr.per_class_val(metrics, classes)
    stats = match_stats(store["images"], gt)
    checks = scoring_checks(f"{model.name}/test", metrics, store, gt, stats, nc)
    t = op["threshold"]
    err = errors(store["images"], gt, t, ev.match_iou)
    prf = operating_prf(store["images"], gt, t, ev.match_iou, nc)
    cm = cm_from_errors(err, nc)
    totals = cm.sum(0)
    recorded = json.loads((model.run_dir / "val_best.json").read_text())
    result = {
        "weights": str(model.weights), "sha256": model.sha256,
        "settings": {"imgsz": model.imgsz, "max_det": model.max_det, "batch": model.batch, "device": model.device,
                     "val_conf": store["conf"]},
        "test_per_class": table.reset_index().to_dict("records"),
        "test_headline_mean": headline,
        "ultralytics_pr_note": "Box-P/R/F1 are read at the confidence maximising the mean F1 over all classes on "
                               "test itself (Ultralytics' ap_per_class), so they are test-derived.",
        "ultralytics_confusion_matrix": {"conf": store["conf"], "iou": CM_IOU,
                                         "counts": np.asarray(metrics.confusion_matrix.matrix).astype(int).tolist()},
        "operating_point": {
            "threshold": t, "chosen_on": "val", "iou": ev.match_iou,
            "test_per_class": {names[c]: v for c, v in prf.items()},
            "test_headline_mean": {k: float(np.mean([prf[c][k] for c in ids])) for k in ("precision", "recall", "f1")},
            "val_per_class": op["val_per_class"], "val_headline_mean": op["val_headline_mean"],
            "confusion_matrix": {"labels": [*names, "background"], "rows": "predicted", "columns": "true",
                                 "counts": cm.tolist(),
                                 "normalized_per_true_class": (cm / np.where(totals == 0, 1, totals)).round(4).tolist()},
            "errors": error_counts(err, names, classes.class_id("ambiguous")),
        },
        "per_hospital": per_group(stats, groups["hospital"], gt, names, ids),
        "val_to_test": {"val": {"per_class": recorded["per_class"], "headline_mean": recorded["headline_mean"]},
                        "test": {"per_class": table.reset_index().to_dict("records"), "headline_mean": headline}},
        "own_ap": ap_summary(ap_table(stats, nc), names, ids),
    }
    return checks, result, store["images"], stats, err

def compare_models(results: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for label in [*names, "headline mean"]:
        for metric in ("mAP50", "mAP75", "mAP50-95"):
            vals = {}
            for m in ("baseline", "final"):
                if label == "headline mean":
                    vals[m] = results[m]["test_headline_mean"][metric]
                else:
                    vals[m] = next(r for r in results[m]["test_per_class"] if r["Class"] == label)[metric]
            rows.append({"class": label, "metric": metric, **vals, "delta": vals["final"] - vals["baseline"]})
    return pd.DataFrame(rows).set_index(["class", "metric"])

def summarize(checks: dict, result: dict | None, operating: dict, frozen: dict, passes: dict | None, names) -> str:
    sections = [_block("Frozen models", json.dumps(frozen, indent=2)), _block("Checks", json.dumps(checks, indent=2))]
    sections.append(_block("Operating thresholds (chosen on val; mean tumor/non_tumor F1 at IoU 0.5)",
                           json.dumps({m: {"threshold": o["threshold"], "val_headline_mean": o["val_headline_mean"]}
                                       for m, o in operating.items()}, indent=2)))
    if result is None:
        return "# Step 6 evaluation: val phase only (--check; no test pass)\n\n" + "\n".join(sections)
    sections.insert(0, _block("Model test passes started, per model (test_access.jsonl)", json.dumps(passes)))
    for m in ("baseline", "final"):
        r = result["models"][m]
        table = pd.DataFrame(r["test_per_class"]).set_index("Class")
        sections.append(_block(f"{m}: test per class (Ultralytics; P/R at its test-derived max-F1 confidence)",
                               pd.concat([table, pd.DataFrame([r["test_headline_mean"]], index=["headline mean"])])
                               .round(4)))
        op = r["operating_point"]
        prf = pd.DataFrame({"val": {f"{c} {k}": v[k] for c, v in op["val_per_class"].items()
                                    for k in ("precision", "recall", "f1")},
                            "test": {f"{c} {k}": v[k] for c, v in op["test_per_class"].items()
                                     for k in ("precision", "recall", "f1")}})
        sections.append(_block(f"{m}: P/R/F1 at the val-chosen threshold {op['threshold']:.2f} (IoU 0.5)",
                               prf.round(4)))
        cm = pd.DataFrame(op["confusion_matrix"]["normalized_per_true_class"], index=[*names, "background"],
                          columns=[*names, "background"])
        sections.append(_block(f"{m}: confusion matrix at the operating point (rows predicted, columns true, "
                               "normalized per true class)", cm))
        sections.append(_block(f"{m}: errors at the operating point", json.dumps(op["errors"], indent=2)))
        hosp = pd.DataFrame({h: {"images": v["images"], **{f"headline {k}": v["headline"][k] for k in v["headline"]}}
                             for h, v in r["per_hospital"].items()}).T
        sections.append(_block(f"{m}: per test hospital (own AP; descriptive)", hosp.round(4)))
        vt = pd.DataFrame({s: r["val_to_test"][s]["headline_mean"] for s in ("val", "test")}).T
        sections.append(_block(f"{m}: headline mean, val -> test", vt.round(4)))
    sections.append(_block("Before/after on test (final - baseline)",
                           pd.DataFrame(result["comparison"]).set_index(["class", "metric"]).round(4)))
    sections.append(_block(f"Slide-level paired bootstrap. Scope: {CI_SCOPE}",
                           json.dumps(result["bootstrap"], indent=2)))
    sections.append(_block("Galleries", "Error tiles: every GT box thin (1 px), the focus GT 2 px, the focus prediction "
                                        "3 px; colours by class (draw.py). Image panels: GT | baseline | final at each "
                                        "model's val-chosen threshold.\n" + json.dumps(result["galleries"], indent=2)))
    return "# Step 6 evaluation on the held-out test split\n\n" + "\n".join(sections)

def run_evaluation(ctx: ex.Context, ev: Evaluate, check_only: bool = False) -> bool:
    root = ctx.paths.artifacts_dir / STEP
    if not check_only and (existing := [o for o in OUTPUTS if (root / o).exists()]):
        raise RuntimeError(f"{root} already holds {existing}; the test evaluation is made once")
    git = tr.git_state()
    if git["dirty"]:
        raise RuntimeError("git working tree is dirty; commit before evaluating so the run is reproducible")
    fingerprint = tr.gate(ctx.paths.processed_dir, ctx.paths.artifacts_dir / "verify")
    before = ex.require_baseline(ctx)
    models, frozen = frozen_models(ctx, ev)
    for m in models:
        tr.check_device(m.device)
    raw_before = raw_hash(ctx.paths.raw_dir)
    print(f"[PASS] gates: dataset {fingerprint['dataset'][:16]}…, selection {frozen['selection_sha256'][:16]}…, "
          + ", ".join(f"{m.name} {m.sha256[:16]}…" for m in models))
    work = root / ("check" if check_only else ".tmp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    classes, nc = ctx.classes, len(ctx.classes.names)
    names, ids = classes.names, [classes.class_id(n) for n in classes.headline]
    checks = {"gates": _check(True, {"dataset": fingerprint["dataset"], "baseline_manifest":
                                     before["detail"]["manifest_sha256"], **frozen, "git": git})}
    gt_val = ground_truth(ctx.paths.processed_dir, "val", nc)
    operating = {}
    for m in models:
        c, operating[m.name] = val_phase(m, ctx, work, gt_val, ev)
        checks |= c
    tr._write_json(work / "operating_points.json", operating)

    def finish(result: dict | None, passes: dict | None) -> bool:
        checks["baseline_untouched_after"] = ex.baseline_check(ctx)
        checks["fingerprint_unchanged_after"] = _check(dataset_fingerprint(ctx.paths.processed_dir) == fingerprint,
                                                       fingerprint["dataset"])
        checks["raw_unchanged_after"] = _check(raw_hash(ctx.paths.raw_dir) == raw_before, raw_before)
        tr._write_json(work / "checks.json", checks)
        if result is not None:
            tr._write_json(work / "result.json", result)
        (work / "summary.md").write_text(summarize(checks, result, operating, frozen, passes, names))
        for k, c in checks.items():
            print(f"[{'PASS' if c['passed'] else 'FAIL'}] {k}")
        return all(c["passed"] for c in checks.values())

    if check_only or not all(c["passed"] for c in checks.values()):
        ok = finish(None, None)
        if not check_only:
            print("Val phase failed; no model was run on test.")
        print(f"Report: {work / 'summary.md'}")
        return ok and check_only

    ledger(root, "evaluation_started", git=git, models={m.name: m.sha256 for m in models})
    try:
        gt_test = ground_truth(ctx.paths.processed_dir, "test", nc)
        ann, checks["ann_ids_map"] = ann_ids(ctx.paths.processed_dir, "test", gt_test)
        splits = pd.read_csv(ctx.paths.processed_dir / "splits.csv").set_index("stem")
        groups = {g: splits.loc[list(gt_test), g].to_dict() for g in ("slide", "hospital")}
        outputs = logged_test_passes(root, models,
                                     lambda m: test_phase(m, ctx, work, gt_test, operating[m.name], ev, groups))
        results, records, stats, errs = {}, {}, {}, {}
        for m in models:
            c, results[m.name], records[m.name], stats[m.name], errs[m.name] = outputs[m.name]
            checks |= c
        images = ctx.paths.processed_dir / "images" / "test"
        shapes = {s: nd.image_size(images / f"{s}.png") for s in gt_test}
        galleries = {}
        for m in models:
            predictions_csv(work / m.name / "test_predictions.csv", records[m.name], shapes, names,
                            operating[m.name]["threshold"])
            galleries[m.name] = error_galleries(work / "galleries" / m.name, errs[m.name], images, gt_test, ann,
                                                names, ev.gallery)
        f1 = {m.name: image_f1(errs[m.name], gt_test, ids) for m in models}
        galleries["images"] = image_galleries(work / "galleries" / "images", images, gt_test, records,
                                              {m: o["threshold"] for m, o in operating.items()}, f1, names,
                                              ev.gallery["images"])
        b = ev.bootstrap
        result = {"frozen": frozen, "models": results, "galleries": galleries,
                  "comparison": compare_models(results, names).reset_index().to_dict("records"),
                  "bootstrap": slide_bootstrap(stats, groups["slide"], ids, nc, b["resamples"], b["seed"], b["level"]),
                  "config": {"match_iou": ev.match_iou, "gallery": ev.gallery, "bootstrap": ev.bootstrap},
                  "versions": tr.versions(), "git": git, "finished": tr._now()}
        passes = test_passes(root)
        ok = finish(result, passes)
        for entry in sorted(work.iterdir()):
            entry.rename(root / entry.name)
        work.rmdir()
    except BaseException as e:
        ledger(root, "evaluation_failed", error=repr(e))
        raise
    ledger(root, "evaluation_completed", all_checks_passed=ok)
    print(f"Model test passes: {passes}\nReport: {root / 'summary.md'}")
    return ok

def evaluate(check_only: bool = False) -> bool:
    return run_evaluation(ex.Context.load(), Evaluate.load(), check_only)