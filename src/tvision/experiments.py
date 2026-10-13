"""Step 5A: controlled validation experiments against the frozen baseline.

Every experiment is the `control` config in configs/experiments.yaml plus overrides, and is refused if it changes
anything beyond its declared factors relative to its `compare_to` config. The optimizer is pinned to the values the
baseline actually trained with. Runs train on train and validate on val only; best.pt and early stopping follow
the headline (tumor/non_tumor) mAP50-95. Nothing runs unless every baseline artifact matches the committed manifest,
and every run re-checks it when it finishes.
"""

from tvision import train as tr  # sets Ultralytics' config directory before Ultralytics is imported

import copy
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer

from tvision.config import (AUGMENTATION_KEYS, CONFIG_DIR, PROJECT_ROOT, Classes, Paths, Train, TwoClass,
                            load_yaml)
from tvision.draw import draw_boxes, side_by_side
from tvision.explore import _block
from tvision.two_class import filter_lines, gate_two_class, kept_classes
from tvision.verify import SPLITS

STEP5A = "step5a"
BASELINE = "baseline"
DATASETS = ("three_class", "two_class")
SPEC_KEYS = {"baseline_manifest_sha256", "control", "smoke", "experiments"}
EXPERIMENT_KEYS = {"factors", "compare_to", "dataset", "augmentation", "model", "imgsz", "epochs", "patience", "batch",
                   "device", "deterministic", "cache", "max_det"}
FACTOR_KEYS = {
    "appearance": {"augmentation.hsv_h", "augmentation.hsv_s", "augmentation.hsv_v"},
    "geometry": {"augmentation.scale", "augmentation.translate"},
    "imgsz": {"imgsz"},
    "two_class": {"dataset"},
    "batch": {"batch"},
    "model": {"model"},
}
NAME = re.compile(r"[a-z0-9][a-z0-9_]*")
ADAM_FAMILY = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}
ADAM_BETA2 = 0.999  # Ultralytics builds Adam-family optimizers with betas=(momentum, 0.999)
SCHEDULE_KEYS = ("weight_decay", "nbs", "lrf", "cos_lr", "warmup_epochs", "warmup_momentum", "warmup_bias_lr")
IGNORED_FILES = {".DS_Store"}
HEADLINE_CSV = "headline_per_epoch.csv"
REPRODUCES_BASELINE = "a0_control"
REPRO_LOSSES = ("train/box_loss", "train/cls_loss", "train/l1_loss")
REPRO_MAPS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
REPRO_LOSS_RTOL = 0.01
REPRO_MAP_ATOL = 0.005
PLATEAU_FROM_EPOCH = 10
STABILITY_HALF_WINDOW = 2
AP_COLUMNS = ["mAP50", "mAP75", "mAP50-95"]
PR_COLUMNS = ["Box-P", "Box-R", "Box-F1"]

@dataclass(frozen=True)
class Context:
    paths: Paths
    classes: Classes
    spec: dict
    two_class: TwoClass
    train_yaml: Path

    @classmethod
    def load(cls) -> "Context":
        return cls(Paths.load(), Classes.load(), load_yaml("experiments.yaml"), TwoClass.load(), CONFIG_DIR / "train.yaml")

    @property
    def runs(self) -> Path:
        return self.paths.runs_dir / STEP5A

    @property
    def artifacts(self) -> Path:
        return self.paths.artifacts_dir / STEP5A

    @property
    def baseline_dir(self) -> Path:
        return self.paths.runs_dir / BASELINE

    @property
    def manifest(self) -> Path:
        return self.artifacts / "baseline_manifest.json"

    @property
    def comparability_dir(self) -> Path:
        return self.artifacts / "comparability"

    def baseline_targets(self) -> dict[str, Path]:
        return {"runs/baseline": self.baseline_dir, "artifacts/train/baseline": self.paths.artifacts_dir / "train" / BASELINE,
                "configs/train.yaml": self.train_yaml}

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _finite(x) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None

def _check(passed: bool, detail) -> dict:
    return {"passed": bool(passed), "detail": detail}

def hash_targets(targets: dict[str, Path]) -> dict[str, str]:
    """SHA-256 per file, keyed by label/relative path; missing targets contribute no files."""
    files = {}
    for label, root in targets.items():
        if root.is_file():
            files[label] = _sha256(root)
        elif root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name not in IGNORED_FILES:
                    files[f"{label}/{p.relative_to(root).as_posix()}"] = _sha256(p)
    return dict(sorted(files.items()))

def committed_bytes(path: Path) -> bytes:
    rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT, capture_output=True, check=True).stdout

def make_manifest(ctx: Context, committed_train_yaml: bytes) -> str:
    """Write the baseline manifest once and return its SHA-256, the anchor to commit."""
    if ctx.manifest.exists():
        raise RuntimeError(f"{ctx.manifest} already exists; the manifest is written once and never regenerated")
    files = hash_targets(ctx.baseline_targets())
    recorded = json.loads((ctx.baseline_dir / "run_meta.json").read_text())["best"]["sha256"]
    if files.get("runs/baseline/weights/best.pt") != recorded:
        raise RuntimeError("runs/baseline/weights/best.pt does not match the hash recorded in its run_meta.json")
    if ctx.train_yaml.read_bytes() != committed_train_yaml:
        raise RuntimeError(f"{ctx.train_yaml} differs from git HEAD; the baseline config must be the committed one")
    ctx.manifest.parent.mkdir(parents=True, exist_ok=True)
    ctx.manifest.write_text(json.dumps({"n_files": len(files), "files": files}, indent=2))
    return _sha256(ctx.manifest)

def baseline_check(ctx: Context) -> dict:
    """Every baseline file must equal the manifest, and the manifest must equal the committed anchor."""
    anchor = ctx.spec.get("baseline_manifest_sha256")
    if anchor is None or not ctx.manifest.is_file():
        return _check(False, "no baseline manifest or anchor; run `python -m tvision baseline-manifest` and commit "
                             "its SHA-256 as baseline_manifest_sha256 in configs/experiments.yaml")
    sha = _sha256(ctx.manifest)
    if sha != anchor:
        return _check(False, {"manifest_sha256": sha, "anchor": anchor})
    expected = json.loads(ctx.manifest.read_text())["files"]
    actual = hash_targets(ctx.baseline_targets())
    diff = {"added": sorted(actual.keys() - expected.keys()), "removed": sorted(expected.keys() - actual.keys()),
            "changed": sorted(k for k in actual.keys() & expected.keys() if actual[k] != expected[k])}
    return _check(not any(diff.values()), {"manifest_sha256": sha, "files": len(actual), **diff})

def require_baseline(ctx: Context) -> dict:
    check = baseline_check(ctx)
    if not check["passed"]:
        raise RuntimeError(f"baseline_untouched failed: {check['detail']}")
    print(f"[PASS] baseline_untouched: {check['detail']['files']} files match manifest "
          f"{check['detail']['manifest_sha256'][:16]}…")
    return check

def baseline_optimizer(baseline: Path) -> tuple[dict, dict]:
    """The optimizer settings the baseline actually trained with, read from its own files.

    Name, lr0 and betas come from run_meta.json (recorded from the live optimizer). Schedule and regularisation come
    from best.pt's train_args, which are saved after the optimizer is built: with optimizer=auto Ultralytics resets
    warmup_bias_lr to 0 after args.yaml is written, so that is the only difference from args.yaml allowed.
    """
    meta_path, args_path, best = baseline / "run_meta.json", baseline / "args.yaml", baseline / "weights" / "best.pt"
    recorded = json.loads(meta_path.read_text())["optimizer"]
    if recorded["name"] not in ADAM_FAMILY:
        raise RuntimeError(f"baseline optimizer {recorded['name']} is not Adam-family; betas cannot be pinned")
    beta1, beta2 = recorded["momentum"]
    if beta2 != ADAM_BETA2:
        raise RuntimeError(f"baseline beta2 {beta2} != {ADAM_BETA2}; Ultralytics cannot reproduce it")
    effective = torch.load(best, map_location="cpu", weights_only=False)["train_args"]
    saved = yaml.safe_load(args_path.read_text())
    differs = sorted(k for k in SCHEDULE_KEYS if saved[k] != effective[k])
    allowed = ["warmup_bias_lr"] if saved["optimizer"] == "auto" and effective["warmup_bias_lr"] == 0.0 else []
    if unexpected := [k for k in differs if k not in allowed]:
        raise RuntimeError(f"baseline args.yaml and best.pt train_args disagree on {unexpected}")
    optimizer = {"optimizer": recorded["name"], "lr0": recorded["lr0"], "momentum": beta1,
                 **{k: effective[k] for k in SCHEDULE_KEYS}}
    source = {"name_lr0_betas": str(meta_path), "schedule": f"{best} train_args",
              "args_yaml_differs": {k: {"args_yaml": saved[k], "train_args": effective[k]} for k in differs}}
    return optimizer, source

def experiment_config(spec: dict, name: str) -> dict:
    """`control` with the experiment's overrides and its dataset; no seed, name or resolved optimizer."""
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"experiments.yaml must have exactly {sorted(SPEC_KEYS)}, got {sorted(spec)}")
    if name not in spec["experiments"]:
        raise ValueError(f"unknown experiment {name!r}; defined: {sorted(spec['experiments'])}")
    if not NAME.fullmatch(name) or name == "smoke":
        raise ValueError(f"invalid experiment name {name!r}")
    entry = dict(spec["experiments"][name] or {})
    if unknown := sorted(set(entry) - EXPERIMENT_KEYS):
        raise ValueError(f"{name}: unknown keys {unknown}")
    if unknown := sorted(set(entry.get("augmentation", {})) - set(AUGMENTATION_KEYS)):
        raise ValueError(f"{name}: unknown augmentation keys {unknown}")
    cfg = copy.deepcopy(spec["control"])
    cfg["augmentation"] = cfg["augmentation"] | entry.pop("augmentation", {})
    cfg["dataset"] = entry.pop("dataset", "three_class")
    if cfg["dataset"] not in DATASETS:
        raise ValueError(f"{name}: dataset must be one of {DATASETS}")
    entry.pop("factors", None)
    entry.pop("compare_to", None)
    return cfg | entry

def flatten(cfg: dict) -> dict:
    return {**{k: v for k, v in cfg.items() if k != "augmentation"},
            **{f"augmentation.{k}": v for k, v in cfg["augmentation"].items()}}

def check_factors(spec: dict, name: str) -> None:
    """Raise unless the experiment differs from its compare_to config in exactly its declared factors."""
    entry = spec["experiments"].get(name) or {}
    if "factors" not in entry or "compare_to" not in entry:
        raise ValueError(f"{name} must declare factors and compare_to")
    factors, compare_to = entry["factors"], entry["compare_to"]
    if unknown := sorted(set(factors) - set(FACTOR_KEYS)):
        raise ValueError(f"{name}: unknown factors {unknown}; known: {sorted(FACTOR_KEYS)}")
    cfg = flatten(experiment_config(spec, name))
    if compare_to is None:
        if factors or cfg != flatten(copy.deepcopy(spec["control"]) | {"dataset": "three_class"}):
            raise ValueError(f"{name} has no compare_to, so it must be the unchanged control with no factors")
        return
    if compare_to == name or compare_to not in spec["experiments"]:
        raise ValueError(f"{name}: compare_to {compare_to!r} must be another defined experiment")
    other = flatten(experiment_config(spec, compare_to))
    diff = {k for k in cfg.keys() | other.keys() if cfg.get(k) != other.get(k)}
    allowed = set().union(*(FACTOR_KEYS[f] for f in factors))
    if extra := sorted(diff - allowed):
        raise ValueError(f"{name} changes {extra} relative to {compare_to}, outside its declared factors {factors}")
    if unused := [f for f in factors if not diff & FACTOR_KEYS[f]]:
        raise ValueError(f"{name} declares factors {unused} but does not change them relative to {compare_to}")

def build_train(spec: dict, name: str, seed: int, optimizer: dict, smoke: bool = False) -> tuple[Train, str]:
    check_factors(spec, name)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    cfg = experiment_config(spec, name)
    dataset = cfg.pop("dataset")
    if cfg.get("optimizer") != "from_baseline":
        raise ValueError("control must set optimizer: from_baseline")
    cfg |= {"optimizer": optimizer, "seed": seed, "name": f"{STEP5A}/{name}/seed{seed}"}
    if smoke:
        if unknown := sorted(set(spec["smoke"]) - {"epochs", "fraction"}):
            raise ValueError(f"smoke may only set epochs and fraction, got {unknown}")
        cfg |= spec["smoke"]
    return Train.from_dict(cfg), dataset

def run_paths(ctx: Context, name: str, seed: int, smoke: bool) -> tuple[Path, Path]:
    parts = (("smoke",) if smoke else ()) + (name, f"seed{seed}")
    run_dir, report = ctx.runs.joinpath(*parts), ctx.artifacts.joinpath(*parts)
    check_run_dir(ctx, run_dir)
    return run_dir, report

def check_run_dir(ctx: Context, run_dir: Path) -> None:
    resolved, root = run_dir.resolve(), ctx.runs.resolve()
    if resolved == root or not resolved.is_relative_to(root) or resolved.is_relative_to(ctx.baseline_dir.resolve()):
        raise RuntimeError(f"refusing run directory {run_dir}: Step 5A runs must be inside {ctx.runs}")

def check_siblings(run_dir: Path, cfg: Train) -> None:
    """Other seeds of the same experiment must have run the same config, so an edited experiment is refused."""
    mine = {k: v for k, v in asdict(cfg).items() if k not in ("seed", "name")}
    for meta in sorted(run_dir.parent.glob("seed*/run_meta.json")):
        theirs = {k: v for k, v in json.loads(meta.read_text())["config"].items() if k not in ("seed", "name")}
        if theirs != mine:
            raise RuntimeError(f"{meta.parent} ran a different config; give the changed experiment a new name")

def read_comparability(ctx: Context) -> dict | None:
    path = ctx.comparability_dir / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None

def headline_fitness(box, ids: list[int]) -> float:
    """Unweighted mean mAP50-95 over the class ids in `ids`, from an Ultralytics `Metric`."""
    present = [int(c) for c in box.ap_class_index]
    if missing := [i for i in ids if i not in present]:
        raise RuntimeError(f"no validation metrics for headline class ids {missing}")
    return float(np.mean([box.all_ap[present.index(i)].mean() for i in ids]))

def epoch_row(box, names: dict, epoch: int, fitness: float) -> dict:
    row = {"epoch": epoch, "fitness": fitness}
    for j, c in enumerate(box.ap_class_index):
        n = names[int(c)]
        row |= {f"{n}/P": float(box.p[j]), f"{n}/R": float(box.r[j]), f"{n}/mAP50": float(box.all_ap[j, 0]),
                f"{n}/mAP75": float(box.all_ap[j, 5]), f"{n}/mAP50-95": float(box.all_ap[j].mean())}
    return row

def headline_trainer(headline: list[str]) -> type[DetectionTrainer]:
    """A DetectionTrainer whose fitness (best.pt and early stopping) is the headline mean mAP50-95."""
    class HeadlineTrainer(DetectionTrainer):
        def validate(self):
            metrics = self.validator(self)
            if metrics is None:
                return None, None
            metrics.pop("fitness", None)
            names = self.data["names"]
            ids = [i for i, n in names.items() if n in headline]
            if len(ids) != len(headline):
                raise RuntimeError(f"headline classes {headline} not all in dataset names {names}")
            box = self.validator.metrics.box
            fitness = headline_fitness(box, ids)
            csv = Path(self.save_dir) / HEADLINE_CSV
            pd.DataFrame([epoch_row(box, names, self.epoch + 1, fitness)]).to_csv(
                csv, mode="a", header=not csv.exists(), index=False)
            if self.best_fitness is None or self.best_fitness < fitness:
                self.best_fitness = fitness
            return metrics, fitness
    return HeadlineTrainer

def optimizer_check(trainer, cfg: Train) -> dict:
    """The optimizer Ultralytics built must equal the pinned config (weight decay scaled as Ultralytics scales it)."""
    o, opt, problems = cfg.optimizer, trainer.optimizer, []
    if type(opt).__name__ != o["optimizer"]:
        problems.append(f"optimizer {type(opt).__name__} != {o['optimizer']}")
    decay = o["weight_decay"] * trainer.batch_size * max(round(o["nbs"] / trainer.batch_size), 1) / o["nbs"]
    for g in opt.param_groups:
        group = g.get("param_group")
        if g["initial_lr"] != o["lr0"]:
            problems.append(f"{group}: initial_lr {g['initial_lr']} != {o['lr0']}")
        if o["optimizer"] in ADAM_FAMILY and tuple(g.get("betas", ())) != (o["momentum"], ADAM_BETA2):
            problems.append(f"{group}: betas {g.get('betas')} != {(o['momentum'], ADAM_BETA2)}")
        want = decay if group == "weight" else 0.0
        if not math.isclose(g["weight_decay"], want, rel_tol=1e-12, abs_tol=0.0):
            problems.append(f"{group}: weight_decay {g['weight_decay']} != {want}")
    for k in SCHEDULE_KEYS:
        if k != "weight_decay" and getattr(trainer.args, k) != o[k]:
            problems.append(f"args.{k} {getattr(trainer.args, k)} != {o[k]}")
    return _check(not problems, problems or {"optimizer": o["optimizer"], "lr0": o["lr0"], "weight_decay": decay})

def reproducibility(baseline_csv: Path, run_csv: Path) -> dict:
    a, b = tr.read_results(baseline_csv).set_index("epoch"), tr.read_results(run_csv).set_index("epoch")
    epochs = a.index.intersection(b.index)
    loss = {c: _finite(((b.loc[epochs, c] - a.loc[epochs, c]).abs() / a.loc[epochs, c].abs()).max())
            for c in REPRO_LOSSES}
    maps = {c: _finite((b.loc[epochs, c] - a.loc[epochs, c]).abs().max()) for c in REPRO_MAPS}
    passed = (len(epochs) > 0 and all(v is not None and v <= REPRO_LOSS_RTOL for v in loss.values())
              and all(v is not None and v <= REPRO_MAP_ATOL for v in maps.values()))
    return _check(passed, {"epochs_compared": len(epochs), "max_rel_loss_diff": loss, "max_abs_map_diff": maps,
                           "loss_rtol": REPRO_LOSS_RTOL, "map_atol": REPRO_MAP_ATOL})

def diagnostics(run_dir: Path, metrics, names: list[str]) -> dict:
    """Overfitting, localization, stability and confusion diagnostics for one finished run (reported, not gates)."""
    res = tr.read_results(run_dir / "results.csv").set_index("epoch")
    fit = pd.read_csv(run_dir / HEADLINE_CSV).set_index("epoch")["fitness"]
    best = int(fit.idxmax())
    vb, tb = res["val/box_loss"], res["train/box_loss"]
    low = int(vb.idxmin())
    plateau = fit[fit.index >= PLATEAU_FROM_EPOCH]
    window = fit[np.abs(fit.index - best) <= STABILITY_HALF_WINDOW]
    per_class = json.loads((run_dir / "val_best.json").read_text())["per_class"]
    cm = np.asarray(metrics.confusion_matrix.matrix, dtype=float)
    totals = cm.sum(0)
    return {
        "best_epoch": best,
        "best_headline_mAP50-95": float(fit[best]),
        "overfitting": {"val_box_min_epoch": low, "val_box_min": float(vb[low]), "val_box_at_best": float(vb[best]),
                        "val_box_final": float(vb.iloc[-1]), "rise_to_best": float(vb[best] - vb[low]),
                        "rise_to_end": float(vb.iloc[-1] - vb[low])},
        "localization": {r["Class"]: {"mAP75/mAP50": _finite(r["mAP75"] / r["mAP50"]) if r["mAP50"] else None,
                                      "mAP50-95/mAP50": _finite(r["mAP50-95"] / r["mAP50"]) if r["mAP50"] else None}
                         for r in per_class},
        "plateau": {"from_epoch": PLATEAU_FROM_EPOCH, "epochs": len(plateau), "mean": _finite(plateau.mean()),
                    "sd": _finite(plateau.std())},
        "stability": {"epochs": [int(e) for e in window.index], "window_mean": float(window.mean()),
                      "best_minus_window_mean": float(fit[best] - window.mean())},
        "train_box_loss": {"first": float(tb.iloc[0]), "final": float(tb.iloc[-1])},
        "confusion_matrix": {"labels": [*names, "background"], "rows": "predicted", "columns": "true",
                             "counts": cm.tolist(),
                             "normalized_per_true_class": (cm / np.where(totals == 0, 1, totals)).round(4).tolist()},
    }

def non_pathological(checks: dict, diag: dict, warmup_epochs: float) -> dict:
    required = ("training_completed", "losses_finite", "val_best_complete", "baseline_untouched", "optimizer_pinned")
    failed = [k for k in required if not checks.get(k, {}).get("passed")]
    after_warmup = diag["best_epoch"] > warmup_epochs
    learned = diag["train_box_loss"]["final"] < diag["train_box_loss"]["first"]
    return _check(not failed and after_warmup and learned,
                  {"failed_checks": failed, "best_epoch": diag["best_epoch"], "best_after_warmup": after_warmup,
                   "train_box_loss_fell": learned})

def smoke_verdict(checks: dict) -> bool:
    """Smoke exit status: as strict as a full run, except that non_pathological may fail on its warmup condition
    alone, because a smoke run is shorter than the warmup. The recorded check is not changed."""
    def ok(name: str, check: dict) -> bool:
        if check["passed"]:
            return True
        if name != "non_pathological":
            return False
        d = check["detail"]
        return not d["failed_checks"] and d["train_box_loss_fell"] and not d["best_after_warmup"]
    return all(ok(k, c) for k, c in checks.items())

def dataset_for(ctx: Context, dataset: str) -> tuple[Path, Classes, list[str], dict, dict]:
    """(processed dir, classes, loader splits, gated fingerprint, extra run_meta) for a dataset name."""
    if dataset == "three_class":
        fingerprint = tr.gate(ctx.paths.processed_dir, ctx.paths.artifacts_dir / "verify")
        return ctx.paths.processed_dir, ctx.classes, list(SPLITS), fingerprint, {}
    derived, source = gate_two_class(ctx.paths, ctx.two_class)
    return (ctx.two_class.processed_dir, kept_classes(ctx.classes, ctx.two_class.exclude), list(ctx.two_class.splits),
            derived, {"source_dataset_fingerprint": source["dataset"]})

def run_experiment(ctx: Context, name: str, seed: int, smoke: bool = False, check_only: bool = False) -> bool:
    before = require_baseline(ctx)
    optimizer, source = baseline_optimizer(ctx.baseline_dir)
    cfg, dataset = build_train(ctx.spec, name, seed, optimizer, smoke)
    entry = ctx.spec["experiments"][name]
    run_dir, report = run_paths(ctx, name, seed, smoke)
    comparability = read_comparability(ctx)
    if not (smoke or check_only):
        if comparability is None:
            raise RuntimeError("run `python -m tvision comparability` before any full experiment run")
        check_siblings(run_dir, cfg)
    processed, classes, splits, fingerprint, extra = dataset_for(ctx, dataset)
    print(f"[PASS] gate: {dataset} dataset {fingerprint['dataset'][:16]}… matches its verified fingerprint")
    baseline_seed = json.loads((ctx.baseline_dir / "run_meta.json").read_text())["config"]["seed"]

    def after(trainer, metrics, meta, checks):
        new = {"optimizer_pinned": optimizer_check(trainer, cfg)}
        diag = diagnostics(run_dir, metrics, classes.names)
        if name == REPRODUCES_BASELINE and seed == baseline_seed and not smoke:
            new["reproduces_baseline"] = reproducibility(ctx.baseline_dir / "results.csv", run_dir / "results.csv")
        if dataset == "three_class" and comparability is not None and not comparability["identical"]:
            view = tr.validate_val(run_dir / "weights" / "best.pt", Path(comparability["view_data_yaml"]), cfg,
                                   run_dir, "val_view")
            table, headline = tr.per_class_val(view, classes)
            tr._write_json(run_dir / "val_view.json", {"per_class": table.reset_index().to_dict("records"),
                                                       "headline_mean": headline})
        new["baseline_untouched"] = baseline_check(ctx)
        new["non_pathological"] = non_pathological(checks | new, diag, cfg.optimizer["warmup_epochs"])
        tr._write_json(run_dir / "diagnostics.json", diag | {"non_pathological": new["non_pathological"]})
        return new

    extra_meta = {"step": "5A", "experiment": name, "factors": entry["factors"], "compare_to": entry["compare_to"],
                  "dataset": dataset, "optimizer_source": source | {"manifest_sha256": before["detail"]["manifest_sha256"]},
                  "baseline_untouched_before": before["detail"], **extra}
    return tr.fit(cfg, processed, classes, run_dir, report, fingerprint, splits=splits, smoke=smoke,
                  check_only=check_only, trainer_cls=headline_trainer(classes.headline), fitness_csv=HEADLINE_CSV,
                  extra_meta=extra_meta, after=after, verdict=smoke_verdict if smoke else None)

def build_view(source: Path, view: Path, classes: Classes, exclude: list[str]) -> dict:
    """Val images and labels with only `exclude` GT lines removed, under the unchanged class schema."""
    n_kept = len(kept_classes(classes, exclude).names)
    (view / "images" / "val").mkdir(parents=True)
    (view / "labels" / "val").mkdir(parents=True)
    differ = []
    for img in sorted((source / "images" / "val").glob("*.png")):
        shutil.copyfile(img, view / "images" / "val" / img.name)
        if _sha256(img) != _sha256(view / "images" / "val" / img.name):
            differ.append(img.stem)
        lines = filter_lines((source / "labels" / "val" / f"{img.stem}.txt").read_text(), n_kept)
        (view / "labels" / "val" / f"{img.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
    data = {"path": str(view), "train": "images/val", "val": "images/val", "names": dict(enumerate(classes.names))}
    (view / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    n_src = len(list((source / "images" / "val").glob("*.png")))
    return {"images": n_src, "images_identical": not differ, "names": classes.names,
            "passed": not differ and len(list((view / "labels" / "val").glob("*.txt"))) == n_src}

def headline_diff(full: pd.DataFrame, viewed: pd.DataFrame, headline: list[str]) -> dict:
    """Headline classes only: AP columns must be identical; P/R/F1 differences are recorded, not compared."""
    ap = float((full.loc[headline, AP_COLUMNS] - viewed.loc[headline, AP_COLUMNS]).abs().to_numpy().max())
    pr = float((full.loc[headline, PR_COLUMNS] - viewed.loc[headline, PR_COLUMNS]).abs().to_numpy().max())
    same_instances = bool((full.loc[headline, "Instances"] == viewed.loc[headline, "Instances"]).all())
    return {"identical": ap == 0.0 and same_instances, "ap_max_abs_diff": ap, "pr_max_abs_diff": pr,
            "headline_instances_equal": same_instances}

def comparability(ctx: Context) -> bool:
    """Frozen baseline best.pt on val vs. on the ambiguous-removed 3-class view: headline AP must be identical."""
    require_baseline(ctx)
    out = ctx.comparability_dir
    if out.exists():
        raise RuntimeError(f"{out} already exists; the comparability result is recorded once")
    tr.gate(ctx.paths.processed_dir, ctx.paths.artifacts_dir / "verify")
    view = out / "val_view"
    view_check = build_view(ctx.paths.processed_dir, view, ctx.classes, ctx.two_class.exclude)
    if not view_check["passed"]:
        raise RuntimeError(f"comparability view failed its checks: {view_check}")
    cfg, weights = Train.load(), ctx.baseline_dir / "weights" / "best.pt"
    full, _ = tr.per_class_val(tr.validate_val(weights, ctx.paths.processed_dir / "data.yaml", cfg, out, "full"),
                               ctx.classes)
    headline_only = replace(ctx.classes, names=ctx.classes.headline)  # the view has no ambiguous GT, so no row
    viewed, _ = tr.per_class_val(tr.validate_val(weights, view / "data.yaml", cfg, out, "view"), headline_only)
    diff = headline_diff(full, viewed, ctx.classes.headline)
    recorded = pd.DataFrame(json.loads((ctx.baseline_dir / "val_best.json").read_text())["per_class"]).set_index("Class")
    ap, pr = diff["ap_max_abs_diff"], diff["pr_max_abs_diff"]
    result = {
        **diff,
        "headline_classes": ctx.classes.headline,
        "ap_columns": AP_COLUMNS,
        "full_matches_recorded_val_best": bool(np.allclose(full[tr.METRIC_COLUMNS].to_numpy(dtype=float),
                                                           recorded.loc[full.index, tr.METRIC_COLUMNS].to_numpy(dtype=float))),
        "note": ("AP is computed per class from that class's predictions and GT only, so removing ambiguous GT "
                 "cannot change tumor/non_tumor AP. P/R/F1 are read at one confidence chosen by the mean F1 over all "
                 "classes present, so they may move when ambiguous GT is removed; they are recorded, not compared."),
        "weights_sha256": _sha256(weights),
        "view_data_yaml": str(view / "data.yaml"),
        "view": view_check,
        "full_per_class": full.reset_index().to_dict("records"),
        "view_per_class": viewed.reset_index().to_dict("records"),
    }
    after = baseline_check(ctx)
    result["baseline_untouched_after"] = after
    tr._write_json(out / "result.json", result)
    (out / "summary.md").write_text("# Comparability: baseline best.pt, val vs. ambiguous-removed 3-class view\n\n"
                                    + _block("Result", json.dumps({k: v for k, v in result.items()
                                                                   if not k.endswith("per_class")}, indent=2))
                                    + _block("Full val (3-class)", full.round(5))
                                    + _block("Ambiguous-removed view (3-class schema)", viewed.round(5)))
    print(f"[{'PASS' if result['identical'] else 'DIFFERENT'}] headline AP identical: max |diff| {ap}; "
          f"P/R max |diff| {pr}\nReport: {out / 'summary.md'}")
    return after["passed"]

def _render(item: dict, names: dict) -> np.ndarray:
    img = np.ascontiguousarray(item["img"].numpy().transpose(1, 2, 0)[..., ::-1])
    h, w = img.shape[:2]
    boxes = [{"x1": (x - bw / 2) * w, "y1": (y - bh / 2) * h, "x2": (x + bw / 2) * w, "y2": (y + bh / 2) * h,
              "class_name": names[int(c)]}
             for (x, y, bw, bh), c in zip(item["bboxes"].tolist(), item["cls"].view(-1).tolist())]
    return draw_boxes(img, boxes, thickness=1)

def aug_preview(ctx: Context, name: str, n: int = 8, variants: int = 3, seed: int = 0) -> Path:
    """Per sampled training image: the unaugmented letterboxed input, then `variants` seeded augmented draws."""
    optimizer, _ = baseline_optimizer(ctx.baseline_dir)
    cfg, dataset = build_train(ctx.spec, name, 0, optimizer)
    processed = ctx.paths.processed_dir if dataset == "three_class" else ctx.two_class.processed_dir
    data = check_det_dataset(str(processed / "data.yaml"), autodownload=False)
    hyp = get_cfg(overrides=cfg.ultralytics_args())
    plain, aug = (YOLODataset(img_path=data["train"], data=data, task="detect", augment=a, hyp=hyp, imgsz=cfg.imgsz)
                  for a in (False, True))
    idx = sorted(np.random.default_rng(seed).choice(len(aug), size=min(n, len(aug)), replace=False).tolist())
    out = ctx.artifacts / "aug_preview" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for k, i in enumerate(idx):
        panels = [_render(plain[i], data["names"])]
        for v in range(variants):
            random.seed(seed * 1000 + k * 10 + v)
            np.random.seed(seed * 1000 + k * 10 + v)
            panels.append(_render(aug[i], data["names"]))
        cv2.imwrite(str(out / f"{k:02d}_{Path(plain.im_files[i]).stem}.png"), side_by_side(*panels))
    changed = {k: v for k, v in cfg.augmentation.items() if v != ctx.spec["control"]["augmentation"][k]}
    (out / "summary.md").write_text(
        f"# Augmentation preview: {name}\n\nEach row: unaugmented input (letterboxed to {cfg.imgsz}) | {variants} "
        f"augmented training draws (seeded). Boxes are the labels after augmentation.\n\n"
        + _block("Changed from control", json.dumps(changed, indent=2))
        + _block("Full augmentation", json.dumps(cfg.augmentation, indent=2)))
    print(f"Preview: {out} ({len(idx)} images)")
    return out

def baseline_manifest() -> bool:
    ctx = Context.load()
    sha = make_manifest(ctx, committed_bytes(ctx.train_yaml))
    files = json.loads(ctx.manifest.read_text())["n_files"]
    print(f"Manifest: {ctx.manifest} ({files} files)\nSHA-256: {sha}\n"
          "Record it as baseline_manifest_sha256 in configs/experiments.yaml, then commit.")
    return True

def experiment(name: str, seed: int, smoke: bool = False, check_only: bool = False) -> bool:
    return run_experiment(Context.load(), name, seed, smoke, check_only)