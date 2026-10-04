"""Step 5: fine-tune the pretrained detector on the verified NuCLS train split.

A thin wrapper around Ultralytics training. Training is refused unless every Step 4 check passed, the
processed dataset still matches its verified fingerprint, and Ultralytics' own loader reads exactly the
verified labels. The test split is loaded only for that integrity check; no model runs on it here.
"""

from tvision.ultralytics_env import check_device  # configures Ultralytics before it is imported

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

from tvision.config import PROJECT_ROOT, Classes, Paths, Train
from tvision.explore import _block
from tvision.verify import SPLITS, dataset_fingerprint

METRIC_COLUMNS = ["Box-P", "Box-R", "Box-F1", "mAP50", "mAP75", "mAP50-95"]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"{path} not found; run `python -m tvision verify` first")
    return json.loads(path.read_text())

def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

def gate(processed: Path, verify_dir: Path) -> dict:
    """Raise unless Step 4 passed and the processed dataset is byte-identical to what it verified."""
    checks = _read_json(verify_dir / "checks.json")
    if failed := [k for k, c in checks.items() if not c["passed"]]:
        raise RuntimeError(f"Step 4 checks failed: {failed}; fix the conversion before training")
    expected = _read_json(verify_dir / "fingerprint.json")
    actual = dataset_fingerprint(processed)
    if actual != expected:
        changed = [k for k in actual if actual[k] != expected.get(k)]
        raise RuntimeError(f"processed dataset differs from the verified fingerprint in {changed}; "
                           "re-run convert and verify")
    if importlib.util.find_spec("albumentations") is not None:
        raise RuntimeError("albumentations is installed; Ultralytics would apply its colour transforms by default")
    return actual

def label_counts(label_dir: Path) -> dict[str, Counter]:
    """Class id counts per label file stem, read from the label text."""
    return {p.stem: Counter(int(line.split()[0]) for line in p.read_text().splitlines())
            for p in sorted(label_dir.glob("*.txt"))}

def loader_check(processed: Path, classes: Classes, splits=SPLITS) -> dict:
    """Ultralytics' loader must read every image with exactly the label lines on disk, per split."""
    data = check_det_dataset(str(processed / "data.yaml"), autodownload=False)
    result = {}
    for split in splits:
        ds = YOLODataset(img_path=data[split], data=data, task="detect", augment=False)
        loaded = {Path(lb["im_file"]).stem: Counter(int(c) for c in lb["cls"].ravel()) for lb in ds.labels}
        expected = label_counts(processed / "labels" / split)
        n_images = len(list((processed / "images" / split).glob("*.png")))
        total_loaded = sum(loaded.values(), Counter())
        total_expected = sum(expected.values(), Counter())
        mismatched = [s for s in sorted(expected.keys() | loaded.keys()) if loaded.get(s) != expected.get(s)]
        result[split] = {
            "images": n_images,
            "loaded": len(loaded),
            "instances": {n: total_loaded[i] for i, n in enumerate(classes.names)},
            "expected": {n: total_expected[i] for i, n in enumerate(classes.names)},
            "images_mismatched": len(mismatched),
            "passed": len(loaded) == n_images == len(expected) and not mismatched,
        }
    return result

def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}

def read_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    return df

def convergence(results_csv: Path, planned_epochs: int, fitness_csv: Path | None = None) -> dict:
    """Best epoch by Ultralytics' all-class mAP50-95, or by the `fitness` column of `fitness_csv` if given."""
    df = read_results(results_csv)
    losses = [c for c in df.columns if "loss" in c]
    best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    out = {
        "epochs_run": len(df),
        "epochs_planned": planned_epochs,
        "stopped_early": len(df) < planned_epochs,
        "best_epoch": int(best["epoch"]),
        "best_val_mAP50-95": float(best["metrics/mAP50-95(B)"]),
        "losses_finite": bool(np.isfinite(df[losses].to_numpy()).all()),
        "final": {c: float(df[c].iloc[-1]) for c in losses},
    }
    if fitness_csv is not None:
        fit_df = pd.read_csv(fitness_csv)
        best = fit_df.loc[fit_df["fitness"].idxmax()]
        out["best_epoch"] = int(best["epoch"])
        out["best_val_mAP50-95"] = float(df.set_index("epoch").loc[out["best_epoch"], "metrics/mAP50-95(B)"])
        out["best_fitness"] = float(best["fitness"])
    return out

def per_class_val(metrics, classes: Classes) -> tuple[pd.DataFrame, dict]:
    """Per-class table from Ultralytics' `metrics.summary()` and the unweighted mean over headline classes."""
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in metrics.summary()]
    table = pd.DataFrame(rows, columns=["Class", "Images", "Instances", *METRIC_COLUMNS]).set_index("Class")
    if missing := sorted(set(classes.names) - set(table.index)):
        raise RuntimeError(f"no validation metrics for classes {missing}")
    table = table.loc[classes.names]
    headline = {k: float(v) for k, v in table.loc[classes.headline, METRIC_COLUMNS].mean().items()}
    return table, headline

def versions() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__, "pandas": pd.__version__}

def summarize(name: str, checks: dict, conv: dict, table: pd.DataFrame, headline: dict, meta: dict) -> str:
    sections = [
        _block("Checks", json.dumps(checks, indent=2)),
        _block("Convergence (results.csv)", json.dumps(conv, indent=2)),
        _block("best.pt on val, per class", table.round(4)),
        _block("Headline mean (unweighted over tumor, non_tumor)", json.dumps(headline, indent=2)),
        _block("Run", json.dumps({k: meta[k] for k in ("run_dir", "device", "git", "versions")}, indent=2)),
    ]
    return f"# Training: {name}\n\n" + "\n".join(sections)

def validate_val(weights: Path, data_yaml: Path, cfg: Train, project: Path, name: str):
    """Validate `weights` on the val split only, with the training imgsz, max_det, device and batch."""
    return YOLO(weights).val(data=str(data_yaml), split="val", imgsz=cfg.imgsz, max_det=cfg.max_det,
                             device=cfg.device, batch=cfg.batch, project=str(project), name=name)

def fit(cfg: Train, processed: Path, classes: Classes, run_dir: Path, report: Path, fingerprint: dict, *,
        splits=SPLITS, smoke: bool = False, check_only: bool = False, trainer_cls=None, fitness_csv: str | None = None,
        extra_meta: dict | None = None, after=None, verdict=None) -> bool:
    """Loader check, training into `run_dir`, val of best.pt, run_meta.json and the report.

    `after(trainer, metrics, meta, checks)` may return further checks, recorded with the others.
    `verdict(checks)` decides the return value; by default every check must pass.
    """
    data_yaml = processed / "data.yaml"
    loader = loader_check(processed, classes, splits)
    checks = {"gate": {"passed": True, "detail": fingerprint["dataset"]},
              "loader_counts_match": {"passed": all(s["passed"] for s in loader.values()), "detail": loader}}
    for split, s in loader.items():
        print(f"[{'PASS' if s['passed'] else 'FAIL'}] loader {split}: {s['loaded']}/{s['images']} images, "
              f"{s['instances']}")
    if check_only or not checks["loader_counts_match"]["passed"]:
        return checks["loader_counts_match"]["passed"]

    check_device(cfg.device)
    git = git_state()
    if git["dirty"] and not smoke:
        raise RuntimeError("git working tree is dirty; commit before a full run so run_meta.json is reproducible")
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to overwrite a run")

    started = _now()
    model = YOLO(cfg.model)
    pretrained = Path(model.ckpt_path).resolve()
    model.train(data=str(data_yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=False,
                trainer=trainer_cls, **cfg.ultralytics_args())
    trainer = model.trainer
    if Path(trainer.save_dir).resolve() != run_dir.resolve():
        raise RuntimeError(f"Ultralytics wrote to {trainer.save_dir}, expected {run_dir}")
    best = run_dir / "weights" / "best.pt"

    metrics = validate_val(best, data_yaml, cfg, run_dir, "val_best")
    table, headline = per_class_val(metrics, classes)
    _write_json(run_dir / "val_best.json", {"per_class": table.reset_index().to_dict("records"),
                                            "headline_mean": headline, "headline_classes": classes.headline})
    conv = convergence(run_dir / "results.csv", cfg.epochs, run_dir / fitness_csv if fitness_csv else None)

    meta = {
        "step": 5,
        "smoke": smoke,
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "ultralytics_args": cfg.ultralytics_args(),
        "data_yaml": str(data_yaml),
        "dataset_fingerprint": {k: v["sha256"] for k, v in fingerprint.items() if isinstance(v, dict)}
                               | {"dataset": fingerprint["dataset"]},
        "pretrained": {"path": str(pretrained), "sha256": _sha256(pretrained)},
        "best": {"path": str(best), "sha256": _sha256(best)},
        "device": {"name": cfg.device, "amp": bool(trainer.amp), "deterministic": cfg.deterministic,
                   "seed": cfg.seed},
        "optimizer": {"name": type(trainer.optimizer).__name__,
                      "lr0": trainer.optimizer.param_groups[0]["initial_lr"],
                      "momentum": trainer.optimizer.defaults.get("momentum", trainer.optimizer.defaults.get("betas"))},
        "git": git,
        "versions": versions(),
        "platform": platform.platform(),
        "started": started,
        "finished": _now(),
    } | (extra_meta or {})
    _write_json(run_dir / "run_meta.json", meta)

    checks["training_completed"] = {"passed": best.is_file() and (run_dir / "weights" / "last.pt").is_file()
                                    and conv["epochs_run"] >= 1, "detail": conv["epochs_run"]}
    checks["losses_finite"] = {"passed": conv["losses_finite"], "detail": conv["final"]}
    checks["val_best_complete"] = {"passed": bool(np.isfinite(table[METRIC_COLUMNS].to_numpy()).all()),
                                   "detail": headline["mAP50-95"]}
    if after is not None:
        checks |= after(trainer, metrics, meta, checks)
    shutil.rmtree(report, ignore_errors=True)
    report.mkdir(parents=True)
    _write_json(report / "checks.json", checks)
    (report / "summary.md").write_text(summarize(cfg.name, checks, conv, table, headline, meta))
    for k, c in checks.items():
        if k not in ("gate", "loader_counts_match"):
            print(f"[{'PASS' if c['passed'] else 'FAIL'}] {k}: {c['detail']}")
    print(f"Run: {run_dir}\nReport: {report / 'summary.md'}")
    return verdict(checks) if verdict is not None else all(c["passed"] for c in checks.values())

def run_training(cfg: Train, paths: Paths, classes: Classes, smoke: bool = False, check_only: bool = False) -> bool:
    fingerprint = gate(paths.processed_dir, paths.artifacts_dir / "verify")
    print(f"[PASS] gate: Step 4 checks passed, dataset {fingerprint['dataset'][:16]}… matches")
    return fit(cfg, paths.processed_dir, classes, paths.runs_dir / cfg.name, paths.artifacts_dir / "train" / cfg.name,
               fingerprint, smoke=smoke, check_only=check_only)

def train(smoke: bool = False, check_only: bool = False) -> bool:
    return run_training(Train.load(smoke), Paths.load(), Classes.load(), smoke, check_only)