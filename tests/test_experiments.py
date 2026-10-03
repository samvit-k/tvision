import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.metrics import DetMetrics

from tvision import experiments as ex
from tvision import train as tr
from tvision.config import CONFIG_DIR, PROJECT_ROOT, Classes, Train, TwoClass, load_yaml

from test_train import _verified

CLASSES = Classes.load()
SPEC = load_yaml("experiments.yaml")
OPT = {"optimizer": "AdamW", "lr0": 0.001429, "momentum": 0.9, "weight_decay": 0.0005, "nbs": 64, "lrf": 0.01,
       "cos_lr": False, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.0}

def _spec(**experiments) -> dict:
    spec = copy.deepcopy(SPEC)
    spec["experiments"] |= experiments
    return spec

def test_every_defined_experiment_changes_only_its_declared_factors():
    for name in SPEC["experiments"]:
        ex.check_factors(SPEC, name)

def test_control_equals_baseline_config():
    baseline = load_yaml("train.yaml")
    control = {k: v for k, v in SPEC["control"].items() if k != "optimizer"}
    assert control == {k: v for k, v in baseline.items() if k not in ("name", "seed", "smoke")}

@pytest.mark.parametrize("entry, match", [
    ({"factors": ["appearance"], "compare_to": "a0_control", "augmentation": {"hsv_s": 0.3}, "imgsz": 768}, "outside"),
    ({"factors": ["appearance", "imgsz"], "compare_to": "a0_control", "augmentation": {"hsv_s": 0.3}}, "does not change"),
    ({"factors": [], "compare_to": None, "batch": 8}, "unchanged control"),
    ({"factors": ["colour"], "compare_to": "a0_control"}, "unknown factors"),
    ({"factors": ["batch"], "compare_to": "x_self", "batch": 8}, "another defined"),
    ({"factors": ["batch"], "compare_to": "a0_control", "lr0": 0.1}, "unknown keys"),
    ({"factors": ["batch"], "batch": 8}, "must declare"),
])
def test_factor_violations_refused(entry, match):
    with pytest.raises(ValueError, match=match):
        ex.check_factors(_spec(x_self=entry), "x_self")

def test_matched_batch_control_isolates_imgsz():
    spec = _spec(a0_control_b8={"factors": ["batch"], "compare_to": "a0_control", "batch": 8},
                 a3_b8={"factors": ["imgsz"], "compare_to": "a0_control_b8", "imgsz": 768, "batch": 8})
    ex.check_factors(spec, "a0_control_b8")
    ex.check_factors(spec, "a3_b8")
    with pytest.raises(ValueError, match="outside"):
        ex.check_factors(_spec(a3_b8={"factors": ["imgsz"], "compare_to": "a0_control", "imgsz": 768, "batch": 8}),
                         "a3_b8")

@pytest.mark.parametrize("name", ["smoke", "Bad-Name", "missing"])
def test_invalid_or_unknown_names_refused(name):
    spec = _spec(**({name: {"factors": [], "compare_to": None}} if name != "missing" else {}))
    with pytest.raises(ValueError):
        ex.build_train(spec, name, 0, OPT)

def test_build_train_pins_optimizer_seed_and_dataset():
    cfg, dataset = ex.build_train(SPEC, "a4_two_class", 2, OPT)
    assert dataset == "two_class" and cfg.seed == 2 and cfg.name == "step5a/a4_two_class/seed2"
    args = cfg.ultralytics_args()
    assert {k: args[k] for k in OPT} == OPT
    smoke, _ = ex.build_train(SPEC, "a1_appearance", 0, OPT, smoke=True)
    assert (smoke.epochs, smoke.fraction) == (SPEC["smoke"]["epochs"], SPEC["smoke"]["fraction"])
    assert smoke.augmentation["hsv_s"] == SPEC["experiments"]["a1_appearance"]["augmentation"]["hsv_s"]

def test_build_train_requires_from_baseline_optimizer():
    spec = copy.deepcopy(SPEC)
    spec["control"]["optimizer"] = "auto"
    with pytest.raises(ValueError, match="from_baseline"):
        ex.build_train(spec, "a0_control", 0, OPT)

@pytest.mark.parametrize("optimizer", [{**OPT, "optimizer": "auto"}, {k: v for k, v in OPT.items() if k != "nbs"}])
def test_train_rejects_bad_pinned_optimizer(optimizer):
    with pytest.raises(ValueError):
        Train.from_dict(load_yaml("train.yaml") | {"optimizer": optimizer})

def _fake_baseline(root: Path, args_yaml: dict | None = None, train_args: dict | None = None,
                   betas=(0.9, 0.999)) -> Path:
    base = root / "runs" / "baseline"
    (base / "weights").mkdir(parents=True)
    saved = {"optimizer": "auto", "lr0": 0.01, "momentum": 0.937, **{k: OPT[k] for k in ex.SCHEDULE_KEYS},
             "warmup_bias_lr": 0.1} | (args_yaml or {})
    effective = saved | {"warmup_bias_lr": 0.0} | (train_args or {})
    torch.save({"train_args": effective}, base / "weights" / "best.pt")
    (base / "args.yaml").write_text(yaml.safe_dump(saved))
    (base / "run_meta.json").write_text(json.dumps({
        "optimizer": {"name": "AdamW", "lr0": 0.001429, "momentum": list(betas)},
        "best": {"sha256": ex._sha256(base / "weights" / "best.pt")},
        "config": {"seed": 0}}))
    return base

def test_baseline_optimizer_uses_effective_train_args(tmp_path):
    optimizer, source = ex.baseline_optimizer(_fake_baseline(tmp_path))
    assert optimizer == OPT
    assert source["args_yaml_differs"] == {"warmup_bias_lr": {"args_yaml": 0.1, "train_args": 0.0}}

@pytest.mark.parametrize("kwargs, match", [({"train_args": {"lrf": 0.02}}, "disagree"),
                                           ({"betas": (0.9, 0.99)}, "beta2")])
def test_baseline_optimizer_refuses_inconsistent_baseline(tmp_path, kwargs, match):
    with pytest.raises(RuntimeError, match=match):
        ex.baseline_optimizer(_fake_baseline(tmp_path, **kwargs))

@pytest.mark.skipif(not (PROJECT_ROOT / "runs" / "baseline" / "run_meta.json").is_file(), reason="no baseline run")
def test_real_baseline_optimizer_values():
    optimizer, _ = ex.baseline_optimizer(PROJECT_ROOT / "runs" / "baseline")
    assert optimizer == OPT

def _built(name: str, lr: float, momentum: float, warmup_bias_lr: float) -> tuple:
    fake = SimpleNamespace(args=SimpleNamespace(lr0=lr, warmup_bias_lr=warmup_bias_lr), data={"nc": 3})
    torch.manual_seed(0)
    opt = BaseTrainer.build_optimizer(fake, nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)), name=name, lr=lr,
                                      momentum=momentum, decay=0.0005, iterations=2000)
    return type(opt).__name__, [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups], \
        fake.args.warmup_bias_lr

def test_pinned_optimizer_builds_what_auto_built_for_the_baseline():
    auto = _built("auto", 0.01, 0.937, 0.1)
    pinned = _built(OPT["optimizer"], OPT["lr0"], OPT["momentum"], OPT["warmup_bias_lr"])
    assert auto == pinned
    assert _built("auto", 0.01, 0.937, 0.1)[1][0]["lr"] != round(0.002 * 5 / (4 + 2), 6)

def _metrics() -> DetMetrics:
    metrics = DetMetrics(names=dict(enumerate(CLASSES.names)))
    metrics.update_stats({"tp": np.array([[1] * 10, [1] * 5 + [0] * 5, [0] * 10, [1] * 10], bool),
                          "conf": np.array([0.9, 0.8, 0.7, 0.6]), "pred_cls": np.array([0, 1, 2, 0]),
                          "target_cls": np.array([0, 1, 2, 0]), "target_img": np.array([0, 1, 2, 3]), "im_name": "a"})
    metrics.process()
    return metrics

def test_headline_fitness_is_mean_over_headline_classes_only():
    box = _metrics().box
    ids = [CLASSES.class_id(n) for n in CLASSES.headline]
    per_class = {int(c): box.all_ap[j].mean() for j, c in enumerate(box.ap_class_index)}
    assert ex.headline_fitness(box, ids) == pytest.approx(np.mean([per_class[i] for i in ids]))
    assert ex.headline_fitness(box, ids) != pytest.approx(box.map)
    with pytest.raises(RuntimeError, match="no validation metrics"):
        ex.headline_fitness(box, [7])

def test_headline_trainer_validate_uses_headline_fitness(tmp_path):
    metrics = _metrics()

    class Validator:
        def __init__(self):
            self.metrics = metrics

        def __call__(self, trainer):
            return {"metrics/mAP50-95(B)": 0.5, "fitness": 9.0}

    fake = SimpleNamespace(validator=Validator(), data={"names": dict(enumerate(CLASSES.names))}, epoch=0,
                           save_dir=tmp_path, best_fitness=None)
    trainer_cls = ex.headline_trainer(CLASSES.headline)
    out, fitness = trainer_cls.validate(fake)
    expected = ex.headline_fitness(metrics.box, [0, 1])
    assert fitness == pytest.approx(expected) and fake.best_fitness == fitness and "fitness" not in out
    fake.epoch, fake.best_fitness = 1, fitness + 1
    trainer_cls.validate(fake)
    assert fake.best_fitness == fitness + 1
    log = pd.read_csv(tmp_path / ex.HEADLINE_CSV)
    assert list(log["epoch"]) == [1, 2] and log["fitness"].iloc[0] == pytest.approx(expected)
    assert {"tumor/mAP75", "ambiguous/mAP50-95"} <= set(log.columns)

def _ctx(tmp_path: Path, anchor: str | None = None, paths=None) -> ex.Context:
    paths = paths or tr.Paths(tmp_path / "raw", tmp_path / "processed", tmp_path / "artifacts", tmp_path / "runs")
    train_yaml = tmp_path / "train.yaml"
    if not train_yaml.exists():
        train_yaml.write_text((CONFIG_DIR / "train.yaml").read_text())
    return ex.Context(paths, CLASSES, copy.deepcopy(SPEC) | {"baseline_manifest_sha256": anchor},
                      TwoClass(tmp_path / "two", ["ambiguous"], ["train", "val"]), train_yaml)

def _anchored(tmp_path: Path, paths=None) -> ex.Context:
    ctx = _ctx(tmp_path, paths=paths)
    if not ctx.baseline_dir.exists():
        _fake_baseline(ctx.paths.runs_dir.parent)
    (ctx.paths.artifacts_dir / "train" / "baseline").mkdir(parents=True, exist_ok=True)
    (ctx.paths.artifacts_dir / "train" / "baseline" / "checks.json").write_text("{}")
    sha = ex.make_manifest(ctx, ctx.train_yaml.read_bytes())
    return replace(ctx, spec=ctx.spec | {"baseline_manifest_sha256": sha})

def test_manifest_covers_all_baseline_artifacts(tmp_path):
    ctx = _anchored(tmp_path)
    files = json.loads(ctx.manifest.read_text())["files"]
    assert {"runs/baseline/weights/best.pt", "runs/baseline/args.yaml", "artifacts/train/baseline/checks.json",
            "configs/train.yaml"} <= set(files)
    assert ex.baseline_check(ctx)["passed"]
    (ctx.baseline_dir / ".DS_Store").write_text("finder")
    assert ex.baseline_check(ctx)["passed"]

@pytest.mark.parametrize("mutate, kind", [
    (lambda c: (c.baseline_dir / "args.yaml").write_text("changed"), "changed"),
    (lambda c: (c.baseline_dir / "weights" / "extra.pt").write_bytes(b"x"), "added"),
    (lambda c: (c.paths.artifacts_dir / "train" / "baseline" / "checks.json").unlink(), "removed"),
    (lambda c: c.train_yaml.write_text(c.train_yaml.read_text() + "\n"), "changed"),
])
def test_baseline_check_catches_any_change(tmp_path, mutate, kind):
    ctx = _anchored(tmp_path)
    mutate(ctx)
    check = ex.baseline_check(ctx)
    assert not check["passed"] and check["detail"][kind]
    with pytest.raises(RuntimeError, match="baseline_untouched"):
        ex.require_baseline(ctx)

def test_baseline_check_requires_matching_anchor(tmp_path):
    ctx = _anchored(tmp_path)
    assert not ex.baseline_check(replace(ctx, spec=ctx.spec | {"baseline_manifest_sha256": None}))["passed"]
    assert not ex.baseline_check(replace(ctx, spec=ctx.spec | {"baseline_manifest_sha256": "0" * 64}))["passed"]
    ctx.manifest.write_text(ctx.manifest.read_text() + " ")
    assert not ex.baseline_check(ctx)["passed"]

def test_make_manifest_refuses_overwrite_and_inconsistent_baseline(tmp_path):
    ctx = _anchored(tmp_path)
    with pytest.raises(RuntimeError, match="already exists"):
        ex.make_manifest(ctx, ctx.train_yaml.read_bytes())
    _fake_baseline(tmp_path / "b")
    other = _ctx(tmp_path / "b")
    with pytest.raises(RuntimeError, match="differs from git HEAD"):
        ex.make_manifest(other, b"something else")
    (other.baseline_dir / "weights" / "best.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="best.pt"):
        ex.make_manifest(other, other.train_yaml.read_bytes())

@pytest.mark.skipif(load_yaml("experiments.yaml")["baseline_manifest_sha256"] is None, reason="anchor not committed yet")
def test_committed_anchor_is_recorded_in_project_status():
    anchor = load_yaml("experiments.yaml")["baseline_manifest_sha256"]
    assert anchor in (PROJECT_ROOT / "docs" / "PROJECT_STATUS.md").read_text()

def test_run_dirs_stay_inside_step5a(tmp_path):
    ctx = _ctx(tmp_path)
    run_dir, report = ex.run_paths(ctx, "a0_control", 1, smoke=False)
    assert run_dir == ctx.runs / "a0_control" / "seed1" and report == ctx.artifacts / "a0_control" / "seed1"
    assert ex.run_paths(ctx, "a0_control", 0, smoke=True)[0] == ctx.runs / "smoke" / "a0_control" / "seed0"
    for bad in (ctx.runs, ctx.baseline_dir / "x", ctx.paths.runs_dir / "other", ctx.runs / ".." / "baseline"):
        with pytest.raises(RuntimeError, match="refusing"):
            ex.check_run_dir(ctx, bad)

def _fake_trainer(**overrides) -> SimpleNamespace:
    o = OPT | overrides
    groups = [{"params": [nn.Parameter(torch.zeros(1))], "weight_decay": wd, "param_group": g}
              for g, wd in (("bias", 0.0), ("weight", o["weight_decay"] * 16 * 4 / 64), ("bn", 0.0))]
    opt = torch.optim.AdamW(groups, lr=o["lr0"], betas=(o["momentum"], 0.999))
    for g in opt.param_groups:
        g["initial_lr"] = o["lr0"]
    args = SimpleNamespace(**{k: o[k] for k in ex.SCHEDULE_KEYS})
    return SimpleNamespace(optimizer=opt, batch_size=16, args=args)

def test_optimizer_check_passes_when_pinned_and_catches_drift():
    cfg = Train.from_dict(load_yaml("train.yaml") | {"optimizer": OPT})
    assert ex.optimizer_check(_fake_trainer(), cfg)["passed"]
    for drift in ({"lr0": 0.001667}, {"momentum": 0.937}, {"warmup_bias_lr": 0.1}, {"weight_decay": 0.001}):
        check = ex.optimizer_check(_fake_trainer(**drift), cfg)
        assert not check["passed"], drift

def _results(path: Path, n: int, loss_scale: float = 1.0, map_shift: float = 0.0, val_box=None) -> Path:
    epochs = np.arange(1, n + 1)
    pd.DataFrame({"epoch": epochs, "time": epochs * 10.0,
                  "train/box_loss": loss_scale * (2.0 - epochs * 0.01), "train/cls_loss": loss_scale * 1.5,
                  "train/l1_loss": loss_scale * 0.7, "metrics/mAP50(B)": 0.4 + map_shift,
                  "metrics/mAP50-95(B)": 0.15 + map_shift + epochs * 1e-4,
                  "val/box_loss": val_box if val_box is not None else 1.0 + (epochs - 5) ** 2 * 0.01,
                  "val/cls_loss": 1.0, "val/l1_loss": 0.5}).to_csv(path, index=False)
    return path

def test_reproducibility_tolerances(tmp_path):
    base = _results(tmp_path / "base.csv", 60)
    assert ex.reproducibility(base, _results(tmp_path / "same.csv", 40, loss_scale=1.009, map_shift=0.004))["passed"]
    assert not ex.reproducibility(base, _results(tmp_path / "loss.csv", 40, loss_scale=1.02))["passed"]
    assert not ex.reproducibility(base, _results(tmp_path / "map.csv", 40, map_shift=0.006))["passed"]
    detail = ex.reproducibility(base, _results(tmp_path / "short.csv", 25))["detail"]
    assert detail["epochs_compared"] == 25

def test_diagnostics_and_non_pathological(tmp_path):
    _results(tmp_path / "results.csv", 30)
    fit = 0.2 + 0.01 * np.sin(np.arange(1, 31))
    fit[11] = 0.3
    pd.DataFrame({"epoch": np.arange(1, 31), "fitness": fit}).to_csv(tmp_path / ex.HEADLINE_CSV, index=False)
    (tmp_path / "val_best.json").write_text(json.dumps({"per_class": [
        {"Class": "tumor", "mAP50": 0.6, "mAP75": 0.2, "mAP50-95": 0.27}]}))
    metrics = SimpleNamespace(confusion_matrix=SimpleNamespace(matrix=np.array([[5, 1], [0, 0]])))
    diag = ex.diagnostics(tmp_path, metrics, ["tumor"])
    assert diag["best_epoch"] == 12 and diag["overfitting"]["val_box_min_epoch"] == 5
    assert diag["stability"]["epochs"] == [10, 11, 12, 13, 14]
    assert diag["localization"]["tumor"]["mAP75/mAP50"] == pytest.approx(0.2 / 0.6)
    assert diag["confusion_matrix"]["normalized_per_true_class"][0] == [1.0, 1.0]
    json.dumps(diag, allow_nan=False)
    ok = {k: {"passed": True} for k in ("training_completed", "losses_finite", "val_best_complete",
                                        "baseline_untouched", "optimizer_pinned")}
    assert ex.non_pathological(ok, diag, 3.0)["passed"]
    assert not ex.non_pathological(ok, diag | {"best_epoch": 3}, 3.0)["passed"]
    assert not ex.non_pathological(ok | {"optimizer_pinned": {"passed": False}}, diag, 3.0)["passed"]
    assert not ex.non_pathological(ok, diag | {"train_box_loss": {"first": 1.0, "final": 1.2}}, 3.0)["passed"]

class _NoTraining(Exception):
    pass

@pytest.fixture
def no_training(monkeypatch):
    monkeypatch.setattr(tr, "check_device", lambda device: None)
    monkeypatch.setattr(tr, "YOLO", lambda *a, **k: (_ for _ in ()).throw(_NoTraining()))

def _experiment_ctx(tmp_path: Path) -> ex.Context:
    return _anchored(tmp_path, paths=_verified(tmp_path))

def test_experiment_refused_without_anchor(tmp_path):
    ctx = _ctx(tmp_path, paths=_verified(tmp_path))
    with pytest.raises(RuntimeError, match="baseline_untouched"):
        ex.run_experiment(ctx, "a0_control", 0, check_only=True)

def test_experiment_check_only_runs_gates_without_training(tmp_path, no_training):
    assert ex.run_experiment(_experiment_ctx(tmp_path), "a0_control", 0, check_only=True)

def test_full_experiment_requires_comparability_and_clean_tree(tmp_path, monkeypatch, no_training):
    ctx = _experiment_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="comparability"):
        ex.run_experiment(ctx, "a1_appearance", 0)
    ctx.comparability_dir.mkdir(parents=True)
    (ctx.comparability_dir / "result.json").write_text(json.dumps({"identical": True}))
    monkeypatch.setattr(tr, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(RuntimeError, match="dirty"):
        ex.run_experiment(ctx, "a1_appearance", 0)
    monkeypatch.setattr(tr, "git_state", lambda: {"commit": "abc", "dirty": False})
    with pytest.raises(_NoTraining):
        ex.run_experiment(ctx, "a1_appearance", 0)
    with pytest.raises(_NoTraining):
        ex.run_experiment(ctx, "a1_appearance", 0, smoke=True)

def test_edited_experiment_refused_for_new_seed(tmp_path, monkeypatch, no_training):
    ctx = _experiment_ctx(tmp_path)
    ctx.comparability_dir.mkdir(parents=True)
    (ctx.comparability_dir / "result.json").write_text(json.dumps({"identical": True}))
    monkeypatch.setattr(tr, "git_state", lambda: {"commit": "abc", "dirty": False})
    cfg, _ = ex.build_train(ctx.spec, "a1_appearance", 0, OPT)
    seed0 = ctx.runs / "a1_appearance" / "seed0"
    seed0.mkdir(parents=True)
    (seed0 / "run_meta.json").write_text(json.dumps({"config": {**cfg.__dict__, "imgsz": 768}}))
    with pytest.raises(RuntimeError, match="different config"):
        ex.run_experiment(ctx, "a1_appearance", 1)

def test_two_class_experiment_gates_on_derived_dataset(tmp_path, no_training):
    ctx = _experiment_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="not built"):
        ex.run_experiment(ctx, "a4_two_class", 0, check_only=True)
    from tvision.two_class import run_two_class
    assert run_two_class(ctx.paths, CLASSES, ctx.two_class)
    assert ex.run_experiment(ctx, "a4_two_class", 0, check_only=True)

def test_comparability_view_keeps_three_class_schema(tmp_path):
    paths = _verified(tmp_path)
    view = tmp_path / "view"
    check = ex.build_view(paths.processed_dir, view, CLASSES, ["ambiguous"])
    assert check["passed"]
    data = yaml.safe_load((view / "data.yaml").read_text())
    assert data["names"] == dict(enumerate(CLASSES.names)) and data["val"] == "images/val"
    assert not (view / "images" / "test").exists() and not (view / "images" / "train").exists()