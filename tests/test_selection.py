import copy
import json
from dataclasses import asdict, replace

import pandas as pd
import pytest

from tvision import experiments as ex
from tvision import selection as sel

from test_experiments import OPT, _anchored, _fake_baseline

def _runs(name: str, compare_to: str | None, headline: list[float], tumor: list[float] | None = None,
          map50: list[float] | None = None, imgsz: int = 640, model: str = "yolo26n.pt", factors: str = "-",
          non_pathological: bool = True, seeds=(0, 1, 2), **extra) -> list[dict]:
    return [{"experiment": name, "seed": s, "compare_to": compare_to, "factors": factors, "imgsz": imgsz,
             "model": model, sel.H: h, sel.T: (tumor or headline)[i], sel.H50: (map50 or [0.6] * 3)[i],
             sel.H75: 0.18, "non_pathological": non_pathological, "val_box_min_epoch": 10,
             "val_box_rise_to_best": 0.1, **extra}
            for i, (s, h) in enumerate(zip(seeds, headline))]

def _rows(*groups) -> pd.DataFrame:
    return pd.DataFrame([r for g in groups for r in g])

CONTROL = _runs("a0_control", None, [0.270, 0.265, 0.275])

def test_qualified_finalist_is_selected_with_median_seed():
    rows = _rows(CONTROL, _runs("a1_appearance", "a0_control", [0.290, 0.300, 0.285], factors="appearance"))
    decision = sel.choose(rows)
    assert decision["experiment"] == "a1_appearance" and decision["seed"] == 0
    e = decision["evaluations"]["a1_appearance"]
    assert e["qualified"] and e["threshold"] == pytest.approx(0.01) and e["gain"] == pytest.approx(0.021667, abs=1e-6)

@pytest.mark.parametrize("finalist, failed", [
    (_runs("a1", "a0_control", [0.300, 0.300, 0.272]), "paired_positive_all_seeds"),
    (_runs("a1", "a0_control", [0.278, 0.274, 0.284]), "gain_at_least_threshold"),
    (_runs("a1", "a0_control", [0.30, 0.30, 0.30], tumor=[0.26, 0.255, 0.265]), "tumor_within_tolerance"),
    (_runs("a1", "a0_control", [0.30, 0.30, 0.30], map50=[0.58, 0.585, 0.585]), "headline_mAP50_within_tolerance"),
    (_runs("a1", "a0_control", [0.30, 0.30, 0.30], non_pathological=False), "all_runs_non_pathological"),
])
def test_each_failed_condition_falls_back_to_control(finalist, failed):
    decision = sel.choose(_rows(CONTROL, finalist))
    assert decision["experiment"] == sel.FALLBACK
    assert not decision["evaluations"]["a1"]["conditions"][failed]

def test_threshold_uses_control_seed_sd_when_larger():
    noisy = _runs("a0_control", None, [0.25, 0.27, 0.29])
    decision = sel.choose(_rows(noisy, _runs("a1", "a0_control", [0.275, 0.285, 0.300])))
    e = decision["evaluations"]["a1"]
    assert e["threshold"] == 0.02 and e["gain"] == pytest.approx(0.016667, abs=1e-6)
    assert e["conditions"]["paired_positive_all_seeds"] and not e["conditions"]["gain_at_least_threshold"]

def test_cheaper_config_wins_within_tolerance():
    rows = _rows(CONTROL, _runs("a1", "a0_control", [0.290, 0.295, 0.292], factors="appearance"),
                 _runs("a3", "a0_control", [0.298, 0.300, 0.297], imgsz=768, factors="imgsz"))
    assert sel.choose(rows)["experiment"] == "a1"

def test_gain_is_measured_against_matched_batch_control():
    b8 = _runs("a0_control_b8", "a0_control", [0.300, 0.300, 0.300], factors="batch")
    a3 = _runs("a3_b8", "a0_control_b8", [0.305, 0.305, 0.305], imgsz=768, factors="imgsz")
    decision = sel.choose(_rows(CONTROL, b8, a3))
    assert decision["evaluations"]["a3_b8"]["compare_to"] == "a0_control_b8"
    assert not decision["evaluations"]["a3_b8"]["qualified"]

def test_fallback_requires_all_control_seeds():
    with pytest.raises(RuntimeError, match="fallback"):
        sel.choose(_rows(_runs("a0_control", None, [0.27], seeds=(0,)), _runs("a1", "a0_control", [0.2, 0.2, 0.2])))

def test_incomplete_finalists_are_ignored():
    rows = _rows(CONTROL, _runs("a1", "a0_control", [0.4], seeds=(0,)))
    assert sel.choose(rows)["finalists"] == []

def test_stage_a_flags_use_fixed_thresholds():
    rows = _rows(_runs("a0_control", None, [0.27], seeds=(0,), val_box_rise_to_best=0.2),
                 _runs("a1", "a0_control", [0.275], tumor=[0.266], seeds=(0,), val_box_rise_to_best=0.15),
                 _runs("a2", "a0_control", [0.274], seeds=(0,), val_box_min_epoch=15, val_box_rise_to_best=0.3),
                 _runs("a3", "a0_control", [0.281], seeds=(0,), val_box_rise_to_best=0.2))
    rows.loc[rows["experiment"] == "a3", sel.H75] = 0.19
    flags = sel.stage_a(rows).set_index("experiment")
    assert flags.loc["a1", "helped"] and flags.loc["a1", "less_overfitting"]
    assert not flags.loc["a2", "helped"] and flags.loc["a2", "less_overfitting"]
    assert flags.loc["a3", "clear_localization_gain"] and not flags.loc["a1", "clear_localization_gain"]
    assert not flags.loc["a3", "less_overfitting"]

def test_stage_a_tumor_guard():
    rows = _rows(_runs("a0_control", None, [0.27], tumor=[0.27], seeds=(0,)),
                 _runs("a1", "a0_control", [0.28], tumor=[0.264], seeds=(0,)))
    assert not sel.stage_a(rows).set_index("experiment").loc["a1", "helped"]

def _fake_run(ctx: ex.Context, name: str, seed: int, headline: float) -> None:
    cfg, dataset = ex.build_train(ctx.spec, name, seed, OPT)
    run_dir = ctx.runs / name / f"seed{seed}"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(f"{name}{seed}".encode())
    entry = ctx.spec["experiments"][name]
    (run_dir / "run_meta.json").write_text(json.dumps({"experiment": name, "factors": entry["factors"],
                                                       "compare_to": entry["compare_to"], "dataset": dataset,
                                                       "config": asdict(cfg)}))
    per_class = [{"Class": c, "mAP50": 0.6, "mAP75": 0.18, "mAP50-95": headline} for c in ("tumor", "non_tumor")]
    (run_dir / "val_best.json").write_text(json.dumps({"per_class": per_class, "headline_mean": {
        "mAP50": 0.6, "mAP75": 0.18, "mAP50-95": headline}}))
    (run_dir / "diagnostics.json").write_text(json.dumps({
        "best_epoch": 20, "overfitting": {"val_box_min_epoch": 10, "rise_to_best": 0.1}, "plateau": {"sd": 0.01},
        "stability": {"best_minus_window_mean": 0.01}, "non_pathological": {"passed": True}}))
    report = ctx.artifacts / name / f"seed{seed}"
    report.mkdir(parents=True)
    (report / "checks.json").write_text(json.dumps({"gate": {"passed": True}}))

def test_compare_and_select_on_run_folders(tmp_path):
    metrics = {"Box-P": 0.5, "Box-R": 0.6, "Box-F1": 0.55, "mAP50": 0.6, "mAP75": 0.2, "mAP50-95": 0.27}
    _fake_baseline(tmp_path)
    (tmp_path / "runs" / "baseline" / "val_best.json").write_text(json.dumps({
        "per_class": [{"Class": "tumor", **metrics}], "headline_mean": metrics}))
    ctx = _anchored(tmp_path)
    for seed, (c, a) in enumerate([(0.27, 0.29), (0.265, 0.30), (0.275, 0.285)]):
        _fake_run(ctx, "a0_control", seed, c)
        _fake_run(ctx, "a1_appearance", seed, a)
    assert sel.compare(ctx)
    assert (ctx.artifacts / "comparison.md").is_file() and len(pd.read_csv(ctx.artifacts / "comparison.csv")) == 6
    assert sel.select(ctx)
    selection = json.loads((ctx.artifacts / "selection.json").read_text())
    assert (selection["experiment"], selection["seed"]) == ("a1_appearance", 0)
    assert selection["best_pt_sha256"] == ex._sha256(ctx.runs / "a1_appearance" / "seed0" / "weights" / "best.pt")
    with pytest.raises(RuntimeError, match="already exists"):
        sel.select(ctx)

def test_edited_experiment_blocks_selection(tmp_path):
    ctx = _anchored(tmp_path)
    for seed in range(3):
        _fake_run(ctx, "a0_control", seed, 0.27)
    spec = copy.deepcopy(ctx.spec)
    spec["control"]["patience"] = 10
    edited = replace(ctx, spec=spec)
    assert not sel.run_rows(edited)["config_matches_yaml"].any()
    with pytest.raises(RuntimeError, match="edited"):
        sel.select(edited)