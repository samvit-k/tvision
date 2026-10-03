"""Step 5A: comparison of experiment runs and the validation-only final selection rule fixed before the runs.

All numbers come from each run's post-training val pass of best.pt (val_best.json, or val_view.json when the
comparability check found headline AP not identical on the ambiguous-removed view). Nothing here reads test data.
"""

from tvision import experiments as ex  # imports tvision.train first, which configures Ultralytics

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from tvision import train as tr
from tvision.explore import _block

STAGE_A_GAIN = 0.005
TUMOR_TOL = 0.005
LOCALIZATION_GAIN = 0.01
LATER_MIN_EPOCHS = 5
RISE_REDUCTION = 0.25
FINAL_MIN_GAIN = 0.01
MAP50_TOL = 0.01
SEEDS = (0, 1, 2)
FALLBACK = "a0_control"
PRIMARY = "tumor"
MODEL_SIZES = "nsmlx"
H, H50, H75 = "headline_mAP50-95", "headline_mAP50", "headline_mAP75"
T = f"{PRIMARY}_mAP50-95"
RULE = {"stage_a_gain": STAGE_A_GAIN, "tumor_tolerance": TUMOR_TOL, "localization_gain": LOCALIZATION_GAIN,
        "later_min_epochs": LATER_MIN_EPOCHS, "rise_reduction": RISE_REDUCTION, "final_min_gain": FINAL_MIN_GAIN,
        "headline_mAP50_tolerance": MAP50_TOL, "seeds": list(SEEDS), "fallback": FALLBACK}

def _d(a: float, b: float) -> float:
    """Difference rounded to 6 decimals, so thresholds are not missed by float noise on 5-decimal metrics."""
    return round(float(a) - float(b), 6)

def _has_control(compare_to) -> bool:
    return isinstance(compare_to, str)  # pandas may hold a missing compare_to as None or NaN

def model_size(model: str) -> int:
    letter = Path(model).stem[-1]
    return MODEL_SIZES.index(letter) if letter in MODEL_SIZES else len(MODEL_SIZES)

def config_matches(ctx: ex.Context, meta: dict) -> bool:
    """The run's recorded config equals what experiments.yaml resolves to today (edited experiments are flagged)."""
    try:
        cfg, _ = ex.build_train(ctx.spec, meta["experiment"], meta["config"]["seed"], meta["config"]["optimizer"])
    except ValueError:
        return False
    return asdict(cfg) == meta["config"]

def run_rows(ctx: ex.Context) -> pd.DataFrame:
    comparability = ex.read_comparability(ctx)
    rows = []
    for meta_path in sorted(ctx.runs.glob("*/seed*/run_meta.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        name, seed = meta["experiment"], meta["config"]["seed"]
        val = json.loads((run_dir / "val_best.json").read_text())
        use_view = meta["dataset"] == "three_class" and comparability is not None and not comparability["identical"]
        scored = json.loads((run_dir / "val_view.json").read_text()) if use_view else val
        per = {r["Class"]: r for r in val["per_class"]} | {r["Class"]: r for r in scored["per_class"]}
        diag = json.loads((run_dir / "diagnostics.json").read_text())
        checks = json.loads((ctx.artifacts / name / f"seed{seed}" / "checks.json").read_text())
        rows.append({
            "experiment": name, "seed": seed, "factors": "+".join(meta["factors"]) or "-",
            "compare_to": meta["compare_to"], "dataset": meta["dataset"], "model": meta["config"]["model"],
            "imgsz": meta["config"]["imgsz"], "batch": meta["config"]["batch"],
            H: scored["headline_mean"]["mAP50-95"], H50: scored["headline_mean"]["mAP50"],
            H75: scored["headline_mean"]["mAP75"],
            **{f"{c}_mAP50-95": per[c]["mAP50-95"] if c in per else np.nan for c in ctx.classes.names},
            "best_epoch": diag["best_epoch"], "val_box_min_epoch": diag["overfitting"]["val_box_min_epoch"],
            "val_box_rise_to_best": diag["overfitting"]["rise_to_best"], "plateau_sd": diag["plateau"]["sd"],
            "stability_gap": diag["stability"]["best_minus_window_mean"],
            "non_pathological": diag["non_pathological"]["passed"],
            "checks_passed": all(c["passed"] for c in checks.values()),
            "config_matches_yaml": config_matches(ctx, meta), "run_dir": str(run_dir),
        })
    return pd.DataFrame(rows)

def stage_a(rows: pd.DataFrame) -> pd.DataFrame:
    """Seed-0 comparison of each experiment with its matched control, with the Stage A/B decision flags."""
    seed0 = rows[rows["seed"] == 0].set_index("experiment")
    out = []
    for name, r in seed0.iterrows():
        if not _has_control(r["compare_to"]) or r["compare_to"] not in seed0.index:
            continue
        c = seed0.loc[r["compare_to"]]
        dh, dt, d75 = _d(r[H], c[H]), _d(r[T], c[T]), _d(r[H75], c[H75])
        later = int(r["val_box_min_epoch"]) - int(c["val_box_min_epoch"]) >= LATER_MIN_EPOCHS
        smaller_rise = c["val_box_rise_to_best"] > 0 and (
            r["val_box_rise_to_best"] <= (1 - RISE_REDUCTION) * c["val_box_rise_to_best"])
        out.append({"experiment": name, "compare_to": r["compare_to"], "d_headline_mAP50-95": dh, "d_tumor_mAP50-95": dt,
                    "d_headline_mAP75": d75, "non_pathological": bool(r["non_pathological"]),
                    "helped": dh >= STAGE_A_GAIN and dt >= -TUMOR_TOL and bool(r["non_pathological"]),
                    "less_overfitting": bool(later or smaller_rise),
                    "clear_localization_gain": dh >= LOCALIZATION_GAIN and d75 >= LOCALIZATION_GAIN})
    return pd.DataFrame(out)

def config_table(rows: pd.DataFrame) -> pd.DataFrame:
    cols = [H, H50, H75, T]
    g = rows.groupby("experiment")
    table = g[cols].agg(["mean", "std"])
    table.columns = [f"{c} {s}" for c, s in table.columns]
    table.insert(0, "seeds", g["seed"].apply(lambda s: ",".join(map(str, sorted(s)))))
    return table

def choose(rows: pd.DataFrame, seeds=SEEDS) -> dict:
    """Apply the final selection rule to per-run rows; returns the chosen experiment, seed and the evaluation."""
    def runs(name: str) -> pd.DataFrame:
        return rows[rows["experiment"] == name].set_index("seed")

    def complete(name: str) -> bool:
        return set(seeds) <= set(runs(name).index)

    finalists = sorted(e for e in rows["experiment"].unique()
                       if _has_control(runs(e)["compare_to"].iloc[0]) and complete(e))
    evaluations = {}
    for f in finalists:
        fr = runs(f).loc[list(seeds)]
        ctrl = fr["compare_to"].iloc[0]
        if not complete(ctrl):
            evaluations[f] = {"compare_to": ctrl, "qualified": False, "reason": f"{ctrl} lacks seeds {list(seeds)}"}
            continue
        cr = runs(ctrl).loc[list(seeds)]
        control_sd = round(float(cr[H].std(ddof=1)), 6)
        threshold = max(FINAL_MIN_GAIN, control_sd)
        gain = _d(fr[H].mean(), cr[H].mean())
        paired = {int(s): _d(fr.loc[s, H], cr.loc[s, H]) for s in seeds}
        conditions = {
            "gain_at_least_threshold": gain >= threshold,
            "paired_positive_all_seeds": all(v > 0 for v in paired.values()),
            "tumor_within_tolerance": _d(fr[T].mean(), cr[T].mean()) >= -TUMOR_TOL,
            "headline_mAP50_within_tolerance": _d(fr[H50].mean(), cr[H50].mean()) >= -MAP50_TOL,
            "all_runs_non_pathological": bool(fr["non_pathological"].all()),
        }
        evaluations[f] = {"compare_to": ctrl, "mean": float(fr[H].mean()), "control_mean": float(cr[H].mean()),
                          "control_sd": control_sd, "threshold": threshold, "gain": gain, "paired_diff": paired,
                          "conditions": conditions, "qualified": all(conditions.values())}
    qualified = [f for f, e in evaluations.items() if e["qualified"]]
    if qualified:
        best = max(qualified, key=lambda f: evaluations[f]["mean"])
        tolerance = max(FINAL_MIN_GAIN, evaluations[best]["control_sd"])
        near = [f for f in qualified if _d(evaluations[best]["mean"], evaluations[f]["mean"]) <= tolerance]

        def cost(f: str) -> tuple:
            r = runs(f).iloc[0]
            return int(r["imgsz"]), model_size(r["model"]), len(r["factors"].split("+")), -evaluations[f]["mean"]

        chosen = min(near, key=cost)
        reason = (f"{chosen} qualified; best mean {best}, configs within {tolerance:.4f} of it: {near}; "
                  "the cheapest of those was chosen (imgsz, model size, number of factors)")
    else:
        chosen, reason = FALLBACK, "no finalist met every condition, so the control config is selected"
        if not complete(FALLBACK):
            raise RuntimeError(f"{FALLBACK} lacks seeds {list(seeds)}; it is required as the fallback")
    ordered = runs(chosen).loc[list(seeds)].reset_index().sort_values([H, "seed"])
    median = ordered.iloc[len(ordered) // 2]
    return {"experiment": chosen, "seed": int(median["seed"]), "reason": reason, "finalists": finalists,
            "evaluations": evaluations, "seed_scores": {int(r["seed"]): float(r[H]) for _, r in ordered.iterrows()}}

def compare(ctx: ex.Context) -> bool:
    ex.require_baseline(ctx)
    rows = run_rows(ctx)
    ctx.artifacts.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((ctx.baseline_dir / "val_best.json").read_text())
    reference = pd.DataFrame(recorded["per_class"]).set_index("Class")
    sections = [_block("Reference: runs/baseline best.pt on val (all-class fitness, frozen)",
                       pd.concat([reference[tr.METRIC_COLUMNS],
                                  pd.DataFrame([recorded["headline_mean"]], index=["headline mean"])]).round(4))]
    edited = []
    if not rows.empty:
        rows.to_csv(ctx.artifacts / "comparison.csv", index=False)
        edited = sorted(rows.loc[~rows["config_matches_yaml"], "experiment"].unique())
        shown = rows.drop(columns=["run_dir"]).set_index(["experiment", "seed"])
        sections += [_block("Runs (best.pt on val)", shown.round(4).T),
                     _block("Per config over seeds", config_table(rows).round(4).T)]
        a = stage_a(rows)
        if not a.empty:
            sections.append(_block("Seed 0 vs matched control (Stage A/B flags)", a.set_index("experiment").T))
        repro = ctx.artifacts / ex.REPRODUCES_BASELINE / "seed0" / "checks.json"
        if repro.is_file():
            sections.append(_block("a0_control seed 0 vs runs/baseline (reproducibility)",
                                   json.dumps(json.loads(repro.read_text()).get("reproduces_baseline"), indent=2)))
    sections.append(_block("Experiments edited after running", edited or "none"))
    (ctx.artifacts / "comparison.md").write_text("# Step 5A comparison (validation only)\n\n" + "\n".join(sections))
    after = ex.baseline_check(ctx)
    print(f"[{'PASS' if after['passed'] else 'FAIL'}] baseline_untouched after compare\n"
          f"Edited experiments: {edited or 'none'}\nReport: {ctx.artifacts / 'comparison.md'}")
    return after["passed"] and not edited

def select(ctx: ex.Context) -> bool:
    before = ex.require_baseline(ctx)
    out = ctx.artifacts / "selection.json"
    if out.exists():
        raise RuntimeError(f"{out} already exists; the selection is made once")
    rows = run_rows(ctx)
    if rows.empty:
        raise RuntimeError("no Step 5A runs found")
    if edited := sorted(rows.loc[~rows["config_matches_yaml"], "experiment"].unique()):
        raise RuntimeError(f"experiments edited after running: {edited}")
    decision = choose(rows)
    run = rows[(rows["experiment"] == decision["experiment"]) & (rows["seed"] == decision["seed"])].iloc[0]
    best = Path(run["run_dir"]) / "weights" / "best.pt"
    meta = json.loads((Path(run["run_dir"]) / "run_meta.json").read_text())
    after = ex.baseline_check(ctx)
    if not after["passed"]:
        raise RuntimeError(f"baseline_untouched failed during selection: {after['detail']}")
    comparability = ex.read_comparability(ctx)
    selection = decision | {
        "run_dir": run["run_dir"], "best_pt": str(best), "best_pt_sha256": ex._sha256(best),
        "dataset": meta["dataset"], "config": meta["config"], "rule": RULE,
        "comparability_identical": comparability["identical"] if comparability else None,
        "manifest_sha256": before["detail"]["manifest_sha256"], "git": tr.git_state(),
    }
    tr._write_json(out, selection)
    print(f"Selected {decision['experiment']} seed {decision['seed']}: {decision['reason']}\n"
          f"best.pt {selection['best_pt_sha256'][:16]}…\nSelection: {out}")
    return True

def compare_runs() -> bool:
    return compare(ex.Context.load())

def select_final() -> bool:
    return select(ex.Context.load())