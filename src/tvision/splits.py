"""Step 2: leakage-safe train/val/test assignment.

Test = the official fold_<k>_test slides. Val = whole hospitals held out from fold_<k>_train, chosen by
exhaustive search over hospital subsets. Train = the rest of fold_<k>_train. Patients and slides are
nested in hospitals, so hospital-level separation also prevents patient- and slide-level leakage.
"""

import json
from fractions import Fraction
from itertools import combinations

import pandas as pd

from tvision import nucls_data as nd
from tvision.config import Classes, Paths, Splits
from tvision.explore import _block

SPLITS = ["train", "val", "test"]
GROUP_KEYS = ["slide", "patient", "hospital"]

def select_val_hospitals(stats: pd.DataFrame, fraction: float) -> list[str]:
    """Nonempty proper hospital subset minimizing, in order: |val FOVs - fraction * total FOVs|,
    |primary share in val - primary share overall| (share of `primary` among `headline` nuclei),
    then the sorted hospital codes lexicographically.

    `stats` is indexed by hospital with columns `fovs`, `primary`, `headline`.
    """
    hospitals = sorted(stats.index)
    if len(hospitals) < 2:
        raise ValueError("At least two hospitals are needed to hold out a validation set")
    s = {h: tuple(int(v) for v in stats.loc[h, ["fovs", "primary", "headline"]]) for h in hospitals}
    target = Fraction(fraction) * sum(v[0] for v in s.values())
    overall = Fraction(sum(v[1] for v in s.values()), sum(v[2] for v in s.values()))
    best, best_key = None, None
    for r in range(1, len(hospitals)):
        for subset in combinations(hospitals, r):
            fovs, primary, headline = (sum(col) for col in zip(*(s[h] for h in subset)))
            share_gap = abs(Fraction(primary, headline) - overall) if headline else float("inf")
            key = (abs(fovs - target), share_gap, subset)
            if best_key is None or key < best_key:
                best, best_key = list(subset), key
    return best

def assign_splits(fovs: pd.DataFrame, fold: int, test_slides: set[str], train_slides: set[str],
                  val_hospitals: list[str]) -> pd.DataFrame:
    """Add `split` and `source`; FOVs whose slide is in neither official file get no split."""
    out = fovs.copy()
    is_test = out["slide"].isin(test_slides)
    is_train = out["slide"].isin(train_slides) & ~is_test
    is_val = is_train & out["hospital"].isin(val_hospitals)
    out["split"] = None
    out.loc[is_train, "split"] = "train"
    out.loc[is_val, "split"] = "val"
    out.loc[is_test, "split"] = "test"
    out["source"] = None
    out.loc[is_train, "source"] = f"fold_{fold}_train.csv"
    out.loc[is_val, "source"] = f"fold_{fold}_train.csv:val-hospital-holdout"
    out.loc[is_test, "source"] = f"fold_{fold}_test.csv"
    return out

def check_splits(splits: pd.DataFrame, counts: pd.DataFrame, test_file: pd.DataFrame,
                 train_file: pd.DataFrame) -> dict[str, dict]:
    """Split invariants. `counts` is split x project class nuclei. Each entry: {'passed': bool, 'detail': ...}."""
    unassigned = int(splits["split"].isna().sum())
    duplicated = int(splits["stem"].duplicated().sum())
    overlap = {k: int((splits.dropna(subset=["split"]).groupby(k)["split"].nunique() > 1).sum())
               for k in GROUP_KEYS}
    by_split = {s: set(splits.loc[splits["split"] == s, "slide"]) for s in SPLITS}
    official_test, official_train = set(test_file["slide_name"]), set(train_file["slide_name"])
    files = pd.concat([test_file, train_file])
    code_mismatch = files[files["slide_name"].map(nd.hospital_of) != files["hospital"]]
    missing_classes = {s: sorted(counts.columns[counts.loc[s] == 0]) if s in counts.index else list(counts.columns)
                       for s in SPLITS}
    missing_classes = {s: m for s, m in missing_classes.items() if m}
    return {
        "every_fov_assigned_once": {"passed": not unassigned and not duplicated,
                                    "detail": {"unassigned": unassigned, "duplicated": duplicated}},
        "no_group_overlap_across_splits": {"passed": not any(overlap.values()), "detail": overlap},
        "test_matches_official": {
            "passed": by_split["test"] == official_test,
            "detail": {"missing": len(official_test - by_split["test"]),
                       "extra": len(by_split["test"] - official_test)},
        },
        "train_val_match_official": {
            "passed": by_split["train"] | by_split["val"] == official_train,
            "detail": {"missing": len(official_train - by_split["train"] - by_split["val"]),
                       "extra": len((by_split["train"] | by_split["val"]) - official_train)},
        },
        "fold_hospital_codes_match_slides": {"passed": code_mismatch.empty, "detail": len(code_mismatch)},
        "every_split_has_every_class": {"passed": not missing_classes, "detail": missing_classes},
    }

def split_summary(splits: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    groups = splits.groupby("split").agg(hospitals=("hospital", "nunique"), patients=("patient", "nunique"),
                                         slides=("slide", "nunique"), fovs=("stem", "size"))
    groups["fov_pct"] = (100 * groups["fovs"] / groups["fovs"].sum()).round(1)
    pct = (100 * counts.div(counts.sum(axis=1), axis=0)).round(1).add_suffix("_pct")
    return groups.join(counts).join(pct).reindex(SPLITS)

def make_splits() -> bool:
    paths, classes, cfg = Paths.load(), Classes.load(), Splits.load()
    raw, out = paths.raw_dir, paths.artifacts_dir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    fovs = nd.fov_table(raw)
    ann = nd.map_classes(nd.load_annotations(raw), classes)
    ann = ann[~ann["type"].isin(classes.drop_types)]
    per_fov = pd.crosstab(ann["stem"], ann["project_class"]).reindex(columns=classes.names, fill_value=0)
    test_file = nd.load_split_file(raw, cfg.test_fold, "test")
    train_file = nd.load_split_file(raw, cfg.test_fold, "train")
    train_slides = set(train_file["slide_name"])

    primary = classes.headline[0]
    fold_train = fovs[fovs["slide"].isin(train_slides)].join(per_fov, on="stem")
    stats = fold_train.groupby("hospital").agg(fovs=("stem", "size"), primary=(primary, "sum"))
    stats["headline"] = fold_train.groupby("hospital")[classes.headline].sum().sum(axis=1)
    val_hospitals = select_val_hospitals(stats, cfg.val_fraction)

    splits = assign_splits(fovs, cfg.test_fold, set(test_file["slide_name"]), train_slides, val_hospitals)
    counts = splits.join(per_fov, on="stem").groupby("split")[classes.names].sum()
    checks = check_splits(splits, counts, test_file, train_file)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    for name, c in checks.items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {name}: {c['detail']}")

    achieved = stats.loc[val_hospitals, "fovs"].sum() / stats["fovs"].sum()
    headline_share = (counts[primary] / counts[classes.headline].sum(axis=1)).reindex(SPLITS).round(3)
    sections = [
        _block("Config", f"test_fold: {cfg.test_fold}  val_fraction: {cfg.val_fraction}"),
        _block("Checks", json.dumps(checks, indent=2)),
        _block("Validation hospitals", f"{val_hospitals}\nval FOVs / fold-train FOVs: target {cfg.val_fraction:.3f}, "
                                       f"achieved {achieved:.3f}"),
        _block(f"Fold-{cfg.test_fold} train hospitals (candidates)", stats.assign(val=stats.index.isin(val_hospitals))),
        _block("Splits (nuclei exclude dropped types)", split_summary(splits, counts)),
        _block(f"{primary} share of headline nuclei ({' + '.join(classes.headline)})", headline_share),
    ]
    (out / "summary.md").write_text("# NuCLS splits\n\n" + "\n".join(sections))
    splits.sort_values("stem")[["stem", "slide", "patient", "hospital", "split", "source"]].to_csv(
        paths.processed_dir / "splits.csv", index=False)
    print(f"Validation hospitals: {val_hospitals} (achieved {achieved:.3f}, target {cfg.val_fraction})")
    print(f"Splits: {paths.processed_dir / 'splits.csv'}  Report: {out / 'summary.md'}")
    return all(c["passed"] for c in checks.values())