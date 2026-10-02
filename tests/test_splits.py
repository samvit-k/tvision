import pandas as pd

from tvision import nucls_data as nd
from tvision.splits import assign_splits, check_splits, select_val_hospitals

def _stats(rows: dict[str, tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index", columns=["fovs", "primary", "headline"])

def _fovs() -> pd.DataFrame:
    slides = ["TCGA-AA-0001-DX1", "TCGA-AA-0002-DX1", "TCGA-BB-0003-DX1", "TCGA-CC-0004-DX1", "TCGA-DD-0005-DX1"]
    rows = [{"stem": f"{s}_id-{i}", "slide": s, "patient": nd.patient_of(s), "hospital": nd.hospital_of(s)}
            for s in slides for i in range(2)]
    return pd.DataFrame(rows)

def _official(slides: list[str], part: str) -> pd.DataFrame:
    return pd.DataFrame({"slide_name": slides, "hospital": [nd.hospital_of(s) for s in slides], "type": part})

TEST_FILE = _official(["TCGA-DD-0005-DX1"], "test")
TRAIN_FILE = _official(["TCGA-AA-0001-DX1", "TCGA-AA-0002-DX1", "TCGA-BB-0003-DX1", "TCGA-CC-0004-DX1"], "train")
COUNTS = pd.DataFrame({"tumor": [5, 2, 3], "non_tumor": [4, 1, 2]}, index=["train", "val", "test"])

def _splits(val_hospitals=("CC",)) -> pd.DataFrame:
    return assign_splits(_fovs(), 1, set(TEST_FILE["slide_name"]), set(TRAIN_FILE["slide_name"]),
                         list(val_hospitals))

def test_patient_parsing():
    assert nd.patient_of("TCGA-A1-A0SK-DX1") == "TCGA-A1-A0SK"

def test_selects_subset_closest_to_target():
    stats = _stats({"A": (50, 5, 10), "B": (30, 5, 10), "C": (15, 5, 10), "D": (5, 5, 10)})
    assert select_val_hospitals(stats, 0.2) == ["C", "D"]  # target 20 FOVs; C+D is exact

def test_size_tie_broken_by_class_similarity():
    # overall tumor share 0.5; B (20 FOVs) is 0.5, A (20 FOVs) is 0.9
    stats = _stats({"A": (20, 9, 10), "B": (20, 5, 10), "C": (60, 36, 80)})
    assert select_val_hospitals(stats, 0.2) == ["B"]

def test_exact_tie_broken_lexicographically_and_order_independent():
    stats = _stats({"B": (20, 5, 10), "A": (20, 5, 10), "C": (60, 30, 60)})
    assert select_val_hospitals(stats, 0.2) == ["A"]
    assert select_val_hospitals(stats.iloc[::-1], 0.2) == ["A"]

def test_assignment_and_checks_pass():
    splits = _splits()
    assert splits["split"].notna().all() and not splits["stem"].duplicated().any()
    assert splits.groupby("hospital")["split"].nunique().max() == 1
    assert set(splits.loc[splits["hospital"] == "CC", "split"]) == {"val"}
    assert splits.loc[splits["split"] == "val", "source"].unique().tolist() == ["fold_1_train.csv:val-hospital-holdout"]
    assert all(c["passed"] for c in check_splits(splits, COUNTS, TEST_FILE, TRAIN_FILE).values())

def test_injected_leakage_fails():
    splits = _splits()
    splits.loc[splits["stem"] == "TCGA-AA-0001-DX1_id-0", "split"] = "val"  # same slide/patient/hospital in two splits
    checks = check_splits(splits, COUNTS, TEST_FILE, TRAIN_FILE)
    assert not checks["no_group_overlap_across_splits"]["passed"]
    assert checks["no_group_overlap_across_splits"]["detail"] == {"slide": 1, "patient": 1, "hospital": 1}

def test_missing_official_slide_fails():
    splits = _splits()
    splits = splits[splits["slide"] != "TCGA-DD-0005-DX1"]
    checks = check_splits(splits, COUNTS, TEST_FILE, TRAIN_FILE)
    assert not checks["test_matches_official"]["passed"]
    assert checks["test_matches_official"]["detail"]["missing"] == 1

def test_unassigned_fov_and_missing_class_fail():
    fovs = pd.concat([_fovs(), pd.DataFrame([{"stem": "x", "slide": "TCGA-EE-0006-DX1",
                                              "patient": "TCGA-EE-0006", "hospital": "EE"}])])
    splits = assign_splits(fovs, 1, set(TEST_FILE["slide_name"]), set(TRAIN_FILE["slide_name"]), ["CC"])
    counts = COUNTS.assign(tumor=[5, 0, 3])
    checks = check_splits(splits, counts, TEST_FILE, TRAIN_FILE)
    assert not checks["every_fov_assigned_once"]["passed"]
    assert checks["every_fov_assigned_once"]["detail"]["unassigned"] == 1
    assert checks["every_split_has_every_class"]["detail"] == {"val": ["tumor"]}

def test_fold_hospital_code_mismatch_fails():
    bad = TEST_FILE.assign(hospital="ZZ")
    assert not check_splits(_splits(), COUNTS, bad, TRAIN_FILE)["fold_hospital_codes_match_slides"]["passed"]