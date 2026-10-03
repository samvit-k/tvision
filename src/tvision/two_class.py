"""Step 5A: two-class dataset derived from the verified one, with excluded classes removed rather than relabeled.

Only the configured splits (train and val) are derived; test labels are not read or derived until Step 6.
Images are byte copies. Each label file keeps the source lines in order minus the excluded classes, whose ids
must be the trailing ones so kept ids are unchanged. The dataset is built in a temporary folder and published
only if every check passes; an existing dataset is re-checked, never rebuilt.
"""

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from tvision.config import Classes, Paths, TwoClass
from tvision.explore import _block
from tvision.train import gate
from tvision.verify import dataset_fingerprint

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def kept_classes(classes: Classes, exclude: list[str]) -> Classes:
    """The class list without `exclude`; refuses exclusions that would renumber kept classes or drop a headline class."""
    if unknown := sorted(set(exclude) - set(classes.names)):
        raise ValueError(f"cannot exclude unknown classes {unknown}")
    kept = [n for n in classes.names if n not in exclude]
    if classes.names[:len(kept)] != kept:
        raise ValueError(f"excluded classes {exclude} must be the trailing class ids so kept ids are unchanged")
    if set(exclude) & set(classes.headline):
        raise ValueError(f"cannot exclude headline classes {sorted(set(exclude) & set(classes.headline))}")
    return replace(classes, names=kept, super_to_class={k: v for k, v in classes.super_to_class.items() if v in kept})

def filter_lines(text: str, n_kept: int) -> list[str]:
    return [line for line in text.splitlines() if int(line.split()[0]) < n_kept]

def data_yaml(out: Path, names: list[str], splits: list[str]) -> dict:
    return {"path": str(out), **{s: f"images/{s}" for s in splits}, "names": dict(enumerate(names))}

def derive(source: Path, root: Path, final: Path, kept: Classes, splits: list[str]) -> None:
    """Write images, labels and data.yaml under `root`; data.yaml points at the published location `final`."""
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for img in sorted((source / "images" / split).glob("*.png")):
            shutil.copyfile(img, root / "images" / split / img.name)
            lines = filter_lines((source / "labels" / split / f"{img.stem}.txt").read_text(), len(kept.names))
            (root / "labels" / split / f"{img.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml(final, kept.names, splits), sort_keys=False))

def _stems(folder: Path, suffix: str) -> set[str]:
    return {p.stem for p in folder.glob(f"*{suffix}")} if folder.is_dir() else set()

def run_checks(source: Path, root: Path, final: Path, classes: Classes, kept: Classes, splits: list[str]) -> dict:
    n_kept = len(kept.names)
    checks = {}
    top = {kind: sorted(p.name for p in (root / kind).iterdir() if p.is_dir()) for kind in ("images", "labels")}
    stray = [f"{kind}/{name}" for kind, names in top.items() for name in names if name not in splits]
    missing, images_differ, labels_differ, bad_lines, counts, empty = [], [], [], [], {}, {}
    for split in splits:
        src = _stems(source / "images" / split, ".png")
        imgs, labs = _stems(root / "images" / split, ".png"), _stems(root / "labels" / split, ".txt")
        missing += [f"{split}/{s}" for s in sorted((src ^ imgs) | (src ^ labs))]
        derived, expected, removed = Counter(), Counter(), Counter()
        empty_derived = empty_expected = 0
        for stem in sorted(src & imgs & labs):
            if _sha256(source / "images" / split / f"{stem}.png") != _sha256(root / "images" / split / f"{stem}.png"):
                images_differ.append(f"{split}/{stem}")
            src_text = (source / "labels" / split / f"{stem}.txt").read_text()
            text = (root / "labels" / split / f"{stem}.txt").read_text()
            if text != "".join(f"{l}\n" for l in filter_lines(src_text, n_kept)):
                labels_differ.append(f"{split}/{stem}")
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5 or not parts[0].isdigit() or int(parts[0]) >= n_kept:
                    bad_lines.append(f"{split}/{stem}: {line}")
                else:
                    derived[int(parts[0])] += 1
            src_ids = [int(line.split()[0]) for line in src_text.splitlines()]
            for c in src_ids:
                (expected if c < n_kept else removed)[c] += 1
            empty_expected += not any(c < n_kept for c in src_ids)
            empty_derived += text == ""
        counts[split] = {"derived": {n: derived[i] for i, n in enumerate(kept.names)},
                         "expected": {n: expected[i] for i, n in enumerate(kept.names)},
                         "removed": {n: removed[classes.class_id(n)] for n in classes.names[n_kept:]},
                         "passed": derived == expected}
        empty[split] = {"derived": empty_derived, "expected": empty_expected,
                        "passed": empty_derived == empty_expected}
    actual_yaml = yaml.safe_load((root / "data.yaml").read_text()) if (root / "data.yaml").is_file() else None
    checks["files_complete"] = {"passed": not missing and not stray, "detail": {"missing_or_extra": missing[:20],
                                                                                 "stray_split_dirs": stray}}
    checks["no_test_split"] = {"passed": "test" not in top["images"] + top["labels"]
                                         and "test" not in (actual_yaml or {}), "detail": top}
    checks["images_identical"] = {"passed": not images_differ, "detail": images_differ[:20]}
    checks["labels_derived"] = {"passed": not labels_differ, "detail": labels_differ[:20]}
    checks["class_ids_valid"] = {"passed": not bad_lines, "detail": bad_lines[:20]}
    checks["counts_match"] = {"passed": all(c["passed"] for c in counts.values()), "detail": counts}
    checks["empty_label_files"] = {"passed": all(e["passed"] for e in empty.values()), "detail": empty}
    checks["data_yaml_valid"] = {"passed": actual_yaml == data_yaml(final, kept.names, splits), "detail": actual_yaml}
    return checks

def summarize(checks: dict, fingerprint: dict, cfg: TwoClass) -> str:
    return (f"# Two-class dataset (excluded: {', '.join(cfg.exclude)}; splits: {', '.join(cfg.splits)})\n\n"
            + _block("Checks", json.dumps(checks, indent=2))
            + _block("Fingerprint", json.dumps(fingerprint, indent=2)))

def _write_report(report: Path, checks: dict, fingerprint: dict | None, cfg: TwoClass) -> None:
    report.mkdir(parents=True, exist_ok=True)
    (report / "checks.json").write_text(json.dumps(checks, indent=2))
    if fingerprint is not None:
        (report / "summary.md").write_text(summarize(checks, fingerprint, cfg))

def run_two_class(paths: Paths, classes: Classes, cfg: TwoClass) -> bool:
    source, out, report = paths.processed_dir, cfg.processed_dir, paths.artifacts_dir / "two_class"
    gate(source, paths.artifacts_dir / "verify")
    print("[PASS] source gate: Step 4 checks passed and the verified fingerprint matches")
    kept = kept_classes(classes, cfg.exclude)
    if out.exists():
        checks = run_checks(source, out, out, classes, kept, cfg.splits)
        recorded = (json.loads((report / "fingerprint.json").read_text())
                    if (report / "fingerprint.json").is_file() else None)
        fingerprint = dataset_fingerprint(out)
        checks["fingerprint_unchanged"] = {"passed": fingerprint == recorded, "detail": fingerprint["dataset"]}
    else:
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            raise RuntimeError(f"{tmp} is left over from a failed build; inspect and remove it first")
        derive(source, tmp, out, kept, cfg.splits)
        checks = run_checks(source, tmp, out, classes, kept, cfg.splits)
        fingerprint = dataset_fingerprint(tmp)
        if not all(c["passed"] for c in checks.values()):
            _write_report(report, checks, None, cfg)
            print(f"Checks failed; build left in {tmp} for inspection, nothing published.")
            return False
        tmp.rename(out)
        report.mkdir(parents=True, exist_ok=True)
        (report / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    _write_report(report, checks, fingerprint, cfg)
    for k, c in checks.items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {k}")
    print(f"Dataset: {out / 'data.yaml'}  Report: {report / 'summary.md'}")
    return all(c["passed"] for c in checks.values())

def gate_two_class(paths: Paths, cfg: TwoClass) -> tuple[dict, dict]:
    """(derived, source) fingerprints; raises unless the source gate and every two-class check pass unchanged."""
    source = gate(paths.processed_dir, paths.artifacts_dir / "verify")
    report = paths.artifacts_dir / "two_class"
    if not (report / "checks.json").is_file() or not (report / "fingerprint.json").is_file():
        raise RuntimeError("two-class dataset not built; run `python -m tvision two-class` first")
    if failed := [k for k, c in json.loads((report / "checks.json").read_text()).items() if not c["passed"]]:
        raise RuntimeError(f"two-class checks failed: {failed}")
    derived = dataset_fingerprint(cfg.processed_dir)
    if derived != json.loads((report / "fingerprint.json").read_text()):
        raise RuntimeError("two-class dataset differs from its recorded fingerprint; re-run `python -m tvision two-class`")
    return derived, source

def two_class() -> bool:
    return run_two_class(Paths.load(), Classes.load(), TwoClass.load())