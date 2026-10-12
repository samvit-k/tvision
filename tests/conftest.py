"""Synthetic NuCLS-like raw dataset shared by the conversion and verification tests."""

import tvision.ultralytics_env  # noqa: F401  # configure Ultralytics (no autoinstall) before any test imports it

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tvision import nucls_data as nd

W, H = 100, 80

def _box(x0, y0, x1, y1, **kw) -> dict:
    return {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, **kw}

STEMS = {"TCGA-AA-0001-DX1_id-a": "train", "TCGA-BB-0002-DX1_id-b": "val", "TCGA-CC-0003-DX1_id-c": "test"}

def _raw(root: Path) -> tuple[Path, Path]:
    raw, out = root / "raw", root / "out"
    for d in ("rgb", "mask", "csv", "visualization"):
        (raw / d).mkdir(parents=True)
    out.mkdir()
    rng = np.random.default_rng(0)
    for i, stem in enumerate(STEMS):
        cv2.imwrite(str(raw / "rgb" / f"{stem}.png"), rng.integers(0, 255, (H, W, 3), np.uint8))
        cv2.imwrite(str(raw / "visualization" / f"{stem}.png"), np.zeros((H, W, 3), np.uint8))
        mask = np.zeros((H, W, 3), np.uint8)
        mask[10:70, 10:90, 2] = nd.FOV_CODE
        cv2.imwrite(str(raw / "mask" / f"{stem}.png"), mask)
        rows = [{"raw_classification": "tumor", "main_classification": "tumor_nonMitotic",
                 "super_classification": "tumor_any", "type": "rectangle", **_box(20, 20, 40, 40 + i)},
                {"raw_classification": "tumor", "main_classification": "tumor_nonMitotic",
                 "super_classification": "tumor_any", "type": "polyline", **_box(20, 20, 40, 40 + i)},
                {"raw_classification": "lymphocyte", "main_classification": "lymphocyte",
                 "super_classification": "sTIL", "type": "point", **_box(5, 5, 5, 5)},
                {"raw_classification": "lymphocyte", "main_classification": "lymphocyte",
                 "super_classification": "sTIL", "type": "polyline", **_box(0, 0, 8, 8)}]
        pd.DataFrame(rows).to_csv(raw / "csv" / f"{stem}.csv")
    pd.DataFrame({"stem": list(STEMS), "split": list(STEMS.values())}).to_csv(out / "splits.csv", index=False)
    return raw, out