"""Read-only access to the NuCLS corrected single-rater QC dataset.

Layout: <raw>/{rgb,mask,visualization}/<stem>.png, <raw>/csv/<stem>.csv,
<raw>/train_test_splits/fold_<k>_{train,test}.csv. The stem prefix before "_id"
is the slide name; the TCGA site code (TCGA-<site>-...) identifies the hospital.
Mask red channel holds class codes: 0 = outside the annotated FOV, 253 = FOV background,
other values = nuclei (which may extend past the FOV rectangle).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tvision.config import Classes

SUBDIRS = {"rgb": ".png", "mask": ".png", "visualization": ".png", "csv": ".csv"}
ANNOTATION_COLUMNS = [
    "raw_classification", "main_classification", "super_classification",
    "type", "xmin", "ymin", "xmax", "ymax",
]
FOV_CODE = 253

def slide_of(stem: str) -> str:
    return stem.split("_id")[0]

def hospital_of(slide: str) -> str:
    return slide.split("-")[1]

def patient_of(slide: str) -> str:
    """TCGA participant barcode: project-TSS-participant."""
    return "-".join(slide.split("-")[:3])

def _stems(raw_dir: Path, subdir: str) -> set[str]:
    ext = SUBDIRS[subdir]
    return {p.name[: -len(ext)] for p in (raw_dir / subdir).glob(f"*{ext}")}

def check_file_pairing(raw_dir: Path) -> dict[str, list[str]]:
    """Per subdir, the stems missing relative to the union of all subdirs."""
    stems = {d: _stems(raw_dir, d) for d in SUBDIRS}
    union = set().union(*stems.values())
    return {d: sorted(union - s) for d, s in stems.items() if union - s}

def image_size(path: Path) -> tuple[int, int]:
    """(height, width) of an image."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Cannot read image: {path}")
    return img.shape[:2]

def mask_codes(mask_bgr: np.ndarray) -> np.ndarray:
    """Class-code plane of a mask loaded by OpenCV: codes live in the red channel, index 2 in BGR order."""
    if mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR mask, got shape {mask_bgr.shape}")
    return mask_bgr[..., 2]

def fov_rect(mask_codes: np.ndarray) -> tuple[int, int, int, int] | None:
    """Annotated FOV rectangle (x0, y0, x1, y1), exclusive end, from the extent of FOV_CODE pixels."""
    ys, xs = np.nonzero(mask_codes == FOV_CODE)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def fov_table(raw_dir: Path) -> pd.DataFrame:
    """One row per FOV from filenames only: stem, slide, patient, hospital."""
    stems = sorted(_stems(raw_dir, "rgb"))
    slides = [slide_of(s) for s in stems]
    return pd.DataFrame({"stem": stems, "slide": slides, "patient": [patient_of(s) for s in slides],
                         "hospital": [hospital_of(s) for s in slides]})

def fov_index(raw_dir: Path) -> pd.DataFrame:
    """One row per FOV: stem, slide, hospital, rgb and mask height/width, FOV rectangle."""
    rows = []
    for stem in sorted(_stems(raw_dir, "rgb")):
        slide = slide_of(stem)
        h, w = image_size(raw_dir / "rgb" / f"{stem}.png")
        mask = cv2.imread(str(raw_dir / "mask" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if mask is None:
            raise OSError(f"Cannot read mask: {stem}")
        rect = fov_rect(mask[..., 2]) or (None,) * 4
        rows.append({"stem": stem, "slide": slide, "hospital": hospital_of(slide),
                     "height": h, "width": w, "mask_height": mask.shape[0], "mask_width": mask.shape[1],
                     **dict(zip(("fov_x0", "fov_y0", "fov_x1", "fov_y1"), rect))})
    return pd.DataFrame(rows)

def load_annotations(raw_dir: Path, stems: list[str] | None = None) -> pd.DataFrame:
    """All nucleus annotations with a `stem` column; `ann_id` is the CSV's own row index."""
    stems = stems if stems is not None else sorted(_stems(raw_dir, "csv"))
    frames = []
    for stem in stems:
        df = pd.read_csv(raw_dir / "csv" / f"{stem}.csv", index_col=0)
        df = df[ANNOTATION_COLUMNS].rename_axis("ann_id").reset_index()
        df.insert(0, "stem", stem)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def map_classes(ann: pd.DataFrame, classes: Classes) -> pd.DataFrame:
    """Add `project_class` and `class_id`; raise if any super class is unmapped."""
    unmapped = set(ann["super_classification"]) - set(classes.super_to_class)
    if unmapped:
        raise ValueError(f"Unmapped NuCLS super classes: {sorted(unmapped)}")
    out = ann.copy()
    out["project_class"] = out["super_classification"].map(classes.super_to_class)
    out["class_id"] = out["project_class"].map(classes.class_id)
    return out

def load_split_file(raw_dir: Path, fold: int | str, part: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "train_test_splits" / f"fold_{fold}_{part}.csv", index_col=0)