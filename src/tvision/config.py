"""Loading of project YAML configs. Relative paths resolve against the project root."""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)

def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    runs_dir: Path

    @classmethod
    def load(cls) -> "Paths":
        cfg = load_yaml("paths.yaml")
        return cls(**{k: _resolve(cfg[k]) for k in ("raw_dir", "processed_dir", "artifacts_dir", "runs_dir")})

@dataclass(frozen=True)
class Classes:
    names: list[str]
    headline: list[str]
    super_to_class: dict[str, str]
    drop_types: list[str]

    @classmethod
    def load(cls) -> "Classes":
        cfg = load_yaml("classes.yaml")
        classes = cls(**cfg)
        unknown = set(classes.super_to_class.values()) - set(classes.names)
        if unknown:
            raise ValueError(f"super_to_class maps to undefined classes: {sorted(unknown)}")
        if not set(classes.headline) <= set(classes.names):
            raise ValueError("headline classes must be a subset of names")
        return classes

    def class_id(self, name: str) -> int:
        return self.names.index(name)

@dataclass(frozen=True)
class Splits:
    test_fold: int
    val_fraction: float

    @classmethod
    def load(cls) -> "Splits":
        splits = cls(**load_yaml("splits.yaml"))
        if splits.test_fold not in (1, 2, 3, 4, 5):
            raise ValueError(f"test_fold must be an official fold 1-5, got {splits.test_fold}")
        if not 0 < splits.val_fraction < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {splits.val_fraction}")
        return splits

@dataclass(frozen=True)
class Convert:
    min_visibility: float
    min_size_px: float

    def __post_init__(self):
        if not 0 <= self.min_visibility <= 1:
            raise ValueError(f"min_visibility must be in [0, 1], got {self.min_visibility}")
        if not self.min_size_px > 0:
            raise ValueError(f"min_size_px must be > 0, got {self.min_size_px}")

    @classmethod
    def load(cls) -> "Convert":
        return cls(**load_yaml("convert.yaml"))

AUGMENTATION_KEYS = ("fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                     "perspective", "mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste", "bgr")
UNIT_INTERVAL_KEYS = ("fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "translate", "mosaic", "mixup", "cutmix",
                      "copy_paste", "bgr")
DEVICES = ("mps", "cuda", "cpu")

@dataclass(frozen=True)
class Train:
    model: str
    imgsz: int
    epochs: int
    patience: int
    batch: int
    device: str
    seed: int
    deterministic: bool
    cache: bool | str
    max_det: int
    name: str
    augmentation: dict[str, float]
    fraction: float = 1.0

    def __post_init__(self):
        aug = self.augmentation
        if set(aug) != set(AUGMENTATION_KEYS):
            raise ValueError(f"augmentation must set exactly {list(AUGMENTATION_KEYS)}; "
                             f"missing {sorted(set(AUGMENTATION_KEYS) - set(aug))}, "
                             f"unknown {sorted(set(aug) - set(AUGMENTATION_KEYS))}")
        bad = [k for k, v in aug.items() if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0
               or (k in UNIT_INTERVAL_KEYS and v > 1)]
        if bad:
            raise ValueError(f"augmentation values out of range: {bad}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.cache not in (False, "ram"):
            raise ValueError(f"cache must be false or 'ram' ('disk' writes into the dataset), got {self.cache!r}")
        if self.imgsz <= 0 or self.imgsz % 32:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        for k in ("epochs", "patience", "batch", "max_det"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1, got {getattr(self, k)}")
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    @classmethod
    def from_dict(cls, cfg: dict, smoke: bool = False) -> "Train":
        cfg = dict(cfg)
        overrides = cfg.pop("smoke", {}) or {}
        fields = set(cls.__dataclass_fields__)
        for where, keys in (("train.yaml", set(cfg)), ("smoke", set(overrides))):
            if unknown := keys - fields:
                raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
        if "augmentation" in overrides:
            raise ValueError("smoke must not override augmentation")
        return cls(**(cfg | overrides if smoke else cfg))

    @classmethod
    def load(cls, smoke: bool = False) -> "Train":
        return cls.from_dict(load_yaml("train.yaml"), smoke)

    def ultralytics_args(self) -> dict:
        """Keyword arguments for `YOLO.train`, excluding data, project and name."""
        args = {k: getattr(self, k) for k in ("imgsz", "epochs", "patience", "batch", "device", "seed",
                                              "deterministic", "cache", "max_det", "fraction")}
        return args | self.augmentation | {"augmentations": []}