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

    @classmethod
    def load(cls) -> "Paths":
        cfg = load_yaml("paths.yaml")
        return cls(**{k: _resolve(cfg[k]) for k in ("raw_dir", "processed_dir", "artifacts_dir")})

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