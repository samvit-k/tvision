"""Project-local Ultralytics configuration shared by training and inference. Import before `ultralytics`."""

import os

from tvision.config import PROJECT_ROOT

if "YOLO_CONFIG_DIR" not in os.environ:
    (PROJECT_ROOT / ".ultralytics").mkdir(exist_ok=True)  # Ultralytics falls back to /tmp if the parent is missing
    os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".ultralytics")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch
from ultralytics import settings

settings.update({"sync": False, "clearml": False, "comet": False, "dvc": False, "mlflow": False,
                 "raytune": False, "tensorboard": False, "wandb": False})

def check_device(device: str) -> None:
    available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
    if not available[device]:
        raise RuntimeError(f"device {device!r} is not available; refusing to fall back to another device")