"""Step 7: standalone inference with the frozen, validation-selected detector.

Deployment-side only: nothing here imports the training, experiment or evaluation code. The checkpoint is refused
unless its SHA-256 equals configs/inference.yaml and its class names equal configs/classes.yaml. Images are analyzed
in one pass at the trained imgsz; images whose long side is outside the configured range are refused, since the model
has only seen NuCLS crops of 256-758 px at about 0.2 um/px.
"""

from tvision.ultralytics_env import check_device  # configures Ultralytics before it is imported

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tvision.config import Classes, Inference
from tvision.contract import Box, Detection, DetectionResult, ImageInfo, ModelInfo
from tvision.draw import draw_boxes

class InputError(ValueError):
    """The image or request cannot be analyzed; the message says why."""

class ModelError(RuntimeError):
    """The configured checkpoint is missing, altered, or not the model this project trained."""

def load_image(source: Path | str | bytes | np.ndarray, min_long_side: int, max_long_side: int) -> np.ndarray:
    """A BGR uint8 image from a file, encoded bytes (PNG, JPEG, ...) or a BGR array, within the long-side bounds.

    Encoded images are decoded as 3-channel colour: alpha is dropped and greyscale is expanded.
    """
    if isinstance(source, np.ndarray):
        img = source
    else:
        if isinstance(source, (str, Path)):
            if not Path(source).is_file():
                raise InputError(f"{source}: no such file")
            source = Path(source).read_bytes()
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR) if len(source) else None
        if img is None:
            raise InputError("the data could not be decoded as an image")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise InputError(f"expected an 8-bit 3-channel image, got dtype {img.dtype} and shape {img.shape}")
    h, w = img.shape[:2]
    if not min_long_side <= max(h, w) <= max_long_side:
        raise InputError(f"image is {w}x{h} px; its long side must be {min_long_side}-{max_long_side} px "
                         "(the model was trained on NuCLS crops of 256-758 px at about 0.2 um/px)")
    return img

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class Predictor:
    """Loads the configured checkpoint once and turns images into `DetectionResult`s.

    Inference deliberately uses a square imgsz letterbox (`rect=False`), matching the 640x640 training canvas.
    Ultralytics 8.4.171's `Model.predict` would otherwise default to `rect=True` (a minimal stride-32 rectangle).
    """

    def __init__(self, cfg: Inference, classes: Classes, device: str | None = None):
        cfg = cfg if device is None else replace(cfg, device=device)
        if not cfg.weights.is_file():
            raise ModelError(f"{cfg.weights} not found")
        if (sha := _sha256(cfg.weights)) != cfg.sha256:
            raise ModelError(f"{cfg.weights} SHA-256 {sha} != {cfg.sha256} in configs/inference.yaml")
        check_device(cfg.device)
        self.model = YOLO(cfg.weights, task="detect")
        names = [self.model.names[i] for i in sorted(self.model.names)]
        if names != classes.names:
            raise ModelError(f"checkpoint classes {names} != {classes.names} in configs/classes.yaml")
        if self.model.model.end2end:
            raise ModelError("checkpoint uses the NMS-free end-to-end head; this project validated NMS outputs")
        self.cfg = cfg
        self.info = ModelInfo(name=cfg.name, weights_sha256=sha, imgsz=cfg.imgsz, max_det=cfg.max_det, iou=cfg.iou,
                              classes=classes.names, headline_classes=classes.headline,
                              default_threshold=cfg.threshold)

    def load(self, source: Path | str | bytes | np.ndarray) -> np.ndarray:
        return load_image(source, self.cfg.min_long_side, self.cfg.max_long_side)

    def predict(self, source: Path | str | bytes | np.ndarray, conf: float | None = None) -> DetectionResult:
        """Detections with confidence above `conf` (default: the val-chosen threshold)."""
        conf = self.cfg.threshold if conf is None else conf
        if not self.cfg.min_threshold <= conf < 1:
            raise InputError(f"conf must be in [{self.cfg.min_threshold}, 1), got {conf}")
        img = self.load(source)
        r = self.model.predict(img, imgsz=self.cfg.imgsz, conf=conf, iou=self.cfg.iou, max_det=self.cfg.max_det,
                               device=self.cfg.device, rect=False, verbose=False)[0]
        return self.to_result(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                              r.boxes.cls.cpu().numpy(), r.orig_shape, conf)

    def to_result(self, xyxy: np.ndarray, confidence: np.ndarray, cls: np.ndarray, shape: tuple[int, int],
                  conf: float) -> DetectionResult:
        """Ultralytics' scaled boxes (already clipped by `scale_boxes`; clipped again here) as the contract."""
        h, w = shape
        names = self.info.classes
        xyxy = np.clip(np.asarray(xyxy, dtype=np.float64).reshape(-1, 4), 0, [w, h, w, h])
        order = np.argsort(-np.asarray(confidence), kind="stable")
        detections = [Detection(class_id=int(cls[i]), class_name=names[int(cls[i])], confidence=float(confidence[i]),
                                box=Box(x1=xyxy[i, 0], y1=xyxy[i, 1], x2=xyxy[i, 2], y2=xyxy[i, 3]))
                      for i in order]
        counts = {n: sum(d.class_name == n for d in detections) for n in names}
        return DetectionResult(image=ImageInfo(width=w, height=h, scale=self.cfg.imgsz / max(h, w)), model=self.info,
                               threshold=conf, detections=detections, counts=counts)

def annotate(image_bgr: np.ndarray, result: DetectionResult, labels: bool = True) -> np.ndarray:
    """A copy of the image with the result's boxes drawn; OpenCV only draws what the model returned."""
    return draw_boxes(image_bgr, [{**d.box.model_dump(), "class_name": d.class_name, "confidence": d.confidence}
                                  for d in result.detections], thickness=1, show_labels=labels)

def predict_images(images: list[Path], conf: float | None = None, device: str | None = None, out: Path | None = None,
                   annotate_images: bool = False) -> int:
    """CLI: JSON to stdout, or `<stem>.json` (and `<stem>.png`) under `out`. Every image is validated before any
    model call, and existing outputs are never overwritten. With `annotate_images`, an image's PNG is written before
    its JSON, so a failed PNG write leaves neither."""
    try:
        predictor = Predictor(Inference.load(), Classes.load(), device)
        stems = [p.stem for p in images]
        if out is not None:
            if len(set(stems)) != len(stems):
                raise InputError("input images must have distinct file names")
            targets = [out / f"{s}{ext}" for s in stems for ext in (".json", ".png")[:1 + annotate_images]]
            if existing := [str(t) for t in targets if t.exists()]:
                raise InputError(f"refusing to overwrite {existing}")
        loaded = []
        for p in images:
            try:
                loaded.append(predictor.load(p))
            except InputError as e:
                raise InputError(f"{p}: {e}") from None
        results = [predictor.predict(img, conf) for img in loaded]
    except (InputError, ModelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if out is None:
        payload = results[0].model_dump(mode="json") if len(results) == 1 else [r.model_dump(mode="json")
                                                                                 for r in results]
        print(json.dumps(payload, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for stem, img, r in zip(stems, loaded, results):
        if annotate_images:
            target = out / f"{stem}.png"
            if not cv2.imwrite(str(target), annotate(img, r)):
                target.unlink(missing_ok=True)
                raise RuntimeError(f"failed to write annotated image to {target}")
        (out / f"{stem}.json").write_text(r.model_dump_json(indent=2))
    print(f"Wrote {len(results)} result(s) to {out}")
    return 0