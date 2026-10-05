# TVision — Tumor Nucleus Detection

Research-use computer vision system for detecting tumor and non-tumor nuclei in H&E breast histopathology images using the NuCLS dataset and YOLO.

> Not a clinical diagnostic tool.

## Setup

```bash
uv sync
```

## Standalone inference

The detector is the validation-selected checkpoint named in `configs/inference.yaml`, which pins its SHA-256, the confidence threshold (0.24, chosen on validation), imgsz 640, max_det 1000, NMS IoU 0.7 and the device.

```bash
# JSON to stdout (one image: an object; several: a list)
uv run python -m tvision predict path/to/fov.png

# <stem>.json and an OpenCV-annotated <stem>.png per image; existing files are never overwritten
uv run python -m tvision predict a.png b.png --out results/ --annotate

# other options: --conf C (at least 0.01), --device {mps,cuda,cpu}
```

From Python:

```python
from tvision.config import Classes, Inference
from tvision.inference import Predictor, annotate

predictor = Predictor(Inference.load(), Classes.load())
result = predictor.predict("fov.png")    # also takes encoded bytes or a BGR uint8 array
print(result.model_dump_json(indent=2))
```

Each `DetectionResult` (`src/tvision/contract.py`) holds:
- the image size and the resize scale applied;
- the model info, including the checkpoint SHA-256;
- the threshold used;
- detections sorted by confidence, each with `class_id`, `class_name`, `confidence` and a pixel box `x1, y1, x2, y2` clipped to the image;
- per-class `counts`.

The classes are `tumor`, `non_tumor` and `ambiguous`; the headline classes are `tumor` and `non_tumor`.

`uv run python -m tvision predict --check` re-measures the deployed path against the Step 6 validation operating point, using validation images only.

### Limitations
- **Research use only.** This is not a clinical diagnostic tool.
- **Training data:** the model was trained only on NuCLS breast-cancer H&E crops at about 0.2 µm/px, with long sides of 256–758 px. Inputs are expected to be similar fields of view at that scale.
- **Input size:** each image is analyzed in one pass at imgsz 640. Images whose long side is outside 192–1024 px are refused rather than tiled, and the pixel scale cannot be checked from the file.
- **Held-out performance:** on the held-out test hospitals, headline mAP50-95 was 0.232. Performance varied strongly by hospital (0.12 for OL against 0.29–0.37 elsewhere), and non_tumor nuclei are often called tumor. Expect weaker results on unseen sites and stains.
- **Deployed operating point (validation):** headline P / R / F1 = 0.6054 / 0.5735 / 0.5890. This path differs slightly from the Step 6 validation pipeline (largest gap: tumor recall −0.0165); see `docs/PROJECT_STATUS.md`, Step 7.