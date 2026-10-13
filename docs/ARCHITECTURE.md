# TVision architecture

How TVision gets from the raw NuCLS dataset to a detector served behind a local web UI, and the controls that keep each stage honest. The README covers setup and usage.

> Research use only. TVision is not a clinical diagnostic tool.

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Explore and validate the raw data | `python -m tvision explore` | dataset summary, checks, ground-truth overlays |
| 2. Leakage-safe splits | `python -m tvision splits` | `splits.csv` (train / val / test by hospital) |
| 3. Convert to YOLO labels | `python -m tvision convert` | cropped images, labels, `data.yaml`, per-annotation audit |
| 4. Verify the conversion independently | `python -m tvision verify` | checks, dataset fingerprint, overlays |
| 5. Train the baseline | `python -m tvision train` | baseline checkpoint and run record |
| 5A. Validation-only experiments and selection | `experiment`, `compare`, `select` | frozen final configuration and checkpoint |
| 6. One-shot held-out test evaluation | `python -m tvision evaluate` | test metrics, error analysis, galleries, access ledger |
| 7. Standalone inference | `python -m tvision predict` | `DetectionResult` JSON, optional annotated PNG |
| 8. HTTP API | `python -m tvision serve` | FastAPI on `127.0.0.1:8000` |
| 9. Web UI | `npm run dev` in `web/` | Nuxt page on `localhost:3000` |

Each stage reads its settings from `configs/*.yaml` and refuses to run when an earlier stage's checks have not passed. The raw data in `QC/` is never modified.

## Data

- **Source:** the NuCLS corrected single-rater dataset (CC0 1.0): 1,744 H&E fields of view (FOVs) from 124 TCGA breast-cancer slides at 18 hospitals, with 59,485 annotated nuclei at about 0.2 µm/px.
- **Classes:** `tumor` (NuCLS `tumor_any`); `non_tumor` (`sTIL`, `nonTIL_stromal`, `other_nucleus`); `ambiguous` (`AMBIGUOUS`). The mapping lives in `configs/classes.yaml`. Tumor and non-tumor are the headline classes; ambiguous is trained on but reported separately.
- **FOV crop:** about 20% of each image lies outside the annotated FOV rectangle, where nuclei are mostly unlabelled. Images are therefore cropped to the rectangle taken from the mask before training.

## Leakage-safe splits

- **Test:** NuCLS's official fold-1 test set: 4 hospitals, 22 slides, 263 FOVs. It is never used for training, tuning or model selection.
- **Validation:** whole hospitals held out from fold 1's training slides (AO, C8, GM, LL; 22 slides, 222 FOVs). They are chosen by exhaustive search for 15% of FOVs with a matching tumor share. The search is deterministic and needs no random seed.
- **Train:** the remaining 10 hospitals (80 slides, 1,259 FOVs).
- No slide, patient or hospital appears in more than one split, and this is checked automatically.

## Conversion and verification

- **Conversion:** each annotation is filtered in a fixed order:
  - point annotations are dropped;
  - boxes are checked against an edge-rounding rule;
  - exact duplicate boxes are removed;
  - boxes are clipped to the crop;
  - boxes left with less than 10% of their area, or under 2 px wide or tall, are dropped.

  The result: 58,828 boxes kept from 59,485 annotations (111 point annotations, 1 duplicate, 543 below visibility and 2 below size dropped). Every annotation's fate is recorded.
- **Verification** recomputes every label from the raw CSVs and masks with its own code; it never imports the converter. It checks:
  - that crops are pixel-identical;
  - every label line, to within 0.00052 px;
  - drop accounting and class IDs;
  - for leakage and duplicates.

  Overlays were reviewed by hand. Training refuses to start unless every check passed and the processed data's SHA-256 fingerprint matches.

## Training and model selection

- **Baseline:** Ultralytics YOLO26n from pretrained weights, imgsz 640, `max_det` 1000, flips-only augmentation, deterministic seed 0, with early stopping on validation.
- **Validation-only experiments:** the baseline stays frozen, and a SHA-256 manifest of its 38 files is re-checked before and after every command. Each experiment changes one declared factor against a matched control, and the code refuses undeclared differences. The candidates were:
  - conservative H&E-preserving appearance augmentation;
  - modest geometric augmentation;
  - imgsz 768;
  - a two-class variant with ambiguous nuclei excluded;
  - a larger YOLO26s.

  Runs use a headline fitness (mean tumor and non-tumor mAP50-95) and an optimizer pinned to the baseline's effective settings.
- **Selection rule, fixed before any run:** a finalist must gain at least max(0.01, control SD) in headline mAP50-95 across 3 seeds, be positive on every seed, and pass guards on tumor AP and mAP50.
  - Selected: **geometric augmentation** (scale 0.1, translate 0.05), headline val mAP50-95 0.2788 (SD 0.0096) against 0.2623 (SD 0.0085) for the control, over 3 seeds.
  - The checkpoint comes from the median seed and is pinned by SHA-256.

## Held-out test evaluation

The baseline and the selected model were each evaluated on the test split once. Their operating thresholds were fixed on validation first (baseline 0.29, final 0.24). An append-only ledger records every test pass with the checkpoint hash.

| Test (headline = mean of tumor and non-tumor) | Baseline | Final |
|---|---|---|
| Headline mAP50 | 0.476 | 0.489 |
| Headline mAP75 | 0.119 | 0.188 |
| Headline mAP50-95 | 0.199 | 0.232 |
| Tumor mAP50-95 | 0.203 | 0.192 |
| Non-tumor mAP50-95 | 0.194 | 0.273 |
| Tumor recall / F1 at the operating point | 0.616 / 0.484 | 0.681 / 0.504 |

- **Uncertainty:** a paired, slide-level bootstrap gives Δ headline mAP50-95 = +0.034, 95% CI [+0.020, +0.054]. It covers sampling within the four test hospitals only, not generalization to new hospitals.
- **Main failure mode:** non-tumor nuclei predicted as tumor, about a third of non-tumor nuclei.
- **Domain shift:** one test hospital (OL, 114 of 263 FOVs) reaches only 0.121 headline mAP50-95, against 0.29–0.37 at the other three.

## Inference contract

- **`Predictor`** (`src/tvision/inference.py`):
  - loads the checkpoint named in `configs/inference.yaml` and refuses it if its SHA-256, class names or head type differ;
  - analyzes each image in one pass at imgsz 640, with a square letterbox, single-label NMS (IoU 0.7) and `max_det` 1000;
  - refuses images whose long side is outside 192–1024 px.
- **`DetectionResult`** (`src/tvision/contract.py`) is a typed pydantic model shared by the CLI, the API and the UI. It holds:
  - image size and resize scale;
  - model info;
  - the threshold used;
  - detections sorted by confidence, each with class, confidence and a pixel box `x1, y1, x2, y2`;
  - per-class counts.

  Its validators enforce internal consistency.
- **Deployed operating point on validation:** headline precision / recall / F1 = 0.605 / 0.574 / 0.589 at confidence > 0.24. This standalone path differs slightly from the validator used for selection (largest gap: tumor recall −0.017); the gap was measured and accepted.
- **Visualization:** OpenCV drawing in `draw.py` and `annotate` only renders detections; it never computes them.

## API and web UI

- **FastAPI** (`src/tvision/api.py`):
  - `GET /health`;
  - `GET /model`: model info and input limits;
  - `POST /detect`: multipart PNG or JPEG, optional `conf`, returns `DetectionResult`.

  Uploads are validated in layers: the Content-Length guard, declared type, size, magic bytes, then the header dimensions, read before any pixels are decoded. Inference is serialized behind a lock, and the API module imports no training or evaluation code.
- **Nuxt UI** (`web/`): a client of the API only.
  - Its TypeScript types are generated from a committed OpenAPI snapshot, and a test fails if the snapshot drifts from the live schema.
  - It sends one request per image at the lowest allowed confidence and filters it with the slider. Thresholds nest, so this equals a fresh request at any slider value.
  - When a response reached `max_det`, the page says which weaker detections were cut off.

## Reproducibility controls

- All experiment-sensitive settings live in `configs/`. Runs record the resolved config, library versions, git commit, and dataset and weight hashes.
- Frozen artifacts are protected by hash manifests. The test split is reached only through the one-shot, ledgered evaluation.
- The test suite runs without the dataset or checkpoint: tests that need them skip.

## Limitations

- The model was trained on NuCLS breast-cancer FOVs at about 0.2 µm/px only. The pixel scale of an uploaded image cannot be checked.
- Performance drops sharply on unfamiliar hospitals and stains, and non-tumor nuclei are often called tumor.
- Confidence scores are not calibrated probabilities.
- Images are analyzed in one pass, not tiled, so whole-slide images are out of scope.