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

## Serving the API

```bash
uv run python -m tvision serve                 # http://127.0.0.1:8000; --host, --port, --device override
```

The server loads the same Predictor once at startup and refuses to start if the checkpoint, its classes or the device don't match `configs/inference.yaml`. Server settings (bind address, CORS origin, accepted types, upload limit) are in `configs/api.yaml`.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /model` | `ServiceInfo`: the model info and the input limits (long side 192–1024 px, `conf` ≥ 0.01, 8 MiB, PNG/JPEG) |
| `POST /detect` | `DetectionResult`, the same JSON as `predict`, for a multipart `file` and an optional form field `conf` |

```bash
curl -F "file=@fov.png;type=image/png" -F conf=0.3 http://127.0.0.1:8000/detect
```

- **Errors:** 411 (no Content-Length), 413 (too large), 415 (not PNG/JPEG, or content that doesn't match the declared type), 422 (unreadable image, long side out of range, `conf` out of range), 500 (inference failed). Each returns a JSON `detail`.
- **Confidence filtering:** a response is capped at `max_det` = 1000 detections. Filtering a low-threshold response client-side matches a request at the higher threshold only if the response was not capped. At `conf` 0.01, one of the 222 validation images reaches the cap, so treat a response with exactly 1000 detections as possibly truncated.
- **Scope:** the API is for local research use. It has no authentication and binds to localhost by default.

## Web UI

A single Nuxt page for uploading an H&E image (PNG or JPEG) and inspecting the detections. It is a client of the API above: it has no model code and only calls `/health`, `/model` and `/detect`.

**Prerequisites:** Node 24 LTS (`web/.nvmrc`) and the API running (`uv run python -m tvision serve`).

```bash
cd web
npm ci                 # install the pinned dependencies
npm run dev            # http://localhost:3000
```

- Open **http://localhost:3000**. The API's CORS setting (`configs/api.yaml`) allows only that origin, so `127.0.0.1:3000` is refused.
- The API address defaults to `http://127.0.0.1:8000`. To use another one, set `NUXT_PUBLIC_API_BASE` (for example `NUXT_PUBLIC_API_BASE=http://127.0.0.1:8001 npm run dev`) and add the page's origin to `cors_origins`.
- **Using the page:** choose an image and press **Detect nuclei**. The page sends one request at the lowest allowed confidence (0.01), and the slider filters that response in the browser, starting at the validation-chosen 0.24. If a response reached the 1,000-detection cap, the page says which weaker detections were cut off.
- **Research use only:** the image is sent to the local API for analysis and is not stored.

| Command (in `web/`) | Purpose |
|---|---|
| `npm run typecheck` | Type check (`nuxt typecheck`). Plain `vue-tsc --noEmit` checks no files here |
| `npm test` | Vitest unit tests |
| `npm run generate` | Static build in `.output/public` |
| `npm run gen:openapi` then `npm run gen:types` | Refresh `openapi.json` and `app/types/api.ts` after an API change; `tests/test_api.py` fails while the snapshot is stale |