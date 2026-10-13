import asyncio
import contextlib
import json
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from ultralytics.engine.results import Results

from tvision import api
from tvision import inference as inf
from tvision.config import PROJECT_ROOT, Api, Classes, Inference, load_yaml
from tvision.contract import DetectionResult, InputLimits, ServiceInfo

from test_inference import CLASSES, REAL, VAL_IMAGES, FakeYOLO, _cfg, _encode, _rgb, needs_checkpoint

API = Api.load()
MAX = API.max_upload_bytes

class BoxYOLO(FakeYOLO):
    def predict(self, img, **kwargs):
        boxes = torch.tensor([[10, 10, 40, 40, 0.9, 0], [50, 60, 90, 100, 0.5, 1]], dtype=torch.float32)
        boxes = boxes[boxes[:, 4] > kwargs["conf"]]
        return [Results(img, path="", names=self.names, boxes=boxes)]

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "YOLO", BoxYOLO)
    cfg = _cfg(tmp_path)
    with TestClient(api.create_app(lambda: inf.Predictor(cfg, CLASSES), API)) as c:
        yield c

def _post(client, data: bytes, content_type: str = "image/png", conf=None, filename: str = "fov.png", **kw):
    form = None if conf is None else {"conf": str(conf)}
    return client.post("/detect", files={"file": (filename, data, content_type)}, data=form, **kw)

def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

def _padded_png(size: int) -> bytes:
    """A valid PNG of exactly `size` bytes, padded with a private ancillary chunk that decoders skip."""
    png = _encode(_rgb())
    pad = size - len(png) - 12
    assert pad >= 0
    return png[:-12] + _png_chunk(b"tvPd", bytes(pad)) + png[-12:]

def _with_dimensions(png: bytes, w: int, h: int) -> bytes:
    """The PNG with its IHDR width and height rewritten (pixel data untouched), as in a decompression bomb."""
    assert png[12:16] == b"IHDR"
    ihdr = struct.pack(">II", w, h) + png[24:29]
    return png[:8] + _png_chunk(b"IHDR", ihdr) + png[33:]

# Config

def test_api_config_loads_and_validates():
    cfg = load_yaml("api.yaml")
    assert API == Api(**cfg)
    with pytest.raises(ValueError, match="exactly"):
        Api.from_dict(cfg | {"extra": 1})
    with pytest.raises(ValueError, match="exactly"):
        Api.from_dict({k: v for k, v in cfg.items() if k != "port"})
    for bad in ({"port": 0}, {"port": True}, {"host": ""}, {"cors_origins": "http://x"},
                {"content_types": ["image/tiff"]}, {"content_types": []}, {"content_types": ["image/png"] * 2},
                {"max_upload_bytes": 0}, {"multipart_overhead_bytes": 100}):
        with pytest.raises(ValueError):
            Api(**cfg | bad)

def test_input_limits_reject_inverted_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        InputLimits(min_long_side=500, max_long_side=400, min_threshold=0.01, max_upload_bytes=1,
                    content_types=["image/png"])

# Endpoints

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_model_reports_predictor_info_and_configured_limits(client):
    info = ServiceInfo.model_validate(client.get("/model").json())
    assert info.model == client.app.state.predictor.info
    cfg = load_yaml("inference.yaml")
    assert info.limits == InputLimits(min_long_side=cfg["min_long_side"], max_long_side=cfg["max_long_side"],
                                      min_threshold=cfg["min_threshold"], max_upload_bytes=MAX,
                                      content_types=load_yaml("api.yaml")["content_types"])

@pytest.mark.parametrize("ext, content_type", [(".png", "image/png"), (".jpg", "image/jpeg")])
def test_detect_returns_the_contract(client, ext, content_type):
    r = _post(client, _encode(_rgb(), ext), content_type)
    assert r.status_code == 200
    result = DetectionResult.model_validate(r.json())
    assert result.threshold == REAL.threshold and result.image.width == 300 and result.image.height == 200
    assert [d.class_name for d in result.detections] == ["tumor", "non_tumor"]
    assert result.counts == {"tumor": 1, "non_tumor": 1, "ambiguous": 0}

def test_detect_passes_conf_through(client):
    result = DetectionResult.model_validate(_post(client, _encode(_rgb()), conf=0.6).json())
    assert result.threshold == 0.6 and [d.confidence for d in result.detections] == pytest.approx([0.9])

def test_detect_accepts_content_type_parameters(client):
    assert _post(client, _encode(_rgb()), "image/png; charset=binary").status_code == 200

@pytest.mark.parametrize("img", [np.zeros((200, 300), np.uint8), np.zeros((200, 300, 4), np.uint8)],
                         ids=["grey", "rgba"])
def test_detect_decodes_grey_and_rgba_to_three_channels(client, monkeypatch, img):
    seen = []
    model = client.app.state.predictor.model
    monkeypatch.setattr(model, "predict", lambda im, **k: seen.append(im.shape) or BoxYOLO.predict(model, im, **k))
    assert _post(client, _encode(img)).status_code == 200
    assert seen == [(200, 300, 3)]

# Refusals

def _refused(r, status: int, match: str) -> None:
    assert r.status_code == status, r.text
    assert match in json.dumps(r.json()["detail"])

def test_missing_file_is_422(client):
    _refused(client.post("/detect", data={"conf": "0.5"}), 422, "file")

@pytest.mark.parametrize("content_type", ["image/gif", "image/tiff", "text/plain", ""])
def test_disallowed_content_type_is_415(client, content_type):
    _refused(_post(client, _encode(_rgb()), content_type), 415, "content type")

def test_content_that_contradicts_the_declared_type_is_415(client):
    _refused(_post(client, _encode(_rgb()), "image/jpeg"), 415, "not image/jpeg")
    _refused(_post(client, _encode(_rgb(), ".tiff"), "image/png"), 415, "not image/png")

def test_upload_of_exactly_the_limit_is_accepted_despite_multipart_overhead(client):
    data = _padded_png(MAX)
    assert len(data) == MAX
    r = _post(client, data, conf=0.3, filename="f" * 200 + ".png")
    assert r.status_code == 200, r.text
    assert int(r.request.headers["content-length"]) > MAX  # the overhead is real, and the guard allowed it

def test_upload_one_byte_over_the_limit_is_413_from_the_file_read(client):
    r = _post(client, _padded_png(MAX + 1))
    assert int(r.request.headers["content-length"]) <= MAX + API.multipart_overhead_bytes
    _refused(r, 413, "file exceeds")

def test_request_over_the_guard_limit_is_413_before_reading(client, monkeypatch):
    monkeypatch.setattr(client.app.state.predictor, "predict", lambda *a, **k: pytest.fail("must not run"))
    _refused(_post(client, bytes(MAX + API.multipart_overhead_bytes + 1)), 413, "request body")

def test_missing_content_length_is_411(client):
    r = client.post("/detect", content=iter([b"--x\r\n"]), headers={"content-type": "multipart/form-data; boundary=x"})
    assert "content-length" not in r.request.headers
    _refused(r, 411, "Content-Length")

@pytest.mark.parametrize("length, status", [(None, 411), ("abc", 400), ("-1", 400), ("1000", None), ("1001", 413)])
def test_size_guard_directly(length, status):
    calls, sent = [], []

    async def app(scope, receive, send):
        calls.append(scope["path"])

    async def send(message):
        sent.append(message)

    headers = [] if length is None else [(b"content-length", length.encode())]
    guard = api.UploadSizeGuard(app, max_body=1000)
    for path, method in (("/detect", "POST"), ("/health", "GET")):
        asyncio.run(guard({"type": "http", "method": method, "path": path, "headers": headers}, None, send))
    if status is None:
        assert calls == ["/detect", "/health"] and not sent
    else:
        assert calls == ["/health"] and sent[0]["status"] == status

def test_empty_file_is_422(client):
    _refused(_post(client, b""), 422, "empty")

def test_truncated_image_is_422(client):
    r = _post(client, _encode(_rgb())[:60])
    assert r.status_code == 422

def test_check_header_refuses_a_corrupt_header_without_decoding():
    png = bytearray(_encode(_rgb()))
    assert png[12:16] == b"IHDR"
    png[29] ^= 0xFF  # corrupt the IHDR CRC, which Pillow verifies while reading the header
    with pytest.raises(HTTPException) as e:
        api._check_header(bytes(png), "image/png", 192, 1024)
    assert e.value.status_code == 422 and "header could not be read" in e.value.detail
    api._check_header(_encode(_rgb()), "image/png", 192, 1024)  # the intact PNG passes

@pytest.mark.parametrize("h, w", [(100, 150), (1100, 600)])
def test_out_of_bounds_dimensions_are_422(client, h, w):
    _refused(_post(client, _encode(_rgb(h, w))), 422, "long side must be")

@pytest.mark.parametrize("w, h", [(4000, 4000), (100_000, 100_000)])
def test_oversized_header_is_refused_without_decoding(client, monkeypatch, w, h):
    monkeypatch.setattr(inf.cv2, "imdecode", lambda *a: pytest.fail("must not decode"))
    assert _post(client, _with_dimensions(_encode(_rgb()), w, h)).status_code == 422

@pytest.mark.parametrize("conf", [0.005, 1.0, "nan", "abc"])
def test_conf_out_of_range_is_422(client, conf):
    assert _post(client, _encode(_rgb()), conf=conf).status_code == 422

def test_unexpected_inference_error_is_500_without_traceback(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal state")

    monkeypatch.setattr(client.app.state.predictor.model, "predict", boom)
    r = _post(client, _encode(_rgb()))
    assert r.status_code == 500 and r.json() == {"detail": "inference failed"}

# CORS

def test_cors_allows_only_the_configured_origin(client):
    origin = API.cors_origins[0]
    assert client.get("/health", headers={"origin": origin}).headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in client.get("/health", headers={"origin": "http://evil.test"}).headers
    pre = client.options("/detect", headers={"origin": origin, "access-control-request-method": "POST"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == origin

def test_guard_refusals_carry_cors_headers(client):
    origin = API.cors_origins[0]
    r = _post(client, bytes(MAX + API.multipart_overhead_bytes + 1), headers={"origin": origin})
    assert r.status_code == 413 and r.headers["access-control-allow-origin"] == origin

# Startup and concurrency

def test_startup_fails_on_altered_weights(tmp_path):
    app = api.create_app(lambda: inf.Predictor(replace(_cfg(tmp_path), sha256="0" * 64), CLASSES), API)
    with pytest.raises(inf.ModelError, match="SHA-256"):
        with TestClient(app):
            pass

def test_startup_fails_on_unavailable_device(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    app = api.create_app(lambda: inf.Predictor(_cfg(tmp_path, device="mps"), CLASSES), API)
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        with TestClient(app):
            pass

def _max_concurrent_predictions(client, monkeypatch, n: int = 4) -> int:
    model, state, guard = client.app.state.predictor.model, {"active": 0, "max": 0}, threading.Lock()

    def slow(img, **kwargs):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.2)
        with guard:
            state["active"] -= 1
        return BoxYOLO.predict(model, img, **kwargs)

    monkeypatch.setattr(model, "predict", slow)
    data, codes = _encode(_rgb()), []
    threads = [threading.Thread(target=lambda: codes.append(_post(client, data).status_code)) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200] * n
    return state["max"]

def test_inference_lock_serializes_concurrent_requests(client, monkeypatch):
    assert _max_concurrent_predictions(client, monkeypatch) == 1

def test_requests_do_overlap_without_the_lock(client, monkeypatch):
    client.app.state.inference_lock = contextlib.nullcontext()
    assert _max_concurrent_predictions(client, monkeypatch) > 1

# Deployment boundary

def test_tests_run_with_ultralytics_autoinstall_off():
    from ultralytics.utils import AUTOINSTALL
    assert AUTOINSTALL is False

def test_web_openapi_snapshot_matches_the_live_schema():
    """The web UI's types are generated from web/openapi.json, so it must equal the served contract."""
    snapshot = PROJECT_ROOT / "web" / "openapi.json"
    live = json.loads(json.dumps(api.create_app(api_cfg=API).openapi()))
    assert json.loads(snapshot.read_text()) == live, "regenerate with `npm run gen:openapi` in web/"

def test_api_imports_no_training_or_evaluation_code():
    out = subprocess.run([sys.executable, "-c", "import sys, tvision.api; "
                          "print(sorted(m for m in sys.modules if m.startswith('tvision')))"],
                         capture_output=True, text=True, check=True).stdout
    assert eval(out) == ["tvision", "tvision.api", "tvision.config", "tvision.contract", "tvision.draw",
                         "tvision.inference", "tvision.ultralytics_env"]

# Real checkpoint (val images only)

@needs_checkpoint
@pytest.mark.skipif(len(VAL_IMAGES) < 3, reason="no processed val images")
def test_real_detect_equals_predictor_on_val_images():
    predictor = inf.Predictor(Inference.load(), Classes.load(), device="cpu")
    with TestClient(api.create_app(partial(api.default_predictor, "cpu"), API)) as client:
        for path in VAL_IMAGES[:3]:
            assert "/images/val/" in path.as_posix()
            data = path.read_bytes()
            r = _post(client, data, filename=path.name)
            assert r.status_code == 200
            expected = predictor.predict(data).model_dump(mode="json")
            assert r.json() == expected and expected["detections"]