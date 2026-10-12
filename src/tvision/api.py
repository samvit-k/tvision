"""Step 8: the FastAPI backend serving the Step 7 Predictor (PROJECT.md 14).

HTTP concerns only: request validation and error mapping. Model loading, inference and the detection contract live in
`tvision.inference` and `tvision.contract`; nothing here imports training, experiment or evaluation code. Responses
are JSON only and nothing is saved. Starlette spools uploads over 1 MB to an anonymous temporary file that is deleted
when the request ends.
"""

import io
import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import JpegImagePlugin, PngImagePlugin
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from tvision.config import Api, Classes, Inference
from tvision.contract import DetectionResult, InputLimits, ServiceInfo
from tvision.inference import InputError, Predictor

log = logging.getLogger(__name__)
DETECT_PATH = "/detect"
MAGIC = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}
# Pillow's format classes are used directly: Ultralytics replaces `PIL.Image.open` process-wide with a wrapper that
# tries to install pi-heif after any failed open (ultralytics/utils/patches.py).
HEADER_READERS = {"image/png": PngImagePlugin.PngImageFile, "image/jpeg": JpegImagePlugin.JpegImageFile}

class UploadSizeGuard:
    """Refuses `POST /detect` without a Content-Length (411) or with one above `max_body` (413) before the body is
    read. Content-Length covers the whole multipart request, so `max_body` is the file limit plus a fixed allowance;
    the file part's own size is checked authoritatively in `detect`."""

    def __init__(self, app, max_body: int):
        self.app = app
        self.max_body = max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == DETECT_PATH:
            length = Headers(scope=scope).get("content-length")
            error = None
            if length is None:
                error = 411, "Content-Length is required"
            elif not length.isdigit():
                error = 400, "Content-Length must be a non-negative integer"
            elif int(length) > self.max_body:
                error = 413, f"request body of {length} bytes exceeds the {self.max_body}-byte limit"
            if error:
                await JSONResponse({"detail": error[1]}, error[0])(scope, receive, send)
                return
        await self.app(scope, receive, send)

def default_predictor(device: str | None = None) -> Predictor:
    return Predictor(Inference.load(), Classes.load(), device)

def _check_header(data: bytes, content_type: str, min_long_side: int, max_long_side: int) -> None:
    """Refuses content that is not the declared format (415) or whose header dimensions are out of bounds (422),
    without decoding the pixels, so oversized images are never decompressed."""
    if not data.startswith(MAGIC[content_type]):
        raise HTTPException(415, f"file content is not {content_type}")
    try:
        with HEADER_READERS[content_type](io.BytesIO(data)) as img:
            w, h = img.size
    except (SyntaxError, OSError, ValueError) as e:
        raise HTTPException(422, f"the image header could not be read: {e}") from None
    if not min_long_side <= max(w, h) <= max_long_side:
        raise HTTPException(422, f"image is {w}x{h} px; its long side must be {min_long_side}-{max_long_side} px "
                                 "(the model was trained on NuCLS crops of 256-758 px at about 0.2 um/px)")

def create_app(predictor_factory: Callable[[], Predictor] = default_predictor, api_cfg: Api | None = None) -> FastAPI:
    """The app; the Predictor is built once at startup, so a missing or altered checkpoint stops the server."""
    api_cfg = api_cfg or Api.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        predictor = predictor_factory()
        cfg = predictor.cfg
        app.state.predictor = predictor
        app.state.inference_lock = threading.Lock()  # one Ultralytics model must not run in several threads at once
        app.state.service = ServiceInfo(model=predictor.info, limits=InputLimits(
            min_long_side=cfg.min_long_side, max_long_side=cfg.max_long_side, min_threshold=cfg.min_threshold,
            max_upload_bytes=api_cfg.max_upload_bytes, content_types=api_cfg.content_types))
        yield

    app = FastAPI(title="TVision", version="0.1.0", lifespan=lifespan,
                  description="Research-use tumor-nucleus detection in H&E images. Not a clinical diagnostic tool.")
    app.add_middleware(UploadSizeGuard, max_body=api_cfg.max_upload_bytes + api_cfg.multipart_overhead_bytes)
    app.add_middleware(CORSMiddleware, allow_origins=api_cfg.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], allow_credentials=False)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model", response_model=ServiceInfo)
    def model(request: Request) -> ServiceInfo:
        return request.app.state.service

    @app.post(DETECT_PATH, response_model=DetectionResult)
    def detect(request: Request, file: Annotated[UploadFile, File()],
               conf: Annotated[float | None, Form()] = None) -> DetectionResult:
        predictor: Predictor = request.app.state.predictor
        content_type = (file.content_type or "").split(";")[0].strip().lower()
        if content_type not in api_cfg.content_types:
            raise HTTPException(415, f"content type {file.content_type!r} is not one of {api_cfg.content_types}")
        data = file.file.read(api_cfg.max_upload_bytes + 1)
        if len(data) > api_cfg.max_upload_bytes:
            raise HTTPException(413, f"file exceeds the {api_cfg.max_upload_bytes}-byte limit")
        if not data:
            raise HTTPException(422, "the file is empty")
        _check_header(data, content_type, predictor.cfg.min_long_side, predictor.cfg.max_long_side)
        try:
            with request.app.state.inference_lock:
                return predictor.predict(data, conf)
        except InputError as e:
            raise HTTPException(422, str(e)) from None
        except Exception:
            log.exception("inference failed")
            raise HTTPException(500, "inference failed") from None

    return app

def serve(host: str | None = None, port: int | None = None, device: str | None = None) -> int:
    import uvicorn
    api_cfg = Api.load()
    uvicorn.run(create_app(partial(default_predictor, device), api_cfg), host=host or api_cfg.host,
                port=port or api_cfg.port, workers=1, lifespan="on")
    return 0