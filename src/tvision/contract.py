"""The detection contract shared by inference, visualization, the API and the frontend (PROJECT.md 12).

Boxes are in original-image pixels (x right, y down), clipped to the image. Every detection has confidence above the
result's threshold. `counts` is derived from `detections` for the UI's class counts and is validated against them.
"""

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class Box(_Frozen):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float
    y2: float

    @model_validator(mode="after")
    def _positive_extent(self) -> "Box":
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError(f"box must have x2 > x1 and y2 > y1, got {self.x1, self.y1, self.x2, self.y2}")
        return self

class Detection(_Frozen):
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(gt=0, le=1)
    box: Box

class ImageInfo(_Frozen):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    scale: float = Field(gt=0, description="Model input size / image long side: the resize applied before inference")

class ModelInfo(_Frozen):
    name: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    imgsz: int = Field(gt=0)
    max_det: int = Field(gt=0)
    iou: float = Field(gt=0, lt=1)
    classes: list[str] = Field(min_length=1)
    headline_classes: list[str]
    default_threshold: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _headline_subset(self) -> "ModelInfo":
        if len(set(self.classes)) != len(self.classes) or not set(self.headline_classes) <= set(self.classes):
            raise ValueError("classes must be unique and headline_classes a subset of them")
        return self

class DetectionResult(_Frozen):
    image: ImageInfo
    model: ModelInfo
    threshold: float = Field(gt=0, lt=1)
    detections: list[Detection]
    counts: dict[str, int]

    @model_validator(mode="after")
    def _consistent(self) -> "DetectionResult":
        names = self.model.classes
        for d in self.detections:
            if not d.class_id < len(names) or names[d.class_id] != d.class_name:
                raise ValueError(f"class_id {d.class_id} does not name {d.class_name!r} in {names}")
            if not d.confidence > self.threshold:
                raise ValueError(f"confidence {d.confidence} is not above the threshold {self.threshold}")
            if d.box.x2 > self.image.width or d.box.y2 > self.image.height:
                raise ValueError(f"box {d.box} lies outside the {self.image.width}x{self.image.height} image")
        confidences = [d.confidence for d in self.detections]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError("detections must be sorted by confidence, highest first")
        expected = Counter(d.class_name for d in self.detections)
        if list(self.counts) != names or any(self.counts[n] != expected[n] for n in names):
            raise ValueError(f"counts {self.counts} must list every class in order and match the detections")
        return self