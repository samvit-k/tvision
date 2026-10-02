"""OpenCV rendering of boxes. Visualization only; never computes detections."""

from collections.abc import Iterable, Mapping

import cv2
import numpy as np

CLASS_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "tumor": (40, 40, 230),
    "non_tumor": (220, 160, 30),
    "ambiguous": (0, 200, 230),
}
DEFAULT_COLOR = (200, 200, 200)

def draw_boxes(
    image_bgr: np.ndarray,
    boxes: Iterable[Mapping],
    thickness: int = 1,
    show_labels: bool = False,
) -> np.ndarray:
    """Return a copy of `image_bgr` with boxes drawn.

    Each box is a mapping with x1, y1, x2, y2 (pixels), `class_name`, and optional `confidence`.
    """
    out = image_bgr.copy()
    for b in boxes:
        color = CLASS_COLORS.get(b["class_name"], DEFAULT_COLOR)
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        cv2.rectangle(out, p1, p2, color, thickness)
        if show_labels:
            text = b["class_name"]
            if b.get("confidence") is not None:
                text += f" {b['confidence']:.2f}"
            cv2.putText(out, text, (p1[0], max(p1[1] - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out

def side_by_side(*images: np.ndarray, gap: int = 6) -> np.ndarray:
    """Concatenate images horizontally, padding each to the tallest height."""
    h = max(im.shape[0] for im in images)
    padded = []
    for i, im in enumerate(images):
        right = gap if i < len(images) - 1 else 0
        padded.append(cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, right,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return np.hstack(padded)