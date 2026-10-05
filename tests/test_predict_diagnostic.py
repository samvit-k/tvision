import inspect

import pytest
import torch
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import nms

from tvision import predict_diagnostic as pdx

NMS_SIGNATURE = inspect.signature(nms.non_max_suppression)

class Stop(Exception):
    pass

def _nms_call(predictor_cls, monkeypatch) -> dict:
    """The full, defaults-applied argument set the predictor passes to non_max_suppression."""
    seen = {}

    def capture(*args, **kwargs):
        bound = NMS_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.update({k: v for k, v in bound.arguments.items() if k != "prediction"})
        raise Stop

    monkeypatch.setattr(nms, "non_max_suppression", capture)
    predictor = predictor_cls(overrides={"conf": 0.24, "iou": 0.7, "max_det": 1000})
    with pytest.raises(Stop):
        predictor.postprocess(torch.zeros(1, 7, 10), torch.zeros(1, 3, 64, 64), [])
    return seen

def test_multilabel_predictor_differs_from_ultralytics_only_in_multi_label(monkeypatch):
    parent = _nms_call(DetectionPredictor, monkeypatch)
    ours = _nms_call(pdx.MultiLabelPredictor, monkeypatch)
    assert parent["multi_label"] is False and ours["multi_label"] is True
    assert {k: v for k, v in ours.items() if k != "multi_label"} == {k: v for k, v in parent.items()
                                                                      if k != "multi_label"}
    assert ours["conf_thres"] == 0.24 and ours["iou_thres"] == 0.7 and ours["max_det"] == 1000

def test_conditions_cover_rect_by_nms_mode_once_each():
    assert sorted(pdx.CONDITIONS.values()) == [(False, False), (False, True), (True, False), (True, True)]
    assert pdx.CONDITIONS["deployed"] == (False, False) and pdx.SPLIT == "val"