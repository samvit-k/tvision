import numpy as np
import pandas as pd
import pytest

from tvision import nucls_data as nd
from tvision.config import Classes
from tvision.explore import (EDGE_TOLERANCE_PX, check_coords_numeric, check_fov_rects, check_geometry,
                             fov_box_visibility)

STEM = "TCGA-A1-A0SK-DX1_id-5ea4095addda5f8398977ebc_left-7053_top-53967_bottom-54231_right-7311"
INDEX = pd.DataFrame([{"stem": STEM, "width": 100, "height": 80}])

def _ann(**overrides) -> pd.DataFrame:
    row = {"stem": STEM, "type": "rectangle", "xmin": 10, "ymin": 10, "xmax": 20, "ymax": 25,
           "super_classification": "tumor_any"}
    return pd.DataFrame([{**row, **overrides}])

def test_slide_and_hospital_parsing():
    assert nd.slide_of(STEM) == "TCGA-A1-A0SK-DX1"
    assert nd.hospital_of("TCGA-A1-A0SK-DX1") == "A1"

def test_class_mapping_is_complete_and_ordered():
    ann = pd.DataFrame({"super_classification": ["tumor_any", "sTIL", "nonTIL_stromal", "other_nucleus", "AMBIGUOUS"]})
    mapped = nd.map_classes(ann, Classes.load())
    assert mapped["project_class"].tolist() == ["tumor", "non_tumor", "non_tumor", "non_tumor", "ambiguous"]
    assert mapped["class_id"].tolist() == [0, 1, 1, 1, 2]

def test_unmapped_super_class_raises():
    with pytest.raises(ValueError, match="Unmapped"):
        nd.map_classes(pd.DataFrame({"super_classification": ["unknown"]}), Classes.load())

@pytest.mark.parametrize("value", [None, "abc", float("nan")])
def test_non_numeric_coords_detected(value):
    result = check_coords_numeric(_ann(xmax=value))
    assert not result["passed"] and result["detail"] == {"xmax": 1}

def test_numeric_coords_pass():
    assert check_coords_numeric(_ann())["passed"]

def test_geometry_valid_box_passes():
    assert all(c["passed"] for c in check_geometry(_ann(), INDEX).values())

def test_geometry_flags_out_of_bounds_and_inverted():
    checks = check_geometry(
        pd.concat([_ann(xmax=105), _ann(xmin=30, xmax=20)]),
        INDEX
    )

    assert not checks["boxes_within_image"]["passed"]
    assert checks["boxes_within_image"]["detail"]["right_bottom_beyond_tolerance"] == 1

    assert not checks["no_inverted_boxes"]["passed"]
    assert checks["no_inverted_boxes"]["detail"] == 1

def test_right_bottom_overshoot_within_tolerance_passes():
    checks = check_geometry(_ann(xmax=100 + EDGE_TOLERANCE_PX, ymax=80 + EDGE_TOLERANCE_PX), INDEX)
    assert checks["boxes_within_image"]["passed"]
    assert checks["boxes_within_image"]["detail"]["right_bottom_beyond_0px"] == 1

@pytest.mark.parametrize("overrides", [{"xmin": -1}, {"ymin": -1}])
def test_any_left_top_overshoot_fails(overrides):
    checks = check_geometry(_ann(**overrides), INDEX)
    assert not checks["boxes_within_image"]["passed"]
    assert checks["boxes_within_image"]["detail"]["left_top_overshoot"] == 1

def test_point_annotations_not_flagged_as_zero_area():
    checks = check_geometry(_ann(type="point", xmin=5, ymin=5, xmax=5, ymax=5), INDEX)
    assert checks["no_zero_area_non_point_boxes"]["passed"]

def test_fov_rect_from_mask_codes():
    codes = np.zeros((10, 12), np.uint8)
    codes[2:8, 3:11] = nd.FOV_CODE
    codes[0:4, 0:2] = 1  # nucleus outside the FOV must not extend the rectangle
    assert nd.fov_rect(codes) == (3, 2, 11, 8)
    assert nd.fov_rect(np.zeros((4, 4), np.uint8)) is None

def test_fov_rects_check():
    rect = {"fov_x0": 0, "fov_y0": 0, "fov_x1": 100, "fov_y1": 80}
    assert check_fov_rects(INDEX.assign(**rect))["passed"]
    assert check_fov_rects(INDEX.assign(**{**rect, "fov_x1": 100 + EDGE_TOLERANCE_PX}))["passed"]
    assert not check_fov_rects(INDEX.assign(**{**rect, "fov_x1": 101 + EDGE_TOLERANCE_PX}))["passed"]
    assert not check_fov_rects(INDEX.assign(**{**rect, "fov_y0": -1}))["passed"]
    assert not check_fov_rects(INDEX.assign(**{**rect, "fov_x1": None}))["passed"]

def test_fov_box_visibility():
    index = INDEX.assign(fov_x0=0, fov_y0=0, fov_x1=15, fov_y1=80)
    ann = pd.concat([_ann(), _ann(xmin=20, xmax=30)], ignore_index=True)  # half inside, fully outside
    assert fov_box_visibility(ann, index).tolist() == [0.5, 0.0]