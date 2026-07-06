import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krea2_regional_lora_masks import _rect_token_mask, _bbox_from_any, _infer_grid


def test_bbox_normalized():
    assert _bbox_from_any({"x": 10, "y": 20, "w": 30, "h": 40}, 100, 200) == (0.1, 0.1, 0.4, 0.3)


def test_grid_factor():
    assert _infer_grid(64, 0, 0, 1.0) == (8, 8)


def test_mask_shape():
    m = _rect_token_mask(8, 8, (0.0, 0.0, 0.5, 1.0), 0.05)
    assert tuple(m.shape) == (64,)
    assert float(m.max()) <= 1.0
    assert float(m.min()) >= 0.0
