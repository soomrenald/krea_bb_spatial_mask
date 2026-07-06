import importlib.util
import pathlib
import sys

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "krea2_regional_lora_masks.py"
spec = importlib.util.spec_from_file_location("k2mod", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_parse_box_indices():
    assert mod._parse_box_indices("1,3-5") == [0, 2, 3, 4]
    assert mod._parse_box_indices([1, "3-4"]) == [0, 2, 3]


def test_bbox_xywh():
    box = mod._bbox_from_any((100, 200, 300, 400), 1000, 1000, "xywh")
    assert box == (0.1, 0.2, 0.4, 0.6)


def test_rect_mask_shape():
    mask = mod._rect_token_mask(8, 8, (0.25, 0.25, 0.75, 0.75), 0.05)
    assert tuple(mask.shape) == (64,)
    assert float(mask.max()) <= 1.0
    assert float(mask.min()) >= 0.0


def test_rect_position_mask_uses_normalized_token_coords():
    x = mod.torch.tensor([0.125, 0.875, 0.125, 0.875])
    y = mod.torch.tensor([0.125, 0.125, 0.875, 0.875])
    image = mod.torch.tensor([True, True, True, True])
    mask = mod._rect_position_mask(x, y, image, (0.0, 0.0, 0.5, 0.5), 0.01)
    assert int(mask.argmax()) == 0
    assert float(mask[0]) > 0.99
    assert float(mask[1]) < 0.01
    assert float(mask[2]) < 0.01
