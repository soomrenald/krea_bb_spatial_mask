"""
Standalone ComfyUI custom nodes for region-masked Krea2 multi-LoRA application.

Design goals:
- standalone, self-contained implementation
- separate multi-LoRA loader node
- ordered LoRA list with human-readable aliases
- one LoRA may target multiple independent boxes
- preview/report helpers so box order is easy to identify
- spatial masking of each LoRA delta on image tokens
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import safetensors.torch
except Exception as e:  # pragma: no cover
    safetensors = None
    _SAFETENSORS_IMPORT_ERROR = e
else:
    _SAFETENSORS_IMPORT_ERROR = None

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

try:
    import comfy.patcher_extension as patcher_extension
except Exception:  # pragma: no cover
    patcher_extension = None

try:
    from PIL import Image, ImageDraw
except Exception as e:  # pragma: no cover
    Image = None
    ImageDraw = None
    _PIL_IMPORT_ERROR = e
else:
    _PIL_IMPORT_ERROR = None

LOGGER = logging.getLogger("Krea2RegionalMultiLoRA")
WEB_DIRECTORY = "./web"
WRAPPER_KEY = "krea2_regional_multi_lora_standalone_v1"
NONE_LORA = "None"
LORA_STACK_TYPE = "KREA2_MULTI_LORA_STACK"

DEFAULT_LORAS_JSON = json.dumps(
    [
        {
            "enabled": True,
            "alias": "character_a",
            "lora": "None",
            "strength": 1.0,
            "boxes": "1",
            "color": "#ff5f57",
        },
        {
            "enabled": True,
            "alias": "character_b",
            "lora": "None",
            "strength": 1.0,
            "boxes": "2",
            "color": "#5fb3ff",
        },
    ],
    indent=2,
)

COMMON_PREFIXES = (
    "lora_unet_",
    "lora_",
    "diffusion_model.",
    "diffusion_model_",
    "model.diffusion_model.",
    "model.diffusion_model_",
    "model.",
    "base_model.model.",
    "base_model.",
    "transformer.",
    "transformer_",
    "unet.",
    "unet_",
)

DEFAULT_EXCLUDED_NAME_FRAGMENTS = (
    "txtfusion",
    "txt_fusion",
    "textfusion",
    "text_fusion",
    "txtmlp",
    "txt_mlp",
    "textmlp",
    "text_mlp",
    "t_embedder",
    "time_embed",
    "time_embedding",
    "timestep",
    "tmlp",
    "tproj",
    "final_layer",
    "last.",
    ".last",
    "pos_embed",
    "posemb",
)


@dataclass
class LoraSelection:
    enabled: bool
    alias: str
    lora: str
    strength: float
    boxes: List[int]
    color: str = "#ff5f57"


@dataclass
class LoraStack:
    selections: List[LoraSelection]


@dataclass
class LoraMatrices:
    down: torch.Tensor  # [rank, in]
    up: torch.Tensor  # [out, rank]
    scale: float
    source_key: str


@dataclass
class LayerPatch:
    selection_index: int
    layer_name: str
    lora_key: str
    strength: float
    matrices: LoraMatrices


@dataclass
class TokenLayout:
    imglen: Optional[int] = None
    txtlen: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    source: str = "unknown"


@dataclass
class RuntimeSession:
    layout: TokenLayout
    mask_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = field(default_factory=dict)
    tensor_cache: Dict[Tuple[int, str, str, str], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    warned: set = field(default_factory=set)


class RegionalApplierState:
    def __init__(
        self,
        stack: LoraStack,
        boxes: List[Tuple[float, float, float, float]],
        layer_entries: Dict[str, List[LayerPatch]],
        seam_feather: float,
        outside_strength: float,
        base_strength: float,
        token_offset_mode: str,
        manual_image_start: int,
        image_rows: int,
        image_cols: int,
        debug: bool,
        canvas_aspect: float,
    ):
        self.stack = stack
        self.boxes = boxes
        self.layer_entries = layer_entries
        self.seam_feather = float(seam_feather)
        self.outside_strength = float(outside_strength)
        self.base_strength = float(base_strength)
        self.token_offset_mode = token_offset_mode
        self.manual_image_start = int(manual_image_start)
        self.image_rows = int(image_rows)
        self.image_cols = int(image_cols)
        self.debug = bool(debug)
        self.canvas_aspect = float(canvas_aspect) if canvas_aspect else 1.0
        self.session: Optional[RuntimeSession] = None

    def wrapper(self, executor, *args, **kwargs):
        model_obj = getattr(executor, "class_obj", None)
        handles = []
        self.session = RuntimeSession(layout=self._infer_layout(args, kwargs))
        try:
            if model_obj is None:
                return executor(*args, **kwargs)
            name_to_module = dict(model_obj.named_modules())
            for layer_name, entries in self.layer_entries.items():
                module = name_to_module.get(layer_name)
                if module is None:
                    continue
                handles.append(module.register_forward_hook(self._make_forward_hook(entries)))
            if self.debug and handles:
                LOGGER.info(
                    "[Krea2RegionalMultiLoRA] installed %d hooks; layout=%s",
                    len(handles),
                    self.session.layout,
                )
            return executor(*args, **kwargs)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.session = None

    def _infer_layout(self, args: Sequence[Any], kwargs: Dict[str, Any]) -> TokenLayout:
        imglen = None
        txtlen = None
        source: List[str] = []

        img = kwargs.get("img", None)
        if torch.is_tensor(img) and img.ndim == 3:
            imglen = int(img.shape[1])
            source.append("kw_img")
        elif args and torch.is_tensor(args[0]) and args[0].ndim == 3:
            imglen = int(args[0].shape[1])
            source.append("arg0_img")

        total_unpadded = None
        mask = kwargs.get("mask", kwargs.get("attention_mask", None))
        candidates = []
        if torch.is_tensor(mask):
            candidates.append(mask)
        for a in args:
            if torch.is_tensor(a) and a.ndim == 2 and a.shape[1] > 1:
                candidates.append(a)
        for c in candidates:
            if imglen is None or int(c.shape[1]) >= imglen:
                total_unpadded = int(c.shape[1])
                source.append("mask_total")
                break
        if imglen is not None and total_unpadded is not None:
            txtlen = max(0, total_unpadded - imglen)

        if txtlen is None:
            context = kwargs.get("context", None)
            if context is None and len(args) > 1 and torch.is_tensor(args[1]):
                context = args[1]
            if torch.is_tensor(context):
                if context.ndim == 3:
                    txtlen = int(context.shape[1])
                    source.append("context_rank3")
                elif context.ndim == 4:
                    txtlen = int(context.shape[1])
                    source.append("context_rank4_best_effort")

        if self.image_rows > 0 and self.image_cols > 0:
            rows, cols = self.image_rows, self.image_cols
            if imglen is None:
                imglen = rows * cols
                source.append("manual_grid_imglen")
        else:
            rows, cols = _infer_grid(imglen, self.image_rows, self.image_cols, self.canvas_aspect)
            if rows and cols:
                source.append("factor_grid")

        return TokenLayout(imglen=imglen, txtlen=txtlen, rows=rows, cols=cols, source="+".join(source) or "unknown")

    def _make_forward_hook(self, entries: List[LayerPatch]):
        def hook(module, inputs, output):
            session = self.session
            if session is None or not torch.is_tensor(output) or not inputs:
                return output
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim != 3 or output.ndim != 3:
                return output
            if x.shape[0] != output.shape[0] or x.shape[1] != output.shape[1]:
                return output
            seq_len = int(x.shape[1])
            out = output
            compute_dtype = _compute_dtype_for(x)
            for entry in entries:
                selection = self.stack.selections[entry.selection_index]
                if not selection.enabled:
                    continue
                mask = self._mask_for_selection(entry.selection_index, seq_len, x.device, compute_dtype)
                if mask is None:
                    if self.debug and seq_len not in session.warned:
                        session.warned.add(seq_len)
                        LOGGER.warning(
                            "[Krea2RegionalMultiLoRA] no usable token mask for seq_len=%s layout=%s; skipped layer %s",
                            seq_len,
                            session.layout,
                            entry.layer_name,
                        )
                    continue
                down, up = self._matrices_on_device(entry, x.device, compute_dtype)
                xin = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
                delta = F.linear(F.linear(xin, down), up)
                delta = delta * (entry.matrices.scale * entry.strength * self.base_strength)
                if self.outside_strength != 0.0:
                    mask = mask + (1.0 - mask) * self.outside_strength
                out = out + (delta * mask).to(dtype=out.dtype)
            return out

        return hook

    def _matrices_on_device(self, entry: LayerPatch, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        session = self.session
        assert session is not None
        key = (entry.selection_index, entry.layer_name + ":" + entry.lora_key, str(device), str(dtype))
        cached = session.tensor_cache.get(key)
        if cached is not None:
            return cached
        down = entry.matrices.down.to(device=device, dtype=dtype, non_blocking=True)
        up = entry.matrices.up.to(device=device, dtype=dtype, non_blocking=True)
        session.tensor_cache[key] = (down, up)
        return down, up

    def _mask_for_selection(self, selection_index: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        session = self.session
        assert session is not None
        layout = session.layout
        imglen = layout.imglen
        if imglen is None or imglen <= 0:
            return None
        rows, cols = layout.rows, layout.cols
        if not rows or not cols or rows * cols != imglen:
            rows, cols = _infer_grid(imglen, self.image_rows, self.image_cols, self.canvas_aspect)
        if not rows or not cols or rows * cols != imglen:
            return None

        if self.token_offset_mode == "manual":
            image_start = max(0, int(self.manual_image_start))
        elif seq_len == imglen:
            image_start = 0
        elif self.token_offset_mode == "legacy_trailing":
            image_start = max(0, seq_len - imglen)
        elif layout.txtlen is not None and seq_len >= layout.txtlen + imglen:
            image_start = int(layout.txtlen)
        else:
            image_start = max(0, seq_len - imglen)

        if image_start + imglen > seq_len:
            return None

        key = (selection_index, seq_len, str(device), str(dtype))
        cached = session.mask_cache.get(key)
        if cached is not None:
            return cached

        selection = self.stack.selections[selection_index]
        box_indices = [i for i in selection.boxes if 0 <= i < len(self.boxes)]
        if not box_indices:
            return None

        token_mask = torch.zeros((rows * cols,), dtype=torch.float32)
        for box_index in box_indices:
            bbox = self.boxes[box_index]
            box_mask = _rect_token_mask(rows, cols, bbox, self.seam_feather)
            token_mask = torch.maximum(token_mask, box_mask)
        token_mask = token_mask.to(device=device, dtype=dtype)

        full = torch.zeros((seq_len,), device=device, dtype=dtype)
        full[image_start:image_start + imglen] = token_mask
        full = full.view(1, seq_len, 1)
        session.mask_cache[key] = full
        return full


# -------------------- generic helpers --------------------


def _lora_names() -> List[str]:
    if folder_paths is None:
        return [NONE_LORA]
    try:
        names = folder_paths.get_filename_list("loras")
        return [NONE_LORA] + [n for n in names if n != NONE_LORA]
    except Exception:
        return [NONE_LORA]


def _resolve_lora_path(name: str) -> Optional[str]:
    if not name or name == NONE_LORA:
        return None
    if os.path.exists(name):
        return name
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("loras", name)
            if path and os.path.exists(path):
                return path
        except Exception:
            pass
    return name if os.path.exists(name) else None


def _json_loads_maybe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            LOGGER.warning("Invalid JSON: %s", s[:256])
            return default
    return default


def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    if v is None:
        return default
    return bool(v)


def _parse_box_indices(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, int):
        return [max(0, v - 1)]
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, int):
                out.append(max(0, item - 1))
            elif isinstance(item, str):
                out.extend(_parse_box_indices(item))
        return sorted(set(out))
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        out = set()
        for part in re.split(r"[,;\s]+", s):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                if a.strip().isdigit() and b.strip().isdigit():
                    lo = int(a.strip())
                    hi = int(b.strip())
                    if hi < lo:
                        lo, hi = hi, lo
                    for i in range(lo, hi + 1):
                        out.add(max(0, i - 1))
            elif part.isdigit():
                out.add(max(0, int(part) - 1))
        return sorted(out)
    return []


def _sanitize_alias(v: Any, default: str) -> str:
    s = str(v or default).strip()
    return s or default


def _sanitize_hex_color(v: Any, default: str = "#ff5f57") -> str:
    s = str(v or default).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return s.lower()
    return default


def _stack_to_jsonable(stack: LoraStack) -> List[Dict[str, Any]]:
    return [
        {
            "enabled": s.enabled,
            "alias": s.alias,
            "lora": s.lora,
            "strength": s.strength,
            "boxes": [i + 1 for i in s.boxes],
            "color": s.color,
        }
        for s in stack.selections
    ]


def _parse_lora_stack(loras_json: str) -> LoraStack:
    raw = _json_loads_maybe(loras_json, [])
    if isinstance(raw, dict):
        raw = [raw]
    selections: List[LoraSelection] = []
    for i, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        alias = _sanitize_alias(item.get("alias", f"lora_{i + 1}"), f"lora_{i + 1}")
        lora = str(item.get("lora", NONE_LORA) or NONE_LORA)
        try:
            strength = float(item.get("strength", 1.0))
        except Exception:
            strength = 1.0
        boxes = _parse_box_indices(item.get("boxes", item.get("box_indices", "")))
        color = _sanitize_hex_color(item.get("color", "#ff5f57"), "#ff5f57")
        enabled = _as_bool(item.get("enabled", True), True)
        selections.append(LoraSelection(enabled=enabled, alias=alias, lora=lora, strength=strength, boxes=boxes, color=color))
    return LoraStack(selections=selections)


def _normalize_key(name: str) -> str:
    s = name.strip().lower()
    for prefix in COMMON_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r"\.(weight|bias)$", "", s)
    s = re.sub(r"_(weight|bias)$", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _compute_dtype_for(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return x.dtype
    if torch.cuda.is_available() and x.device.type == "cuda":
        return torch.float16
    return torch.float32


def _factor_pairs(n: int) -> Iterable[Tuple[int, int]]:
    if n <= 0:
        return []
    out = []
    r = int(math.sqrt(n))
    for a in range(1, r + 1):
        if n % a == 0:
            out.append((a, n // a))
            if a != n // a:
                out.append((n // a, a))
    return out


def _infer_grid(imglen: Optional[int], manual_rows: int, manual_cols: int, aspect: float) -> Tuple[Optional[int], Optional[int]]:
    if manual_rows > 0 and manual_cols > 0:
        return manual_rows, manual_cols
    if imglen is None or imglen <= 0:
        return None, None
    pairs = list(_factor_pairs(imglen))
    if not pairs:
        return None, None
    target = max(1e-6, aspect)
    pairs.sort(key=lambda rc: abs((rc[1] / max(1, rc[0])) - target))
    return pairs[0]


def _rect_token_mask(rows: int, cols: int, bbox: Tuple[float, float, float, float], feather: float) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    c = torch.arange(cols, dtype=torch.float32) + 0.5
    r = torch.arange(rows, dtype=torch.float32) + 0.5
    cc = c.unsqueeze(0).expand(rows, cols)
    rr = r.unsqueeze(1).expand(rows, cols)
    fx = max(1e-4, float(feather) * max(1.0, cols))
    fy = max(1e-4, float(feather) * max(1.0, rows))
    left = torch.sigmoid((cc - x0 * cols) / fx)
    right = torch.sigmoid((x1 * cols - cc) / fx)
    top = torch.sigmoid((rr - y0 * rows) / fy)
    bottom = torch.sigmoid((y1 * rows - rr) / fy)
    return (left * right * top * bottom).reshape(-1).clamp(0.0, 1.0)


# -------------------- bbox parsing --------------------


def _normalize_bboxes_input(bboxes: Any) -> List[Any]:
    if bboxes is None:
        return []
    if isinstance(bboxes, tuple) and len(bboxes) == 1:
        bboxes = bboxes[0]
    if isinstance(bboxes, str):
        bboxes = _json_loads_maybe(bboxes, [])
    if isinstance(bboxes, dict):
        for k in ("boxes", "bboxes", "regions", "items"):
            if isinstance(bboxes.get(k), list):
                return bboxes[k]
        return [bboxes]
    if isinstance(bboxes, list):
        if len(bboxes) == 1 and isinstance(bboxes[0], list):
            return bboxes[0]
        return bboxes
    return []


def _bbox_from_xywh(x: float, y: float, w: float, h: float, canvas_w: int, canvas_h: int) -> Optional[Tuple[float, float, float, float]]:
    return _bbox_from_xyxy(x, y, x + w, y + h, canvas_w, canvas_h)


def _bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float, canvas_w: int, canvas_h: int) -> Optional[Tuple[float, float, float, float]]:
    try:
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
            cw = max(1.0, float(canvas_w))
            ch = max(1.0, float(canvas_h))
            x0, x1 = x0 / cw, x1 / cw
            y0, y1 = y0 / ch, y1 / ch
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = min(1.0, max(0.0, float(x0)))
        y0 = min(1.0, max(0.0, float(y0)))
        x1 = min(1.0, max(0.0, float(x1)))
        y1 = min(1.0, max(0.0, float(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        return None


def _bbox_from_any(box: Any, canvas_w: int, canvas_h: int, list_format: str = "xyxy") -> Optional[Tuple[float, float, float, float]]:
    if box is None:
        return None
    try:
        if isinstance(box, str):
            box = _json_loads_maybe(box, None)
        if isinstance(box, dict):
            if "bbox" in box and box["bbox"] is not box:
                return _bbox_from_any(box["bbox"], canvas_w, canvas_h, list_format)
            # KJNodes / modern BoundingBox object
            if "x" in box and "y" in box and ("width" in box or "w" in box) and ("height" in box or "h" in box):
                return _bbox_from_xywh(
                    float(box.get("x", 0.0)),
                    float(box.get("y", 0.0)),
                    float(box.get("width", box.get("w", 0.0))),
                    float(box.get("height", box.get("h", 0.0))),
                    canvas_w,
                    canvas_h,
                )
            if "x1" in box and "y1" in box:
                return _bbox_from_xyxy(
                    float(box.get("x0", box.get("x", 0.0))),
                    float(box.get("y0", box.get("y", 0.0))),
                    float(box.get("x1")),
                    float(box.get("y1")),
                    canvas_w,
                    canvas_h,
                )
            return _bbox_from_xywh(
                float(box.get("x0", box.get("x", 0.0))),
                float(box.get("y0", box.get("y", 0.0))),
                float(box.get("w", box.get("width", 0.0))),
                float(box.get("h", box.get("height", 0.0))),
                canvas_w,
                canvas_h,
            )

        vals = list(box)[:4]
        if len(vals) < 4:
            return None
        x0, y0, a, b = [float(v) for v in vals]
        if list_format == "xywh":
            return _bbox_from_xywh(x0, y0, a, b, canvas_w, canvas_h)
        if list_format == "xyxy":
            return _bbox_from_xyxy(x0, y0, a, b, canvas_w, canvas_h)
        # auto
        if a > x0 and b > y0 and max(abs(x0), abs(y0), abs(a), abs(b)) <= 1.0:
            return _bbox_from_xyxy(x0, y0, a, b, canvas_w, canvas_h)
        return _bbox_from_xywh(x0, y0, a, b, canvas_w, canvas_h)
    except Exception:
        return None


def _ideogram_boxes_from_prompt_json(prompt_json: Any, canvas_w: int, canvas_h: int) -> List[Tuple[float, float, float, float]]:
    raw = _json_loads_maybe(prompt_json, None)
    if raw is None:
        return []
    out = []
    entries = raw if isinstance(raw, list) else [raw]
    for item in entries:
        if not isinstance(item, dict):
            continue
        # common nesting: {"prompt": [{"bbox": [ymin, xmin, ymax, xmax], ...}, ...]}
        candidate_lists = []
        if isinstance(item.get("prompt"), list):
            candidate_lists.append(item.get("prompt"))
        if isinstance(item.get("regions"), list):
            candidate_lists.append(item.get("regions"))
        if isinstance(item.get("items"), list):
            candidate_lists.append(item.get("items"))
        for lst in candidate_lists:
            for region in lst:
                if not isinstance(region, dict):
                    continue
                bbox = region.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in bbox[:4]]
                    out_box = _bbox_from_xyxy(xmin / 1000.0, ymin / 1000.0, xmax / 1000.0, ymax / 1000.0, canvas_w, canvas_h)
                    if out_box is not None:
                        out.append(out_box)
    return out


def _collect_boxes(
    bboxes: Any,
    kj_bboxes: Any,
    ideogram_prompt_json: Any,
    canvas_w: int,
    canvas_h: int,
    bbox_list_format: str,
) -> List[Tuple[float, float, float, float]]:
    modern = [_bbox_from_any(b, canvas_w, canvas_h, "xyxy") for b in _normalize_bboxes_input(bboxes)]
    modern = [b for b in modern if b is not None]
    if modern:
        return modern

    legacy = [_bbox_from_any(b, canvas_w, canvas_h, bbox_list_format) for b in _normalize_bboxes_input(kj_bboxes)]
    legacy = [b for b in legacy if b is not None]
    if legacy:
        return legacy

    ideogram = _ideogram_boxes_from_prompt_json(ideogram_prompt_json, canvas_w, canvas_h)
    return ideogram


# -------------------- lora loading and patch graph --------------------


def _load_lora(path: str) -> Dict[str, LoraMatrices]:
    if _SAFETENSORS_IMPORT_ERROR is not None:
        raise RuntimeError(f"safetensors import failed: {_SAFETENSORS_IMPORT_ERROR}")
    sd = safetensors.torch.load_file(path, device="cpu")
    groups: Dict[str, Dict[str, torch.Tensor]] = {}
    alphas: Dict[str, float] = {}

    for key, value in sd.items():
        k = str(key)
        if k.endswith(".alpha") or k.endswith("_alpha") or k.endswith(".scale"):
            base = re.sub(r"(\.alpha|_alpha|\.scale)$", "", k)
            try:
                alphas[base] = float(value.item())
            except Exception:
                pass
            continue
        if k.endswith(".lora_down.weight"):
            base = k[: -len(".lora_down.weight")]
            groups.setdefault(base, {})["down"] = value
        elif k.endswith(".lora_up.weight"):
            base = k[: -len(".lora_up.weight")]
            groups.setdefault(base, {})["up"] = value
        elif k.endswith(".down.weight"):
            base = k[: -len(".down.weight")]
            groups.setdefault(base, {})["down"] = value
        elif k.endswith(".up.weight"):
            base = k[: -len(".up.weight")]
            groups.setdefault(base, {})["up"] = value

    out: Dict[str, LoraMatrices] = {}
    for base, parts in groups.items():
        down = parts.get("down")
        up = parts.get("up")
        if down is None or up is None:
            continue
        if down.ndim != 2 or up.ndim != 2:
            continue
        rank = int(down.shape[0])
        alpha = float(alphas.get(base, rank))
        scale = alpha / max(1, rank)
        out[_normalize_key(base)] = LoraMatrices(down=down.contiguous(), up=up.contiguous(), scale=scale, source_key=base)
    return out


def _iter_named_linears(model_obj, apply_to: str) -> Iterable[Tuple[str, Any]]:
    for name, module in model_obj.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        lname = name.lower()
        if apply_to == "krea_blocks_only":
            if any(fragment in lname for fragment in DEFAULT_EXCLUDED_NAME_FRAGMENTS):
                continue
            if not any(token in lname for token in ("double_blocks", "single_blocks", "blocks.", "joint_blocks", "img_mlp", "img_attn", "attn", "mlp")):
                continue
        yield name, module


def _build_layer_entries(model_obj, stack: LoraStack, apply_to: str) -> Tuple[Dict[str, List[LayerPatch]], List[str]]:
    all_entries: Dict[str, List[LayerPatch]] = {}
    report: List[str] = []
    model_modules = list(_iter_named_linears(model_obj, apply_to))
    model_key_map = {_normalize_key(name): (name, module) for name, module in model_modules}

    for sel_idx, selection in enumerate(stack.selections):
        if not selection.enabled:
            report.append(f"[{sel_idx + 1}] {selection.alias}: disabled")
            continue
        path = _resolve_lora_path(selection.lora)
        if not path:
            report.append(f"[{sel_idx + 1}] {selection.alias}: missing LoRA '{selection.lora}'")
            continue
        lora_sd = _load_lora(path)
        matched = 0
        shape_mismatch = 0
        for lora_key, matrices in lora_sd.items():
            pair = model_key_map.get(lora_key)
            if pair is None:
                continue
            layer_name, module = pair
            weight = getattr(module, "weight")
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            if int(matrices.down.shape[1]) != in_features or int(matrices.up.shape[0]) != out_features or int(matrices.up.shape[1]) != int(matrices.down.shape[0]):
                shape_mismatch += 1
                continue
            all_entries.setdefault(layer_name, []).append(
                LayerPatch(
                    selection_index=sel_idx,
                    layer_name=layer_name,
                    lora_key=lora_key,
                    strength=selection.strength,
                    matrices=matrices,
                )
            )
            matched += 1
        report.append(
            f"[{sel_idx + 1}] {selection.alias}: file={selection.lora} boxes={','.join(str(i + 1) for i in selection.boxes) or '-'} matched_layers={matched} shape_mismatch={shape_mismatch}"
        )
    return all_entries, report


# -------------------- report / preview --------------------


def _format_assignment_report(stack: LoraStack, boxes: List[Tuple[float, float, float, float]], include_box_coords: bool = True) -> str:
    lines = [f"LoRA count: {len(stack.selections)}", f"BBox count: {len(boxes)}", "Assignments:"]
    for i, sel in enumerate(stack.selections, start=1):
        box_numbers = [b + 1 for b in sel.boxes]
        lines.append(
            f"  [{i}] alias={sel.alias} lora={sel.lora} enabled={sel.enabled} strength={sel.strength:.3f} boxes={box_numbers if box_numbers else []} color={sel.color}"
        )
    if include_box_coords and boxes:
        lines.append("Boxes:")
        for i, box in enumerate(boxes, start=1):
            x0, y0, x1, y1 = box
            lines.append(f"  box {i}: ({x0:.4f}, {y0:.4f}) -> ({x1:.4f}, {y1:.4f})")
    return "\n".join(lines)


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _draw_preview(stack: LoraStack, boxes: List[Tuple[float, float, float, float]], width: int, height: int) -> torch.Tensor:
    if _PIL_IMPORT_ERROR is not None:
        raise RuntimeError(f"PIL import failed: {_PIL_IMPORT_ERROR}")
    width = max(64, int(width))
    height = max(64, int(height))
    img = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(img)

    # grid
    for x in range(0, width, max(32, width // 8)):
        draw.line((x, 0, x, height), fill=(40, 40, 40), width=1)
    for y in range(0, height, max(32, height // 8)):
        draw.line((0, y, width, y), fill=(40, 40, 40), width=1)

    # build reverse map box -> lora aliases
    box_to_aliases: Dict[int, List[str]] = {i: [] for i in range(len(boxes))}
    box_to_colors: Dict[int, Tuple[int, int, int]] = {}
    for idx, sel in enumerate(stack.selections, start=1):
        rgb = tuple(int(c * 255) for c in _hex_to_rgb01(sel.color))
        for box_idx in sel.boxes:
            if 0 <= box_idx < len(boxes):
                box_to_aliases.setdefault(box_idx, []).append(f"[{idx}] {sel.alias}")
                box_to_colors[box_idx] = rgb

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        px0 = int(round(x0 * width))
        py0 = int(round(y0 * height))
        px1 = int(round(x1 * width))
        py1 = int(round(y1 * height))
        color = box_to_colors.get(i, (220, 220, 220))
        draw.rectangle((px0, py0, px1, py1), outline=color, width=4)
        label = f"box {i + 1}"
        aliases = box_to_aliases.get(i) or []
        if aliases:
            label += "\n" + "\n".join(aliases)
        tx = max(0, min(width - 10, px0 + 6))
        ty = max(0, min(height - 10, py0 + 6))
        bbox = draw.multiline_textbbox((tx, ty), label)
        draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
        draw.multiline_text((tx, ty), label, fill=color)

    tensor = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0
    return tensor.unsqueeze(0)


# -------------------- ComfyUI nodes --------------------


class Krea2MultiLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras_json": ("STRING", {"multiline": True, "default": DEFAULT_LORAS_JSON}),
            }
        }

    RETURN_TYPES = (LORA_STACK_TYPE, "STRING")
    RETURN_NAMES = ("lora_stack", "report")
    FUNCTION = "build"
    CATEGORY = "Krea2/Regional LoRA"

    def build(self, loras_json):
        stack = _parse_lora_stack(loras_json)
        report = _format_assignment_report(stack, [])
        payload = {"selections": _stack_to_jsonable(stack)}
        return (payload, report)


class Krea2RegionalLoRAApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": (LORA_STACK_TYPE,),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 65535}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 65535}),
                "bbox_list_format": (["xywh", "xyxy", "auto"], {"default": "xywh"}),
                "seam_feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005}),
                "outside_strength": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "token_offset_mode": (["auto_txt_img_pad_safe", "manual", "legacy_trailing"], {"default": "auto_txt_img_pad_safe"}),
                "manual_image_start": ("INT", {"default": 0, "min": 0, "max": 65535}),
                "image_rows": ("INT", {"default": 0, "min": 0, "max": 65535}),
                "image_cols": ("INT", {"default": 0, "min": 0, "max": 65535}),
                "apply_to": (["krea_blocks_only", "all_matched_linears"], {"default": "krea_blocks_only"}),
                "debug_logging": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "kj_bboxes": ("BBOX",),
                "ideogram_prompt_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "Krea2/Regional LoRA"

    def apply(
        self,
        model,
        lora_stack,
        canvas_width,
        canvas_height,
        bbox_list_format,
        seam_feather,
        outside_strength,
        base_strength,
        token_offset_mode,
        manual_image_start,
        image_rows,
        image_cols,
        apply_to,
        debug_logging,
        bboxes=None,
        kj_bboxes=None,
        ideogram_prompt_json="",
    ):
        if patcher_extension is None:
            raise RuntimeError("This node requires a recent ComfyUI build with comfy.patcher_extension")
        stack = _parse_lora_stack(json.dumps(lora_stack.get("selections", []))) if isinstance(lora_stack, dict) else _parse_lora_stack(lora_stack)
        boxes = _collect_boxes(bboxes, kj_bboxes, ideogram_prompt_json, canvas_width, canvas_height, bbox_list_format)
        if not stack.selections:
            return (model, "No LoRAs selected.")
        if not boxes:
            return (model, _format_assignment_report(stack, []) + "\nNo external boxes were provided.")

        model_out = model.clone()
        model_obj = model_out.get_model_object("diffusion_model")
        layer_entries, lines = _build_layer_entries(model_obj, stack, apply_to)
        aspect = float(canvas_width) / max(1.0, float(canvas_height))
        state = RegionalApplierState(
            stack=stack,
            boxes=boxes,
            layer_entries=layer_entries,
            seam_feather=seam_feather,
            outside_strength=outside_strength,
            base_strength=base_strength,
            token_offset_mode=token_offset_mode,
            manual_image_start=manual_image_start,
            image_rows=image_rows,
            image_cols=image_cols,
            debug=bool(debug_logging),
            canvas_aspect=aspect,
        )
        model_out.add_wrapper_with_key(
            patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            state.wrapper,
        )
        report = _format_assignment_report(stack, boxes) + "\n\nPatch summary:\n" + "\n".join(lines)
        return (model_out, report)


class Krea2RegionalLoRAPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": (LORA_STACK_TYPE,),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 65535}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 65535}),
                "preview_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "preview_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "bbox_list_format": (["xywh", "xyxy", "auto"], {"default": "xywh"}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "kj_bboxes": ("BBOX",),
                "ideogram_prompt_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "report")
    FUNCTION = "preview"
    CATEGORY = "Krea2/Regional LoRA"

    def preview(self, lora_stack, canvas_width, canvas_height, preview_width, preview_height, bbox_list_format, bboxes=None, kj_bboxes=None, ideogram_prompt_json=""):
        stack = _parse_lora_stack(json.dumps(lora_stack.get("selections", []))) if isinstance(lora_stack, dict) else _parse_lora_stack(lora_stack)
        boxes = _collect_boxes(bboxes, kj_bboxes, ideogram_prompt_json, canvas_width, canvas_height, bbox_list_format)
        img = _draw_preview(stack, boxes, preview_width, preview_height)
        report = _format_assignment_report(stack, boxes)
        return (img, report)


NODE_CLASS_MAPPINGS = {
    "Krea2MultiLoRALoader": Krea2MultiLoRALoader,
    "Krea2RegionalLoRAApply": Krea2RegionalLoRAApply,
    "Krea2RegionalLoRAPreview": Krea2RegionalLoRAPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2MultiLoRALoader": "Krea2 Multi LoRA Loader",
    "Krea2RegionalLoRAApply": "Krea2 Regional LoRA Apply",
    "Krea2RegionalLoRAPreview": "Krea2 Regional LoRA Preview",
}
