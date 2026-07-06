"""
Krea2 Regional LoRA Masks for ComfyUI.

This node applies multiple Krea2 LoRAs regionally by injecting each LoRA's
low-rank activation delta only on spatial image tokens that fall inside that
region's bounding box/mask.

Core idea:
    y = base_linear(x) + spatial_mask * LoRA_delta(x)

Main safety improvement over tail-token masking:
    Official Krea2 concatenates text tokens before image tokens and pads after
    them. The preferred offset is therefore txtlen : txtlen + imglen, inferred
    from the Krea2 forward call's img/context/mask tensors when available.

This is a prototype custom node intended for Krea2-style MMDiT models in recent
ComfyUI builds that expose ModelPatcher.add_wrapper_with_key and
comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import traceback
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

LOGGER = logging.getLogger("Krea2RegionalLoRAMasks")
WRAPPER_KEY = "krea2_regional_lora_masks_v2"
NONE_LORA = "None"

DEFAULT_REGIONS_JSON = """[
  {
    "name": "left_character",
    "lora": "None",
    "strength": 1.0,
    "enabled": true,
    "bbox": {"x": 0.05, "y": 0.05, "w": 0.40, "h": 0.85}
  },
  {
    "name": "right_character",
    "lora": "None",
    "strength": 1.0,
    "enabled": true,
    "bbox": {"x": 0.55, "y": 0.05, "w": 0.40, "h": 0.85}
  }
]"""

COMMON_PREFIXES = (
    "lora_unet_",
    "lora_te_",
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

# Modules outside the single-stream image/text transformer generally either do
# not have a spatial image-token axis or are too global for identity isolation.
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


def _lora_names() -> List[str]:
    if folder_paths is None:
        return [NONE_LORA]
    try:
        names = folder_paths.get_filename_list("loras")
        return [NONE_LORA] + [n for n in names if n != NONE_LORA]
    except Exception:
        return [NONE_LORA]


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


def _normalize_key(name: str) -> str:
    s = name.strip().lower()
    for prefix in COMMON_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Remove common weight suffixes but keep useful module path information.
    s = re.sub(r"\.(weight|bias)$", "", s)
    s = re.sub(r"_(weight|bias)$", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


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


def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    if v is None:
        return default
    return bool(v)


@dataclass
class RegionSpec:
    name: str
    lora: str
    strength: float = 1.0
    enabled: bool = True
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class LoraMatrices:
    down: torch.Tensor  # [rank, in]
    up: torch.Tensor  # [out, rank]
    scale: float
    source_key: str


@dataclass
class LayerEntry:
    region_index: int
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
    mask_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = field(default_factory=dict)
    tensor_cache: Dict[Tuple[int, str, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    warned: set = field(default_factory=set)


class RegionalState:
    def __init__(
        self,
        regions: List[RegionSpec],
        layer_entries: Dict[str, List[LayerEntry]],
        boxes: List[Tuple[float, float, float, float]],
        seam_feather: float,
        outside_strength: float,
        base_strength: float,
        token_offset_mode: str,
        manual_image_start: int,
        image_rows: int,
        image_cols: int,
        debug: bool,
        canvas_aspect: float = 1.0,
    ):
        self.regions = regions
        self.layer_entries = layer_entries
        self.boxes = boxes
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
                    "[Krea2RegionalLoRA] installed %d hooks; layout=%s",
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
        source = []

        # Official Krea2 forward: forward(img, context, t, pos, mask=None).
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
                    # Krea2 text fusion commonly receives [B, L, N, D] and later
                    # produces [B, L, D]. This is best-effort; mask/pos is safer.
                    txtlen = int(context.shape[1])
                    source.append("context_rank4_best_effort")

        if self.image_rows > 0 and self.image_cols > 0:
            rows, cols = self.image_rows, self.image_cols
            if imglen is None:
                imglen = rows * cols
                source.append("manual_grid_imglen")
        else:
            rows, cols = _infer_grid(imglen, self.image_rows, self.image_cols, self._canvas_aspect())
            if rows and cols:
                source.append("factor_grid")

        return TokenLayout(imglen=imglen, txtlen=txtlen, rows=rows, cols=cols, source="+".join(source) or "unknown")

    def _canvas_aspect(self) -> float:
        # Placeholder hook for future per-state canvas aspect if needed by cache.
        # Current grid inference uses bbox coordinates only and therefore does not
        # need absolute canvas size here; rows/cols manual override is preferred.
        return self.canvas_aspect

    def _make_forward_hook(self, entries: List[LayerEntry]):
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
            if not entries:
                return output

            out = output
            compute_dtype = _compute_dtype_for(x)
            for entry in entries:
                region = self.regions[entry.region_index]
                if not region.enabled:
                    continue
                mask = self._mask_for_region(entry.region_index, seq_len, x.device, compute_dtype)
                if mask is None:
                    if self.debug and seq_len not in session.warned:
                        session.warned.add(seq_len)
                        LOGGER.warning(
                            "[Krea2RegionalLoRA] no usable token mask for seq_len=%s layout=%s; skipped layer %s",
                            seq_len,
                            session.layout,
                            entry.layer_name,
                        )
                    continue
                down, up = self._matrices_on_device(entry, x.device, compute_dtype)
                xin = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
                try:
                    delta = F.linear(F.linear(xin, down), up)
                except Exception:
                    LOGGER.error(
                        "[Krea2RegionalLoRA] failed LoRA matmul at layer=%s lora_key=%s x=%s down=%s up=%s",
                        entry.layer_name,
                        entry.lora_key,
                        tuple(x.shape),
                        tuple(down.shape),
                        tuple(up.shape),
                    )
                    raise
                delta = delta * (entry.matrices.scale * entry.strength * self.base_strength)
                if self.outside_strength != 0.0:
                    mask = mask + (1.0 - mask) * self.outside_strength
                out = out + (delta * mask).to(dtype=out.dtype)
            return out

        return hook

    def _matrices_on_device(self, entry: LayerEntry, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        session = self.session
        assert session is not None
        key = (entry.region_index, entry.layer_name + ":" + entry.lora_key, device, dtype)
        cached = session.tensor_cache.get(key)
        if cached is not None:
            return cached
        down = entry.matrices.down.to(device=device, dtype=dtype, non_blocking=True)
        up = entry.matrices.up.to(device=device, dtype=dtype, non_blocking=True)
        session.tensor_cache[key] = (down, up)
        return down, up

    def _mask_for_region(self, region_index: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        session = self.session
        assert session is not None
        layout = session.layout
        imglen = layout.imglen
        if imglen is None or imglen <= 0:
            return None
        rows, cols = layout.rows, layout.cols
        if not rows or not cols or rows * cols != imglen:
            rows, cols = _infer_grid(imglen, self.image_rows, self.image_cols, 1.0)
        if not rows or not cols or rows * cols != imglen:
            return None

        image_start: Optional[int]
        if self.token_offset_mode == "manual":
            image_start = max(0, int(self.manual_image_start))
        elif seq_len == imglen:
            image_start = 0
        elif self.token_offset_mode == "legacy_trailing":
            image_start = max(0, seq_len - imglen)
        elif layout.txtlen is not None and seq_len >= layout.txtlen + imglen:
            image_start = int(layout.txtlen)
        else:
            # Last resort for models that do not expose masks/context through the
            # wrapper. This is deliberately only fallback behavior.
            image_start = max(0, seq_len - imglen)

        if image_start is None or image_start + imglen > seq_len:
            return None
        key = (region_index, seq_len, device, dtype)
        cached = session.mask_cache.get(key)
        if cached is not None:
            return cached

        bbox = self.boxes[region_index] if region_index < len(self.boxes) else None
        if bbox is None:
            return None
        token_mask = _rect_token_mask(rows, cols, bbox, self.seam_feather).to(device=device, dtype=dtype)
        full = torch.zeros((seq_len,), device=device, dtype=dtype)
        full[image_start:image_start + imglen] = token_mask
        full = full.view(1, seq_len, 1)
        session.mask_cache[key] = full
        return full


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
    # cols / rows should approximate aspect.
    pairs.sort(key=lambda rc: abs((rc[1] / max(1, rc[0])) - target))
    return pairs[0]


def _rect_token_mask(rows: int, cols: int, bbox: Tuple[float, float, float, float], feather: float) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    c = torch.arange(cols, dtype=torch.float32) + 0.5
    r = torch.arange(rows, dtype=torch.float32) + 0.5
    cc = c.unsqueeze(0).expand(rows, cols)
    rr = r.unsqueeze(1).expand(rows, cols)
    # Feather is a fraction of token-grid size. Small epsilon prevents hard
    # numerical division issues while still allowing effectively hard masks.
    fx = max(1e-4, float(feather) * max(1.0, cols))
    fy = max(1e-4, float(feather) * max(1.0, rows))
    left = torch.sigmoid((cc - x0 * cols) / fx)
    right = torch.sigmoid((x1 * cols - cc) / fx)
    top = torch.sigmoid((rr - y0 * rows) / fy)
    bottom = torch.sigmoid((y1 * rows - rr) / fy)
    return (left * right * top * bottom).reshape(-1).clamp(0.0, 1.0)


def _parse_regions(regions_json: str) -> List[RegionSpec]:
    raw = _json_loads_maybe(regions_json, [])
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    regions = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        lora = str(item.get("lora", item.get("lora_name", NONE_LORA)) or NONE_LORA)
        try:
            strength = float(item.get("strength", item.get("strength_model", 1.0)))
        except Exception:
            strength = 1.0
        enabled = _as_bool(item.get("enabled", item.get("enable", True)), True)
        name = str(item.get("name", f"region_{i+1}") or f"region_{i+1}")
        bbox = _bbox_from_any(item.get("bbox", item), canvas_w=1, canvas_h=1)
        regions.append(RegionSpec(name=name, lora=lora, strength=strength, enabled=enabled, bbox=bbox))
    return regions


def _normalize_bboxes_input(bboxes: Any) -> List[Any]:
    if bboxes is None:
        return []
    # Common KJNodes style outputs may be tuple/list-wrapped.
    if isinstance(bboxes, tuple) and len(bboxes) == 1:
        bboxes = bboxes[0]
    if isinstance(bboxes, str):
        bboxes = _json_loads_maybe(bboxes, [])
    if isinstance(bboxes, dict):
        # Could be {"boxes": [...]} or one box.
        for k in ("boxes", "bboxes", "regions", "items"):
            if isinstance(bboxes.get(k), list):
                return bboxes[k]
        return [bboxes]
    if isinstance(bboxes, list):
        if len(bboxes) == 1 and isinstance(bboxes[0], list):
            return bboxes[0]
        return bboxes
    return []


def _bbox_from_any(box: Any, canvas_w: int, canvas_h: int) -> Optional[Tuple[float, float, float, float]]:
    if box is None:
        return None
    try:
        if isinstance(box, str):
            box = _json_loads_maybe(box, None)
        if isinstance(box, dict):
            if "bbox" in box and box["bbox"] is not box:
                return _bbox_from_any(box["bbox"], canvas_w, canvas_h)
            if "x1" in box and "y1" in box:
                x0 = float(box.get("x0", box.get("x", 0.0)))
                y0 = float(box.get("y0", box.get("y", 0.0)))
                x1 = float(box.get("x1"))
                y1 = float(box.get("y1"))
            else:
                x0 = float(box.get("x0", box.get("x", 0.0)))
                y0 = float(box.get("y0", box.get("y", 0.0)))
                w = float(box.get("w", box.get("width", box.get("W", 0.0))))
                h = float(box.get("h", box.get("height", box.get("H", 0.0))))
                x1 = x0 + w
                y1 = y0 + h
        else:
            vals = list(box)[:4]
            if len(vals) < 4:
                return None
            x0, y0, x1, y1 = [float(v) for v in vals]
            # For list values, assume x0,y0,x1,y1 unless x1/y1 look like width/height.
            if x1 <= x0 or y1 <= y0:
                x1 = x0 + max(0.0, x1)
                y1 = y0 + max(0.0, y1)

        max_abs = max(abs(x0), abs(y0), abs(x1), abs(y1))
        if max_abs > 1.0:
            cw = max(1.0, float(canvas_w))
            ch = max(1.0, float(canvas_h))
            x0, x1 = x0 / cw, x1 / cw
            y0, y1 = y0 / ch, y1 / ch
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = min(1.0, max(0.0, x0))
        y0 = min(1.0, max(0.0, y0))
        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        return None


def _auto_split_boxes(n: int, mode: str) -> List[Tuple[float, float, float, float]]:
    if n <= 0:
        return []
    out = []
    if mode == "auto_horizontal":
        for i in range(n):
            out.append((0.0, i / n, 1.0, (i + 1) / n))
    else:
        for i in range(n):
            out.append((i / n, 0.0, (i + 1) / n, 1.0))
    return out


def _collect_boxes(regions: List[RegionSpec], bboxes: Any, split_mode: str, canvas_w: int, canvas_h: int) -> List[Optional[Tuple[float, float, float, float]]]:
    boxes: List[Optional[Tuple[float, float, float, float]]] = []
    external = [_bbox_from_any(b, canvas_w, canvas_h) for b in _normalize_bboxes_input(bboxes)]
    external = [b for b in external if b is not None]

    for i, r in enumerate(regions):
        if i < len(external):
            boxes.append(external[i])
        elif r.bbox is not None:
            boxes.append(r.bbox)
        else:
            boxes.append(None)

    if any(b is None for b in boxes) and split_mode in {"auto_vertical", "auto_horizontal"}:
        auto = _auto_split_boxes(len(regions), split_mode)
        boxes = [boxes[i] if boxes[i] is not None else auto[i] for i in range(len(regions))]
    return boxes


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
                alphas[_normalize_key(base)] = float(value.flatten()[0].item())
            except Exception:
                pass
            continue
        m = re.match(r"^(.*)\.(lora_down|lora_A|down)\.weight$", k)
        if m:
            groups.setdefault(_normalize_key(m.group(1)), {})["down"] = value.detach().float().contiguous()
            continue
        m = re.match(r"^(.*)\.(lora_up|lora_B|up)\.weight$", k)
        if m:
            groups.setdefault(_normalize_key(m.group(1)), {})["up"] = value.detach().float().contiguous()
            continue

    out: Dict[str, LoraMatrices] = {}
    for base, mats in groups.items():
        down = mats.get("down")
        up = mats.get("up")
        if down is None or up is None:
            continue
        if down.ndim != 2 or up.ndim != 2:
            continue
        rank = int(down.shape[0])
        alpha = alphas.get(base, float(rank))
        out[base] = LoraMatrices(down=down, up=up, scale=float(alpha) / max(1.0, float(rank)), source_key=base)
    return out


def _is_linear_like(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return torch.is_tensor(weight) and weight.ndim == 2 and callable(getattr(module, "forward", None))


def _include_layer_name(name: str, apply_to: str) -> bool:
    lname = name.lower()
    if apply_to == "all_matched_linears":
        return True
    if any(frag in lname for frag in DEFAULT_EXCLUDED_NAME_FRAGMENTS):
        return False
    # Krea2 official model uses blocks.* for the single-stream DiT blocks.
    return "blocks" in lname or ".block" in lname


def _best_lora_match(module_name: str, module: torch.nn.Module, lora: Dict[str, LoraMatrices]) -> Optional[Tuple[str, LoraMatrices]]:
    mod_key = _normalize_key(module_name)
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        return None
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])

    candidates = []
    for lk, mats in lora.items():
        if not _lora_shape_matches(mats, in_features, out_features):
            continue
        if lk == mod_key:
            score = 100000 + len(lk)
        elif mod_key.endswith(lk) or lk.endswith(mod_key):
            score = 50000 + min(len(lk), len(mod_key))
        elif lk in mod_key or mod_key in lk:
            score = 1000 + min(len(lk), len(mod_key))
        else:
            continue
        candidates.append((score, lk, mats))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, lk, mats = candidates[0]
    return lk, mats


def _lora_shape_matches(mats: LoraMatrices, in_features: int, out_features: int) -> bool:
    return int(mats.down.shape[1]) == in_features and int(mats.up.shape[0]) == out_features and int(mats.down.shape[0]) == int(mats.up.shape[1])


def _build_layer_entries(model_obj: torch.nn.Module, regions: List[RegionSpec], apply_to: str) -> Tuple[Dict[str, List[LayerEntry]], List[str]]:
    report = []
    layer_entries: Dict[str, List[LayerEntry]] = {}
    modules = [(n, m) for n, m in model_obj.named_modules() if _is_linear_like(m) and _include_layer_name(n, apply_to)]

    loaded_cache: Dict[str, Dict[str, LoraMatrices]] = {}
    for ridx, region in enumerate(regions):
        if not region.enabled or not region.lora or region.lora == NONE_LORA:
            report.append(f"region {ridx+1} {region.name}: disabled/no LoRA")
            continue
        path = _resolve_lora_path(region.lora)
        if not path:
            report.append(f"region {ridx+1} {region.name}: LoRA not found: {region.lora}")
            continue
        if path not in loaded_cache:
            loaded_cache[path] = _load_lora(path)
        lora = loaded_cache[path]
        matched = 0
        for layer_name, module in modules:
            bm = _best_lora_match(layer_name, module, lora)
            if bm is None:
                continue
            lora_key, mats = bm
            layer_entries.setdefault(layer_name, []).append(
                LayerEntry(
                    region_index=ridx,
                    layer_name=layer_name,
                    lora_key=lora_key,
                    strength=region.strength,
                    matrices=mats,
                )
            )
            matched += 1
        report.append(f"region {ridx+1} {region.name}: matched {matched} layers from {os.path.basename(path)}")
    return layer_entries, report


class Krea2RegionalLoRAMasks:
    CATEGORY = "Krea2/Regional LoRA"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regions_json": ("STRING", {"multiline": True, "default": DEFAULT_REGIONS_JSON}),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "split_mode": (["bbox_or_json", "auto_vertical", "auto_horizontal"], {"default": "bbox_or_json"}),
                "seam_feather": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.005}),
                "outside_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "token_offset_mode": (["auto_txt_img_pad_safe", "manual", "legacy_trailing"], {"default": "auto_txt_img_pad_safe"}),
                "manual_image_start": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "image_rows": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "image_cols": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "apply_to": (["krea_blocks_only", "all_matched_linears"], {"default": "krea_blocks_only"}),
                "debug_logging": (["off", "on"], {"default": "off"}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
            },
        }

    def apply(
        self,
        model,
        regions_json: str,
        canvas_width: int,
        canvas_height: int,
        split_mode: str,
        seam_feather: float,
        outside_strength: float,
        base_strength: float,
        token_offset_mode: str,
        manual_image_start: int,
        image_rows: int,
        image_cols: int,
        apply_to: str,
        debug_logging: str,
        bboxes=None,
    ):
        if patcher_extension is None:
            raise RuntimeError("This node requires recent ComfyUI with comfy.patcher_extension.")
        if not hasattr(model, "clone"):
            raise RuntimeError("Expected a ComfyUI MODEL / ModelPatcher with clone().")
        if not hasattr(model, "add_wrapper_with_key"):
            raise RuntimeError("Expected recent ComfyUI ModelPatcher.add_wrapper_with_key(). Update ComfyUI.")

        regions = _parse_regions(regions_json)
        if not regions:
            return (model, "No valid regions_json entries; model unchanged.")

        boxes = _collect_boxes(regions, bboxes, split_mode, canvas_width, canvas_height)
        valid_regions = []
        valid_boxes = []
        for r, b in zip(regions, boxes):
            if b is None:
                LOGGER.warning("[Krea2RegionalLoRA] region %s has no valid bbox; disabling", r.name)
                r.enabled = False
            valid_regions.append(r)
            valid_boxes.append(b)

        model_obj = None
        try:
            model_obj = model.get_model_object("diffusion_model")
        except Exception:
            try:
                model_obj = model.model.diffusion_model
            except Exception:
                try:
                    model_obj = model.model
                except Exception:
                    model_obj = None
        if model_obj is None:
            raise RuntimeError("Could not access diffusion_model from MODEL.")

        layer_entries, report_lines = _build_layer_entries(model_obj, valid_regions, apply_to)
        if not layer_entries:
            report = "No LoRA layers matched. Check that these are Krea2 LoRAs and that the filenames exist.\n" + "\n".join(report_lines)
            return (model, report)

        cloned = model.clone()
        # Avoid stacking duplicate copies with the same key when this node is re-run.
        try:
            if hasattr(cloned, "remove_wrappers_with_key"):
                cloned.remove_wrappers_with_key(patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        except Exception:
            pass

        state = RegionalState(
            regions=valid_regions,
            layer_entries=layer_entries,
            boxes=valid_boxes,
            seam_feather=seam_feather,
            outside_strength=outside_strength,
            base_strength=base_strength,
            token_offset_mode=token_offset_mode,
            manual_image_start=manual_image_start,
            image_rows=image_rows,
            image_cols=image_cols,
            debug=debug_logging == "on",
            canvas_aspect=max(1e-6, float(canvas_width) / max(1.0, float(canvas_height))),
        )
        cloned.add_wrapper_with_key(patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, state.wrapper)
        total_hooks = sum(len(v) for v in layer_entries.values())
        report = [
            "Krea2 Regional LoRA Masks installed.",
            f"regions: {len(valid_regions)}",
            f"matched live layers: {len(layer_entries)}",
            f"regional layer-entry count: {total_hooks}",
            f"token_offset_mode: {token_offset_mode}",
            "boxes: " + json.dumps(valid_boxes),
            *report_lines,
        ]
        return (cloned, "\n".join(report))


class Krea2RegionMaskPreview:
    CATEGORY = "Krea2/Regional LoRA"
    FUNCTION = "preview"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions_json": ("STRING", {"multiline": True, "default": DEFAULT_REGIONS_JSON}),
                "canvas_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "canvas_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "preview_height": ("INT", {"default": 64, "min": 8, "max": 2048}),
                "preview_width": ("INT", {"default": 64, "min": 8, "max": 2048}),
                "seam_feather": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.005}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
            },
        }

    def preview(self, regions_json, canvas_width, canvas_height, preview_height, preview_width, seam_feather, bboxes=None):
        regions = _parse_regions(regions_json)
        boxes = _collect_boxes(regions, bboxes, "bbox_or_json", canvas_width, canvas_height)
        masks = []
        for b in boxes:
            if b is None:
                masks.append(torch.zeros((preview_height, preview_width), dtype=torch.float32))
            else:
                masks.append(_rect_token_mask(preview_height, preview_width, b, seam_feather).reshape(preview_height, preview_width))
        if not masks:
            masks = [torch.zeros((preview_height, preview_width), dtype=torch.float32)]
        out = torch.stack(masks, dim=0)
        report = f"preview masks: {len(masks)} at {preview_width}x{preview_height}"
        return (out, report)


NODE_CLASS_MAPPINGS = {
    "Krea2RegionalLoRAMasks": Krea2RegionalLoRAMasks,
    "Krea2RegionMaskPreview": Krea2RegionMaskPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalLoRAMasks": "Krea2 Regional LoRA Masks (patched)",
    "Krea2RegionMaskPreview": "Krea2 Region Mask Preview",
}
