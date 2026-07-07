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
import types
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
    import comfy.lora as comfy_lora
    import comfy.lora_convert as comfy_lora_convert
    import comfy.utils as comfy_utils
except Exception:  # pragma: no cover
    comfy_lora = None
    comfy_lora_convert = None
    comfy_utils = None

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
NODE_VERSION = "2026-07-06.13-output-delta-blend"
TEXT_TOKEN_STRENGTH = 0.0
KREA_OUTPUT_IMAGE_INDICATOR = 2
OUTPUT_DELTA_BLEND_MODE = True

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

GLOBAL_CONDITIONING_NAME_FRAGMENTS = (
    "txtfusion",
    "txt_fusion",
    "textfusion",
    "text_fusion",
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
    mask_scope: str = "regional"


@dataclass
class TokenLayout:
    imglen: Optional[int] = None
    txtlen: Optional[int] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    source: str = "unknown"
    grid_candidates: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class RuntimeSession:
    layout: TokenLayout
    transformer_options: Dict[str, Any] = field(default_factory=dict)
    mask_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = field(default_factory=dict)
    tensor_cache: Dict[Tuple[int, str, str, str], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    observed_seq_lens: Dict[int, int] = field(default_factory=dict)
    mask_debug: Dict[Tuple[int, int], str] = field(default_factory=dict)
    debug_logged: set = field(default_factory=set)
    warned: set = field(default_factory=set)
    wrapper_calls: int = 0
    hook_calls: int = 0
    applied_calls: int = 0
    skipped_no_mask: int = 0
    geometry_updates: int = 0
    geometry_source: str = "none"
    geometry_seq_len: Optional[int] = None
    geometry_image_mask: Optional[torch.Tensor] = None
    geometry_x_norm: Optional[torch.Tensor] = None
    geometry_y_norm: Optional[torch.Tensor] = None
    geometry_rows: Optional[int] = None
    geometry_cols: Optional[int] = None
    output_blend_passes: int = 0


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
        transformer_options = _find_transformer_options(args, kwargs)
        layout = self._infer_layout(args, kwargs)
        self._apply_runtime_latent_grid(layout, args, model_obj)
        self.session = RuntimeSession(
            layout=layout,
            transformer_options=transformer_options,
            wrapper_calls=1,
        )
        original_backbone = None
        try:
            if model_obj is None:
                return executor(*args, **kwargs)
            if OUTPUT_DELTA_BLEND_MODE:
                return self._wrapper_output_delta_blend(executor, model_obj, args, kwargs)
            original_backbone = self._install_krea_backbone_capture(model_obj)
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
            result = executor(*args, **kwargs)
            if self.debug:
                LOGGER.info(
                    "[Krea2RegionalMultiLoRA] runtime stats: hooks=%d applied=%d skipped_no_mask=%d observed_seq_lens=%s mask_debug=%s geometry_updates=%d geometry=%s seq=%s grid=%sx%s transformer_img_slice=%s layout=%s",
                    self.session.hook_calls,
                    self.session.applied_calls,
                    self.session.skipped_no_mask,
                    dict(sorted(self.session.observed_seq_lens.items())),
                    _format_sample(self.session.mask_debug.values(), 4),
                    self.session.geometry_updates,
                    self.session.geometry_source,
                    self.session.geometry_seq_len,
                    self.session.geometry_rows,
                    self.session.geometry_cols,
                    self.session.transformer_options.get("img_slice"),
                    self.session.layout,
                )
            return result
        finally:
            if original_backbone is not None:
                try:
                    setattr(model_obj, "_backbone", original_backbone)
                except Exception:
                    pass
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.session = None

    def _wrapper_output_delta_blend(self, executor, model_obj: Any, args: Sequence[Any], kwargs: Dict[str, Any]):
        session = self.session
        assert session is not None
        name_to_module = dict(model_obj.named_modules())
        if self.debug:
            LOGGER.info(
                "[Krea2RegionalMultiLoRA] output-delta blend mode enabled; baseline plus one global LoRA pass per enabled selection"
            )
        base = executor(*args, **kwargs)
        if not torch.is_tensor(base):
            return base
        result = base
        pass_reports: List[str] = []
        for selection_index, selection in enumerate(self.stack.selections):
            if not selection.enabled:
                continue
            if not [i for i in selection.boxes if 0 <= i < len(self.boxes)]:
                continue
            handles = self._install_selection_hooks(name_to_module, selection_index, force_global=True)
            if not handles:
                continue
            try:
                session.output_blend_passes += 1
                lora_out = executor(*args, **kwargs)
            finally:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
            if not torch.is_tensor(lora_out) or lora_out.shape != base.shape:
                if self.debug:
                    LOGGER.warning(
                        "[Krea2RegionalMultiLoRA] output blend skipped alias=%s because output shape changed: base=%s lora=%s",
                        selection.alias,
                        tuple(base.shape),
                        getattr(lora_out, "shape", None),
                    )
                continue
            mask = self._output_mask_for_selection(selection_index, base)
            if mask is None:
                if self.outside_strength != 0.0:
                    mask = torch.full(_spatial_broadcast_shape(base), float(self.outside_strength), device=base.device, dtype=base.dtype)
                else:
                    continue
            elif self.outside_strength != 0.0:
                mask = mask + (1.0 - mask) * self.outside_strength
            delta = lora_out - base
            result = result + delta * mask
            if self.debug:
                pass_reports.append(
                    f"{selection.alias}:mask_range=({float(mask.min().detach().cpu()):.4f},{float(mask.max().detach().cpu()):.4f}) "
                    f"mask_sum={float(mask.sum().detach().cpu()):.2f} "
                    f"delta_mean={float(delta.abs().mean().detach().cpu()):.6f} delta_max={float(delta.abs().max().detach().cpu()):.6f}"
                )
        if self.debug:
            LOGGER.info(
                "[Krea2RegionalMultiLoRA] output blend stats: passes=%d observed_seq_lens=%s applied_hook_calls=%d reports=%s",
                session.output_blend_passes,
                dict(sorted(session.observed_seq_lens.items())),
                session.applied_calls,
                _format_sample(pass_reports, 8),
            )
        return result

    def _install_selection_hooks(
        self,
        name_to_module: Dict[str, Any],
        selection_index: int,
        force_global: bool,
        scope_filter: Optional[str] = None,
    ) -> List[Any]:
        handles = []
        for layer_name, entries in self.layer_entries.items():
            selected = [entry for entry in entries if entry.selection_index == selection_index]
            if scope_filter is not None:
                selected = [entry for entry in selected if entry.mask_scope == scope_filter]
            if not selected:
                continue
            module = name_to_module.get(layer_name)
            if module is None:
                continue
            handles.append(module.register_forward_hook(self._make_forward_hook(selected, force_global=force_global)))
        return handles

    def _output_mask_for_selection(self, selection_index: int, output: torch.Tensor) -> Optional[torch.Tensor]:
        if output.ndim < 4:
            return None
        selection = self.stack.selections[selection_index]
        box_indices = [i for i in selection.boxes if 0 <= i < len(self.boxes)]
        if not box_indices:
            return None
        rows = int(output.shape[-2])
        cols = int(output.shape[-1])
        token_mask = torch.zeros((rows * cols,), dtype=torch.float32)
        for box_index in box_indices:
            token_mask = torch.maximum(token_mask, _rect_token_mask(rows, cols, self.boxes[box_index], self.seam_feather))
        mask = token_mask.view(1, 1, rows, cols).to(device=output.device, dtype=output.dtype)
        return mask

    def _install_krea_backbone_capture(self, model_obj: Any) -> Optional[Any]:
        original_backbone = getattr(model_obj, "_backbone", None)
        if original_backbone is None or not callable(original_backbone):
            return None

        def patched_backbone(this_model, llm_features, x, t, position_ids, attn_mask, indicator, transformer_options={}):
            session = self.session
            if session is not None:
                if isinstance(transformer_options, dict):
                    session.transformer_options = transformer_options
                self._update_krea_geometry(position_ids, indicator)
            return original_backbone(llm_features, x, t, position_ids, attn_mask, indicator, transformer_options=transformer_options)

        setattr(model_obj, "_backbone", types.MethodType(patched_backbone, model_obj))
        return original_backbone

    def _update_krea_geometry(self, position_ids: Any, indicator: Any) -> None:
        session = self.session
        if session is None or not torch.is_tensor(position_ids) or not torch.is_tensor(indicator):
            return
        if position_ids.ndim != 3 or position_ids.shape[-1] < 3 or indicator.ndim != 2:
            return
        if int(position_ids.shape[1]) != int(indicator.shape[1]):
            return

        seq_len = int(indicator.shape[1])
        image_mask_2d = indicator == KREA_OUTPUT_IMAGE_INDICATOR
        if not bool(image_mask_2d.any().detach().cpu()):
            return
        image_mask = image_mask_2d.any(dim=0)
        if not bool(image_mask.any().detach().cpu()):
            return

        pos = position_ids[0]
        rows_raw = pos[:, 1].to(dtype=torch.float32)
        cols_raw = pos[:, 2].to(dtype=torch.float32)
        img_rows = rows_raw[image_mask]
        img_cols = cols_raw[image_mask]
        row_min = img_rows.min()
        col_min = img_cols.min()
        row_max = img_rows.max()
        col_max = img_cols.max()
        rows = max(1, int((row_max - row_min + 1).detach().cpu().item()))
        cols = max(1, int((col_max - col_min + 1).detach().cpu().item()))

        y_norm = torch.zeros((seq_len,), device=position_ids.device, dtype=torch.float32)
        x_norm = torch.zeros((seq_len,), device=position_ids.device, dtype=torch.float32)
        y_norm[image_mask] = (rows_raw[image_mask] - row_min + 0.5) / float(rows)
        x_norm[image_mask] = (cols_raw[image_mask] - col_min + 0.5) / float(cols)

        session.geometry_updates += 1
        session.geometry_source = "krea_backbone_position_ids_indicator"
        session.geometry_seq_len = seq_len
        session.geometry_image_mask = image_mask.detach()
        session.geometry_x_norm = x_norm.detach()
        session.geometry_y_norm = y_norm.detach()
        session.geometry_rows = rows
        session.geometry_cols = cols
        session.layout.imglen = int(image_mask.sum().detach().cpu().item())
        session.layout.txtlen = int(image_mask.to(torch.long).argmax().detach().cpu().item()) if bool(image_mask[0].logical_not().detach().cpu()) else 0
        session.layout.rows = rows
        session.layout.cols = cols
        if "krea_position_ids" not in session.layout.source:
            session.layout.source = (session.layout.source + "+krea_position_ids").strip("+")
        session.mask_cache.clear()

        if self.debug:
            coverage = []
            for idx, selection in enumerate(self.stack.selections):
                if not selection.enabled:
                    continue
                mask = self._mask_for_selection(idx, seq_len, position_ids.device, torch.float32)
                if mask is None:
                    coverage.append(f"{selection.alias}=none")
                else:
                    image_values = mask.view(seq_len)[image_mask]
                    coverage.append(
                        f"{selection.alias}=sum{float(image_values.sum().detach().cpu()):.1f}/max{float(image_values.max().detach().cpu()):.3f}"
                    )
            LOGGER.info(
                "[Krea2RegionalMultiLoRA] captured Krea geometry seq=%d image_tokens=%d grid=%dx%d coverage=%s",
                seq_len,
                int(image_mask.sum().detach().cpu().item()),
                rows,
                cols,
                ", ".join(coverage) if coverage else "-",
            )

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
            if context is None:
                for a in args[1:]:
                    if torch.is_tensor(a) and a.ndim == 3:
                        context = a
                        break
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

    def _apply_runtime_latent_grid(self, layout: TokenLayout, args: Sequence[Any], model_obj: Any) -> None:
        if not args or not torch.is_tensor(args[0]) or args[0].ndim < 4:
            return
        latent = args[0]
        h = int(latent.shape[-2])
        w = int(latent.shape[-1])
        patch_size = int(getattr(model_obj, "patch_size", 2) or 2)
        layout.grid_candidates.append((h, w, f"latent_hw_{h}x{w}"))
        if patch_size > 1 and h >= patch_size and w >= patch_size:
            layout.grid_candidates.append((max(1, h // patch_size), max(1, w // patch_size), f"latent_div_patch_{h}x{w}_p{patch_size}"))
        if h >= 2 and w >= 2 and patch_size != 2:
            layout.grid_candidates.append((max(1, h // 2), max(1, w // 2), f"latent_div_2_{h}x{w}"))
        if layout.imglen is None:
            layout.rows = h
            layout.cols = w
            layout.imglen = h * w
        layout.source = (layout.source + f"+latent_candidates_{h}x{w}_p{patch_size}").strip("+")

    def _make_forward_hook(self, entries: List[LayerPatch], force_global: bool = False):
        def hook(module, inputs, output):
            session = self.session
            if session is None or not torch.is_tensor(output) or not inputs:
                return output
            session.hook_calls += 1
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2 or output.ndim < 2:
                return output
            if x.shape[:-1] != output.shape[:-1]:
                return output
            seq_len = int(x.shape[-2])
            session.observed_seq_lens[seq_len] = session.observed_seq_lens.get(seq_len, 0) + 1
            out = output
            compute_dtype = _compute_dtype_for(x)
            for entry in entries:
                selection = self.stack.selections[entry.selection_index]
                if not selection.enabled:
                    continue
                if force_global or entry.mask_scope == "global_conditioning":
                    mask = torch.ones((1, seq_len, 1), device=x.device, dtype=compute_dtype)
                else:
                    mask = self._mask_for_selection(entry.selection_index, seq_len, x.device, compute_dtype)
                if mask is None:
                    if self.outside_strength != 0.0:
                        mask = torch.full((1, seq_len, 1), float(self.outside_strength), device=x.device, dtype=compute_dtype)
                    else:
                        session.skipped_no_mask += 1
                        if self.debug and seq_len not in session.warned:
                            session.warned.add(seq_len)
                            LOGGER.warning(
                                "[Krea2RegionalMultiLoRA] no usable token mask for seq_len=%s layout=%s; skipped layer %s",
                                seq_len,
                                session.layout,
                                entry.layer_name,
                        )
                        continue
                elif self.outside_strength != 0.0:
                    mask = mask + (1.0 - mask) * self.outside_strength
                if self.debug:
                    self._log_runtime_mask(entry, seq_len, mask)
                mask = _reshape_token_mask(mask, x.ndim)
                down, up = self._matrices_on_device(entry, x.device, compute_dtype)
                xin = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
                delta = F.linear(F.linear(xin, down), up)
                delta = delta * (entry.matrices.scale * entry.strength * self.base_strength)
                out = out + (delta * mask).to(dtype=out.dtype)
                session.applied_calls += 1
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

    def _log_runtime_mask(self, entry: LayerPatch, seq_len: int, mask: torch.Tensor) -> None:
        session = self.session
        if session is None:
            return
        key = (entry.selection_index, seq_len, entry.mask_scope)
        if key in session.debug_logged:
            return
        session.debug_logged.add(key)
        selection = self.stack.selections[entry.selection_index]
        flat = mask.reshape(-1)
        max_value = float(flat.max().detach().cpu()) if flat.numel() else 0.0
        min_value = float(flat.min().detach().cpu()) if flat.numel() else 0.0
        sum_value = float(flat.sum().detach().cpu()) if flat.numel() else 0.0
        nonzero = int((flat.abs() > 1e-5).sum().detach().cpu().item()) if flat.numel() else 0
        info = session.mask_debug.get((entry.selection_index, seq_len), "global_or_unknown_span")
        LOGGER.info(
            "[Krea2RegionalMultiLoRA] mask coverage alias=%s seq_len=%d scope=%s range=(%.4f, %.4f) sum=%.2f nonzero=%d outside=%.4f info=%s first_layer=%s",
            selection.alias,
            seq_len,
            entry.mask_scope,
            min_value,
            max_value,
            sum_value,
            nonzero,
            self.outside_strength,
            info,
            entry.layer_name,
        )

    def _mask_for_selection(self, selection_index: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        session = self.session
        assert session is not None
        layout = session.layout
        geometry_mask = self._mask_from_krea_geometry(selection_index, seq_len, device, dtype)
        if geometry_mask is not None:
            return geometry_mask

        chosen = self._choose_image_span(seq_len)
        if chosen is None:
            return None
        image_start, imglen, rows, cols, source = chosen
        layout.imglen = imglen
        layout.rows = rows
        layout.cols = cols
        if source not in layout.source:
            layout.source = (layout.source + f"+{source}").strip("+")

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
        if image_start > 0 and TEXT_TOKEN_STRENGTH != 0.0:
            full[:image_start] = float(TEXT_TOKEN_STRENGTH)
        full[image_start:image_start + imglen] = token_mask
        full = full.view(1, seq_len, 1)
        self._record_mask_debug(
            selection_index,
            seq_len,
            source,
            image_start,
            imglen,
            rows,
            cols,
            token_mask,
        )
        session.mask_cache[key] = full
        return full

    def _choose_image_span(self, seq_len: int) -> Optional[Tuple[int, int, int, int, str]]:
        session = self.session
        assert session is not None
        layout = session.layout
        candidates: List[Tuple[int, int, str]] = []

        if layout.txtlen is not None:
            txtlen = int(layout.txtlen)
            if 0 <= txtlen < seq_len:
                candidates.append((txtlen, seq_len - txtlen, "seq_minus_txt"))

        img_slice = session.transformer_options.get("img_slice")
        if isinstance(img_slice, (list, tuple)) and len(img_slice) >= 2:
            try:
                image_start = max(0, int(img_slice[0]))
                image_end = min(seq_len, int(img_slice[1]))
                if image_end > image_start:
                    candidates.append((image_start, image_end - image_start, "img_slice"))
            except Exception:
                pass

        for rows, cols, source in layout.grid_candidates:
            imglen = int(rows) * int(cols)
            if imglen <= 0 or imglen > seq_len:
                continue
            if seq_len == imglen:
                candidates.append((0, imglen, source))
            if layout.txtlen is not None and 0 <= int(layout.txtlen) <= seq_len - imglen:
                candidates.append((int(layout.txtlen), imglen, source + "_after_txt"))
            candidates.append((seq_len - imglen, imglen, source + "_trailing"))

        if layout.imglen is not None and int(layout.imglen) > 0 and int(layout.imglen) <= seq_len:
            imglen = int(layout.imglen)
            if seq_len == imglen:
                candidates.append((0, imglen, "layout_exact"))
            if layout.txtlen is not None and 0 <= int(layout.txtlen) <= seq_len - imglen:
                candidates.append((int(layout.txtlen), imglen, "layout_after_txt"))
            if self.token_offset_mode == "manual":
                candidates.append((max(0, int(self.manual_image_start)), imglen, "manual"))
            candidates.append((seq_len - imglen, imglen, "layout_trailing"))

        if self.image_rows > 0 and self.image_cols > 0:
            imglen = self.image_rows * self.image_cols
            if imglen <= seq_len:
                start = int(layout.txtlen) if layout.txtlen is not None and int(layout.txtlen) <= seq_len - imglen else seq_len - imglen
                candidates.append((start, imglen, "manual_grid"))

        seen = set()
        for image_start, imglen, source in candidates:
            if imglen <= 0 or image_start < 0 or image_start + imglen > seq_len:
                continue
            ident = (image_start, imglen)
            if ident in seen:
                continue
            seen.add(ident)
            rows, cols = self._grid_for_imglen(imglen)
            if rows and cols and rows * cols == imglen:
                return image_start, imglen, rows, cols, source
        return None

    def _grid_for_imglen(self, imglen: int) -> Tuple[Optional[int], Optional[int]]:
        layout = self.session.layout if self.session is not None else None
        if self.image_rows > 0 and self.image_cols > 0 and self.image_rows * self.image_cols == imglen:
            return self.image_rows, self.image_cols
        if layout is not None:
            for rows, cols, _source in layout.grid_candidates:
                if rows * cols == imglen:
                    return rows, cols
            if layout.rows and layout.cols and layout.rows * layout.cols == imglen:
                return layout.rows, layout.cols
        return _infer_grid(imglen, self.image_rows, self.image_cols, self.canvas_aspect)

    def _mask_from_krea_geometry(self, selection_index: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        session = self.session
        assert session is not None
        if (
            session.geometry_seq_len != seq_len
            or session.geometry_image_mask is None
            or session.geometry_x_norm is None
            or session.geometry_y_norm is None
        ):
            return None

        key = (selection_index, seq_len, str(device), str(dtype))
        cached = session.mask_cache.get(key)
        if cached is not None:
            return cached

        selection = self.stack.selections[selection_index]
        box_indices = [i for i in selection.boxes if 0 <= i < len(self.boxes)]
        if not box_indices:
            return None

        image_mask = session.geometry_image_mask.to(device=device)
        x_norm = session.geometry_x_norm.to(device=device, dtype=torch.float32)
        y_norm = session.geometry_y_norm.to(device=device, dtype=torch.float32)
        token_mask = torch.zeros((seq_len,), device=device, dtype=torch.float32)
        for box_index in box_indices:
            box_mask = _rect_position_mask(x_norm, y_norm, image_mask, self.boxes[box_index], self.seam_feather)
            token_mask = torch.maximum(token_mask, box_mask)

        full = torch.zeros((seq_len,), device=device, dtype=dtype)
        if TEXT_TOKEN_STRENGTH != 0.0:
            full[~image_mask] = float(TEXT_TOKEN_STRENGTH)
        full[image_mask] = token_mask[image_mask].to(dtype=dtype)
        full = full.view(1, seq_len, 1)
        self._record_mask_debug(
            selection_index,
            seq_len,
            "krea_position_ids",
            0,
            int(image_mask.sum().detach().cpu().item()),
            session.geometry_rows or 0,
            session.geometry_cols or 0,
            token_mask[image_mask],
        )
        session.mask_cache[key] = full
        return full

    def _record_mask_debug(
        self,
        selection_index: int,
        seq_len: int,
        source: str,
        image_start: int,
        imglen: int,
        rows: int,
        cols: int,
        token_mask: torch.Tensor,
    ) -> None:
        session = self.session
        if session is None:
            return
        if token_mask.numel() == 0:
            mask_sum = 0.0
            mask_max = 0.0
            nonzero = 0
        else:
            flat = token_mask.reshape(-1)
            mask_sum = float(flat.sum().detach().cpu())
            mask_max = float(flat.max().detach().cpu())
            nonzero = int((flat.abs() > 1e-5).sum().detach().cpu().item())
        session.mask_debug[(selection_index, seq_len)] = (
            f"source={source} start={image_start} imglen={imglen} grid={rows}x{cols} "
            f"token_sum={mask_sum:.2f} token_max={mask_max:.4f} token_nonzero={nonzero}"
        )


class SimpleRegionalApplierState(RegionalApplierState):
    """Fedor-style activation delta injection: runtime latent grid, one sequence mask for every matched layer."""

    def wrapper(self, executor, *args, **kwargs):
        model_obj = getattr(executor, "class_obj", None)
        handles = []
        transformer_options = _find_transformer_options(args, kwargs)
        layout = self._simple_layout(args, kwargs, model_obj)
        self.session = RuntimeSession(layout=layout, transformer_options=transformer_options, wrapper_calls=1)
        try:
            if model_obj is None:
                return executor(*args, **kwargs)
            name_to_module = dict(model_obj.named_modules())
            for layer_name, entries in self.layer_entries.items():
                module = name_to_module.get(layer_name)
                if module is None:
                    continue
                handles.append(module.register_forward_hook(self._make_simple_hook(entries)))
            if self.debug:
                LOGGER.info("[Krea2RegionalMultiLoRA] simple mode installed %d hooks; layout=%s", len(handles), layout)
            result = executor(*args, **kwargs)
            if self.debug:
                LOGGER.info(
                    "[Krea2RegionalMultiLoRA] simple mode stats: hooks=%d applied=%d skipped=%d observed_seq_lens=%s mask_debug=%s layout=%s",
                    self.session.hook_calls,
                    self.session.applied_calls,
                    self.session.skipped_no_mask,
                    dict(sorted(self.session.observed_seq_lens.items())),
                    _format_sample(self.session.mask_debug.values(), 4),
                    self.session.layout,
                )
            return result
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.session = None

    def _simple_layout(self, args: Sequence[Any], kwargs: Dict[str, Any], model_obj: Any) -> TokenLayout:
        txtlen = None
        context = kwargs.get("context", None)
        if context is None and len(args) >= 3:
            context = args[2]
        if torch.is_tensor(context) and context.ndim == 3:
            txtlen = int(context.shape[1])

        rows = self.image_rows if self.image_rows > 0 else None
        cols = self.image_cols if self.image_cols > 0 else None
        source = ["simple"]
        if (not rows or not cols) and args and torch.is_tensor(args[0]) and args[0].ndim >= 4:
            h = int(args[0].shape[-2])
            w = int(args[0].shape[-1])
            patch_size = int(getattr(model_obj, "patch_size", 2) or 2)
            rows = max(1, h // max(1, patch_size))
            cols = max(1, w // max(1, patch_size))
            source.append(f"latent_{h}x{w}_p{patch_size}")
        if not rows or not cols:
            rows = max(1, int(round(math.sqrt(4096 / max(1e-6, self.canvas_aspect)))))
            cols = max(1, int(round(rows * self.canvas_aspect)))
            source.append("canvas_fallback")
        if txtlen is not None:
            source.append("context_txtlen")
        return TokenLayout(imglen=int(rows * cols), txtlen=txtlen, rows=int(rows), cols=int(cols), source="+".join(source))

    def _make_simple_hook(self, entries: List[LayerPatch]):
        def hook(module, inputs, output):
            session = self.session
            if session is None or not torch.is_tensor(output) or not inputs:
                return output
            session.hook_calls += 1
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2 or output.ndim < 2 or x.shape[:-1] != output.shape[:-1]:
                return output
            seq_len = int(x.shape[-2])
            session.observed_seq_lens[seq_len] = session.observed_seq_lens.get(seq_len, 0) + 1
            out = output
            compute_dtype = _compute_dtype_for(x)
            for entry in entries:
                selection = self.stack.selections[entry.selection_index]
                if not selection.enabled:
                    continue
                mask = self._simple_mask_for_selection(entry.selection_index, seq_len, x.device, compute_dtype)
                if mask is None:
                    session.skipped_no_mask += 1
                    continue
                if self.outside_strength != 0.0:
                    mask = mask + (1.0 - mask) * self.outside_strength
                if self.debug:
                    self._log_runtime_mask(entry, seq_len, mask)
                mask = _reshape_token_mask(mask, x.ndim)
                down, up = self._matrices_on_device(entry, x.device, compute_dtype)
                xin = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
                delta = F.linear(F.linear(xin, down), up)
                delta = delta * (entry.matrices.scale * entry.strength * self.base_strength)
                out = out + (delta * mask).to(dtype=out.dtype)
                session.applied_calls += 1
            return out

        return hook

    def _simple_mask_for_selection(self, selection_index: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        session = self.session
        assert session is not None
        layout = session.layout
        rows, cols, imglen = layout.rows, layout.cols, layout.imglen
        if not rows or not cols or not imglen:
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
            token_mask = torch.maximum(token_mask, _rect_token_mask(rows, cols, self.boxes[box_index], self.seam_feather))
        token_mask = token_mask.to(device=device, dtype=dtype)
        full = torch.zeros((seq_len,), device=device, dtype=dtype)
        source = "simple"
        if imglen > seq_len:
            full[:] = token_mask.mean()
            source = "simple_mean_for_short_seq"
        else:
            start = seq_len - imglen
            if layout.txtlen is not None and 0 <= int(layout.txtlen) <= seq_len - imglen:
                start = int(layout.txtlen)
                source = "simple_after_txt"
            else:
                source = "simple_trailing"
            full[start:start + imglen] = token_mask
        self._record_mask_debug(selection_index, seq_len, source, 0 if imglen > seq_len else start, min(imglen, seq_len), rows, cols, token_mask)
        full = full.view(1, seq_len, 1)
        session.mask_cache[key] = full
        return full


class DiagnosticRegionalApplierState(RegionalApplierState):
    def __init__(self, *args, diagnostic_lora_limit: int = 2, max_steps: int = 1, return_mode: str = "passthrough", **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_lora_limit = max(1, int(diagnostic_lora_limit))
        self.max_steps = max(1, int(max_steps))
        self.return_mode = return_mode
        self._calls = 0

    def wrapper(self, executor, *args, **kwargs):
        model_obj = getattr(executor, "class_obj", None)
        transformer_options = _find_transformer_options(args, kwargs)
        layout = self._infer_layout(args, kwargs)
        self._apply_runtime_latent_grid(layout, args, model_obj)
        self.session = RuntimeSession(layout=layout, transformer_options=transformer_options, wrapper_calls=1)
        try:
            base = executor(*args, **kwargs)
            if model_obj is None or not torch.is_tensor(base):
                return base
            self._calls += 1
            if self._calls > self.max_steps:
                return base
            name_to_module = dict(model_obj.named_modules())
            selected_indices = [
                i for i, s in enumerate(self.stack.selections)
                if s.enabled and i < len(self.stack.selections)
            ][: self.diagnostic_lora_limit]
            reports: List[str] = []
            return_tensor = base
            for selection_index in selected_indices:
                selection = self.stack.selections[selection_index]
                bbox_mask = self._output_mask_for_selection(selection_index, base)
                for label, force_global, scope_filter in (
                    ("global_all", True, None),
                    ("global_blocks_only", True, "regional"),
                    ("global_txtfusion_only", True, "global_conditioning"),
                    ("token_masked_all", False, None),
                ):
                    out = self._run_variant(executor, args, kwargs, name_to_module, selection_index, force_global, scope_filter)
                    if out is None:
                        reports.append(f"{selection.alias}:{label}=no_hooks")
                        continue
                    reports.append(f"{selection.alias}:{label} {_delta_stats(base, out, bbox_mask)}")
                    if self.return_mode == f"{label}_first" and return_tensor is base:
                        return_tensor = out
                if bbox_mask is not None:
                    reports.append(
                        f"{selection.alias}:output_mask shape={tuple(bbox_mask.shape)} sum={float(bbox_mask.sum().detach().cpu()):.2f} max={float(bbox_mask.max().detach().cpu()):.4f}"
                    )
            LOGGER.info(
                "[Krea2RegionalMultiLoRA] diagnostic step=%d output_shape=%s layout=%s observed_seq_lens=%s reports=%s",
                self._calls,
                tuple(base.shape),
                self.session.layout,
                dict(sorted(self.session.observed_seq_lens.items())),
                " | ".join(reports),
            )
            return return_tensor
        finally:
            self.session = None

    def _run_variant(
        self,
        executor,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        name_to_module: Dict[str, Any],
        selection_index: int,
        force_global: bool,
        scope_filter: Optional[str],
    ) -> Optional[torch.Tensor]:
        handles = self._install_selection_hooks(name_to_module, selection_index, force_global=force_global, scope_filter=scope_filter)
        if not handles:
            return None
        try:
            out = executor(*args, **kwargs)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        return out if torch.is_tensor(out) else None


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


def _find_transformer_options(args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    for value in reversed(args):
        if isinstance(value, dict) and (
            "patches" in value
            or "wrappers" in value
            or "callbacks" in value
            or "transformer_index" in value
            or "block_index" in value
            or "img_slice" in value
        ):
            return value
    for value in reversed(args):
        if isinstance(value, dict):
            return value
    return {}


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


def _rect_position_mask(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    image_mask: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    feather: float,
) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    f = max(1e-4, float(feather))
    left = torch.sigmoid((x_norm - float(x0)) / f)
    right = torch.sigmoid((float(x1) - x_norm) / f)
    top = torch.sigmoid((y_norm - float(y0)) / f)
    bottom = torch.sigmoid((float(y1) - y_norm) / f)
    out = (left * right * top * bottom).clamp(0.0, 1.0)
    return out * image_mask.to(dtype=out.dtype)


def _reshape_token_mask(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    if target_ndim <= 2:
        return mask.view(mask.shape[-2], 1)
    return mask.view(*([1] * (target_ndim - 2)), mask.shape[-2], 1)


def _spatial_broadcast_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    if tensor.ndim >= 4:
        return (1, 1, int(tensor.shape[-2]), int(tensor.shape[-1]))
    if tensor.ndim >= 2:
        return (1,) * (tensor.ndim - 1) + (1,)
    return (1,)


def _delta_stats(base: torch.Tensor, other: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    if not torch.is_tensor(base) or not torch.is_tensor(other) or base.shape != other.shape:
        return f"shape_mismatch base={getattr(base, 'shape', None)} other={getattr(other, 'shape', None)}"
    delta = (other - base).detach().abs().to(dtype=torch.float32)
    parts = [
        f"mean={float(delta.mean().cpu()):.6f}",
        f"max={float(delta.max().cpu()):.6f}",
    ]
    if mask is not None and torch.is_tensor(mask):
        m = mask.detach().to(device=delta.device, dtype=torch.float32)
        while m.ndim < delta.ndim:
            m = m.unsqueeze(1)
        inside_w = m.clamp(0.0, 1.0)
        outside_w = 1.0 - inside_w
        inside_full = torch.ones_like(delta) * inside_w
        outside_full = torch.ones_like(delta) * outside_w
        inside_den = inside_full.sum().clamp_min(1.0)
        outside_den = outside_full.sum().clamp_min(1.0)
        parts.extend(
            [
                f"inside_mean={float((delta * inside_w).sum().cpu() / inside_den.cpu()):.6f}",
                f"outside_mean={float((delta * outside_w).sum().cpu() / outside_den.cpu()):.6f}",
                f"mask_sum={float(inside_w.sum().cpu()):.2f}",
                f"mask_max={float(inside_w.max().cpu()):.4f}",
            ]
        )
    return " ".join(parts)


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


def _sample_list(values: Iterable[Any], limit: int = 6) -> List[str]:
    out = []
    for value in values:
        out.append(str(value))
        if len(out) >= limit:
            break
    return out


def _format_sample(values: Iterable[Any], limit: int = 6) -> str:
    sample = _sample_list(values, limit)
    return "[" + "; ".join(sample) + "]" if sample else "[]"


def _load_lora_state_dicts(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load and convert a LoRA exactly like ComfyUI's standard loader path."""
    if comfy_utils is not None:
        raw = comfy_utils.load_torch_file(path, safe_load=True)
    else:
        if _SAFETENSORS_IMPORT_ERROR is not None:
            raise RuntimeError(f"safetensors import failed and comfy.utils is unavailable: {_SAFETENSORS_IMPORT_ERROR}")
        raw = safetensors.torch.load_file(path, device="cpu")

    converted = raw
    if comfy_lora_convert is not None:
        converted = comfy_lora_convert.convert_lora(raw)
    return raw, converted


def _target_weight_key(target: Any) -> Optional[str]:
    # ComfyUI key maps can point to either a weight key string or to a tuple
    # describing a slice of a fused weight. The regional hook currently supports
    # only full Linear weights; fused slices are counted as unsupported.
    if isinstance(target, str):
        return target
    if isinstance(target, tuple) and target and isinstance(target[0], str):
        return target[0]
    return None


def _target_is_full_weight(target: Any) -> bool:
    return isinstance(target, str)


def _target_display(target: Any) -> str:
    if isinstance(target, tuple):
        return repr(target)
    return str(target)


def _module_name_from_weight_key(weight_key: str, name_to_module: Dict[str, Any], normalized_name_map: Dict[str, Tuple[str, Any]]) -> Optional[str]:
    if not weight_key.endswith(".weight"):
        return None
    candidates = [weight_key[:-len(".weight")]]
    prefixes = (
        "diffusion_model.",
        "model.diffusion_model.",
        "model.",
        "base_model.model.",
    )
    for c in list(candidates):
        for prefix in prefixes:
            if c.startswith(prefix):
                candidates.append(c[len(prefix):])
    for c in candidates:
        if c in name_to_module:
            return c
    for c in candidates:
        mapped = normalized_name_map.get(_normalize_key(c))
        if mapped is not None:
            return mapped[0]
    return None


def _native_key_map_for_model(model_obj) -> Dict[str, Any]:
    if comfy_lora is None:
        return {}
    try:
        # This is the same UNet/model-side key mapping used by ComfyUI's normal
        # Load LoRA path via comfy.sd.load_lora_for_models.
        return comfy_lora.model_lora_keys_unet(model_obj, {})
    except Exception as e:
        LOGGER.warning("[Krea2RegionalMultiLoRA] ComfyUI model_lora_keys_unet failed: %s", e)
        return {}


def _native_loaded_patches(sd: Dict[str, torch.Tensor], native_key_map: Dict[str, Any]) -> Tuple[Dict[Any, Any], Optional[str]]:
    if comfy_lora is None or not native_key_map:
        return {}, "comfy_lora_unavailable_or_empty_key_map"
    try:
        return comfy_lora.load_lora(sd, native_key_map, log_missing=False), None
    except TypeError as e:
        try:
            return comfy_lora.load_lora(sd, native_key_map), None
        except Exception as fallback_e:
            return {}, f"{type(fallback_e).__name__}: {fallback_e}"
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"


def _spatial_matrices_from_patch(patch: Any, source_key: str) -> Tuple[Optional[LoraMatrices], Optional[str]]:
    """Return ordinary low-rank Linear LoRA matrices from a ComfyUI loaded patch."""
    if isinstance(patch, tuple) and len(patch) >= 2 and patch[0] == "lora":
        weights = patch[1]
        name = "lora"
    else:
        name = getattr(patch, "name", None)
        weights = getattr(patch, "weights", None)

    if name != "lora" or not isinstance(weights, tuple) or len(weights) < 6:
        return None, f"adapter={type(patch).__name__}"

    up, down, alpha, mid, dora_scale, reshape = weights[:6]
    if dora_scale is not None:
        return None, "dora"
    if reshape is not None:
        return None, f"reshape={reshape}"
    if not torch.is_tensor(up) or not torch.is_tensor(down):
        return None, "non_tensor"
    if up.ndim != 2 or down.ndim != 2:
        return None, f"non_linear up={tuple(up.shape)} down={tuple(down.shape)}"
    if mid is not None:
        if not torch.is_tensor(mid) or mid.ndim != 2:
            return None, f"unsupported_mid={getattr(mid, 'shape', None)}"
        return None, "mid_adapter"

    rank = int(down.shape[0])
    if alpha is None:
        scale = 1.0
    else:
        try:
            scale = float(alpha) / max(1, rank)
        except Exception:
            scale = 1.0
    return LoraMatrices(down=down.contiguous(), up=up.contiguous(), scale=scale, source_key=source_key), None

def _iter_named_linears(model_obj, apply_to: str) -> Iterable[Tuple[str, Any]]:
    for name, module in model_obj.named_modules():
        if not hasattr(module, "weight"):
            continue
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        lname = name.lower()
        if apply_to == "krea_blocks_only":
            is_global_conditioning = any(fragment in lname for fragment in GLOBAL_CONDITIONING_NAME_FRAGMENTS)
            if not is_global_conditioning and any(fragment in lname for fragment in DEFAULT_EXCLUDED_NAME_FRAGMENTS):
                continue
            if not is_global_conditioning and not any(token in lname for token in ("double_blocks", "single_blocks", "blocks.", "joint_blocks", "img_mlp", "img_attn", "attn", "mlp")):
                continue
        yield name, module


def _mask_scope_for_layer(layer_name: str) -> str:
    lname = layer_name.lower()
    if any(fragment in lname for fragment in GLOBAL_CONDITIONING_NAME_FRAGMENTS):
        return "global_conditioning"
    return "regional"


def _build_layer_entries(
    key_map_model_obj: Any,
    hook_model_obj: Any,
    stack: LoraStack,
    apply_to: str,
) -> Tuple[Dict[str, List[LayerPatch]], List[str]]:
    all_entries: Dict[str, List[LayerPatch]] = {}
    report: List[str] = []
    all_linear_modules = list(_iter_named_linears(hook_model_obj, "all_matched_linears"))
    all_name_to_module = {name: module for name, module in all_linear_modules}
    all_normalized_name_map = {_normalize_key(name): (name, module) for name, module in all_linear_modules}
    model_modules = list(_iter_named_linears(hook_model_obj, apply_to))
    name_to_module = {name: module for name, module in model_modules}
    normalized_name_map = {_normalize_key(name): (name, module) for name, module in model_modules}
    native_key_map = _native_key_map_for_model(key_map_model_obj)
    report.append(
        "ComfyUI native key map: "
        f"model_type={type(key_map_model_obj).__name__} hook_type={type(hook_model_obj).__name__} "
        f"hook_linears={len(model_modules)} length={len(native_key_map)} "
        f"sample={_format_sample([f'{k} -> {_target_display(v)}' for k, v in native_key_map.items()], 5)}"
    )

    for sel_idx, selection in enumerate(stack.selections):
        if not selection.enabled:
            report.append(f"[{sel_idx + 1}] {selection.alias}: disabled")
            continue
        path = _resolve_lora_path(selection.lora)
        if not path:
            report.append(f"[{sel_idx + 1}] {selection.alias}: missing LoRA '{selection.lora}'")
            continue

        raw_sd, sd = _load_lora_state_dicts(path)
        loaded, load_error = _native_loaded_patches(sd, native_key_map)
        tensor_count = sum(1 for v in raw_sd.values() if torch.is_tensor(v))
        parsed_pairs = 0
        matched = 0
        shape_mismatch = 0
        unsupported_targets = 0
        unsupported_reasons: Dict[str, int] = {}
        unsupported_samples: List[str] = []
        shape_samples: List[str] = []
        missing_samples: List[str] = []
        excluded_samples: List[str] = []
        missing_modules = 0
        excluded_by_apply_to = 0
        global_conditioning_layers = 0

        for target, patch in loaded.items():
            matrices, reason = _spatial_matrices_from_patch(patch, _target_display(target))
            if matrices is None:
                unsupported_targets += 1
                unsupported_reasons[reason or "unknown"] = unsupported_reasons.get(reason or "unknown", 0) + 1
                if len(unsupported_samples) < 6:
                    unsupported_samples.append(f"{_target_display(target)} ({reason or 'unknown'})")
                continue
            parsed_pairs += 1
            if not _target_is_full_weight(target):
                unsupported_targets += 1
                unsupported_reasons["fused_or_sliced_target"] = unsupported_reasons.get("fused_or_sliced_target", 0) + 1
                if len(unsupported_samples) < 6:
                    unsupported_samples.append(f"{_target_display(target)} (fused_or_sliced_target)")
                continue
            weight_key = _target_weight_key(target)
            if weight_key is None:
                unsupported_targets += 1
                unsupported_reasons["unknown_target"] = unsupported_reasons.get("unknown_target", 0) + 1
                if len(unsupported_samples) < 6:
                    unsupported_samples.append(f"{_target_display(target)} (unknown_target)")
                continue
            layer_name = _module_name_from_weight_key(weight_key, name_to_module, normalized_name_map)
            if layer_name is None:
                excluded_layer_name = _module_name_from_weight_key(weight_key, all_name_to_module, all_normalized_name_map)
                if excluded_layer_name is not None:
                    excluded_by_apply_to += 1
                    if len(excluded_samples) < 6:
                        excluded_samples.append(weight_key)
                    continue
                missing_modules += 1
                if len(missing_samples) < 6:
                    missing_samples.append(weight_key)
                continue
            module = name_to_module[layer_name]
            down, up = matrices.down, matrices.up
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight) or weight.ndim != 2:
                shape_mismatch += 1
                if len(shape_samples) < 6:
                    shape_samples.append(f"{weight_key}: target_weight={getattr(weight, 'shape', None)} down={tuple(down.shape)} up={tuple(up.shape)}")
                continue
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            if int(down.shape[1]) != in_features or int(up.shape[0]) != out_features or int(up.shape[1]) != int(down.shape[0]):
                shape_mismatch += 1
                if len(shape_samples) < 6:
                    shape_samples.append(f"{weight_key}: target_weight={tuple(weight.shape)} down={tuple(down.shape)} up={tuple(up.shape)}")
                continue
            all_entries.setdefault(layer_name, []).append(
                LayerPatch(
                    selection_index=sel_idx,
                    layer_name=layer_name,
                    lora_key=weight_key,
                    strength=selection.strength,
                    matrices=matrices,
                    mask_scope=_mask_scope_for_layer(layer_name),
                )
            )
            if _mask_scope_for_layer(layer_name) == "global_conditioning":
                global_conditioning_layers += 1
            matched += 1

        unsupported_summary = ", ".join(f"{k}:{v}" for k, v in sorted(unsupported_reasons.items())) or "-"
        report.append(
            f"[{sel_idx + 1}] {selection.alias}: file={selection.lora} boxes={','.join(str(i + 1) for i in selection.boxes) or '-'} "
            f"path={path} tensors={tensor_count} sample_lora_keys={_format_sample(raw_sd.keys(), 5)} "
            f"converted_sample_keys={_format_sample(sd.keys(), 5)} native_loaded={len(loaded)} "
            f"native_load_error={load_error or '-'} native_patch_sample={_format_sample(loaded.keys(), 5)} "
            f"parsed_pairs={parsed_pairs} matched_layers={matched} "
            f"shape_mismatch={shape_mismatch} unsupported_targets={unsupported_targets} unsupported_reasons={unsupported_summary} "
            f"global_conditioning_layers={global_conditioning_layers} excluded_by_apply_to={excluded_by_apply_to} missing_modules={missing_modules} "
            f"unsupported_samples={_format_sample(unsupported_samples, 6)} excluded_samples={_format_sample(excluded_samples, 6)} "
            f"shape_samples={_format_sample(shape_samples, 6)} missing_samples={_format_sample(missing_samples, 6)}"
        )
        if len(loaded) > 0 and matched == 0:
            report.append(
                f"[{sel_idx + 1}] {selection.alias}: ERROR no spatially usable layers matched. "
                f"Loaded patch targets sample={_format_sample(loaded.keys(), 12)}"
            )
    return all_entries, report


# -------------------- report / preview --------------------


def _format_assignment_report(stack: LoraStack, boxes: List[Tuple[float, float, float, float]], include_box_coords: bool = True) -> str:
    lines = [
        f"Krea2 Regional LoRA node version: {NODE_VERSION}",
        f"Mask semantics: text_token_strength={TEXT_TOKEN_STRENGTH:.3f}, image_tokens=bbox_mask, padding_tokens=outside_strength",
        "Mask scope: txtfusion/textfusion LoRA layers apply globally as conditioning; image/block layers apply regionally",
        "Execution mode: output-delta blend (baseline denoiser pass plus one global LoRA pass per enabled LoRA, blended by bbox on output latent)",
        "Mask layout: Krea _backbone position_ids/indicator first; fallback chooses image span from live seq_len/text length before latent-grid guesses",
        "Runtime debug: logs observed_seq_lens plus per-LoRA mask source/start/imglen/grid/sum/max/nonzero when debug_logging=True",
        f"LoRA count: {len(stack.selections)}",
        f"BBox count: {len(boxes)}",
        "Assignments:",
    ]
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
        hook_model_obj = model_out.get_model_object("diffusion_model")
        key_map_model_obj = getattr(model_out, "model", hook_model_obj)
        layer_entries, lines = _build_layer_entries(key_map_model_obj, hook_model_obj, stack, apply_to)
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


class Krea2RegionalLoRAApplySimple:
    @classmethod
    def INPUT_TYPES(cls):
        return Krea2RegionalLoRAApply.INPUT_TYPES()

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
        hook_model_obj = model_out.get_model_object("diffusion_model")
        key_map_model_obj = getattr(model_out, "model", hook_model_obj)
        layer_entries, lines = _build_layer_entries(key_map_model_obj, hook_model_obj, stack, apply_to)
        aspect = float(canvas_width) / max(1.0, float(canvas_height))
        state = SimpleRegionalApplierState(
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
            WRAPPER_KEY + "_simple",
            state.wrapper,
        )
        report = (
            _format_assignment_report(stack, boxes)
            + "\nExecution mode: simple activation-delta injection, modeled after the external Fedor node. "
            + "Uses runtime latent_grid/patch_size and applies the same full-sequence regional mask to txtfusion and image blocks; short text-only sequences receive mask mean, not full global strength."
            + "\n\nPatch summary:\n"
            + "\n".join(lines)
        )
        return (model_out, report)


class Krea2RegionalLoRADiagnostics:
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
                "base_strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "apply_to": (["krea_blocks_only", "all_matched_linears"], {"default": "krea_blocks_only"}),
                "diagnostic_lora_limit": ("INT", {"default": 2, "min": 1, "max": 8}),
                "max_steps": ("INT", {"default": 1, "min": 1, "max": 32}),
                "return_mode": (["passthrough", "global_all_first", "global_blocks_only_first", "global_txtfusion_only_first", "token_masked_all_first"], {"default": "passthrough"}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "kj_bboxes": ("BBOX",),
                "ideogram_prompt_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "diagnose"
    CATEGORY = "Krea2/Regional LoRA"

    def diagnose(
        self,
        model,
        lora_stack,
        canvas_width,
        canvas_height,
        bbox_list_format,
        seam_feather,
        base_strength,
        apply_to,
        diagnostic_lora_limit,
        max_steps,
        return_mode,
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
        hook_model_obj = model_out.get_model_object("diffusion_model")
        key_map_model_obj = getattr(model_out, "model", hook_model_obj)
        layer_entries, lines = _build_layer_entries(key_map_model_obj, hook_model_obj, stack, apply_to)
        aspect = float(canvas_width) / max(1.0, float(canvas_height))
        state = DiagnosticRegionalApplierState(
            stack=stack,
            boxes=boxes,
            layer_entries=layer_entries,
            seam_feather=seam_feather,
            outside_strength=0.0,
            base_strength=base_strength,
            token_offset_mode="auto_txt_img_pad_safe",
            manual_image_start=0,
            image_rows=0,
            image_cols=0,
            debug=True,
            canvas_aspect=aspect,
            diagnostic_lora_limit=diagnostic_lora_limit,
            max_steps=max_steps,
            return_mode=return_mode,
        )
        model_out.add_wrapper_with_key(
            patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY + "_diagnostics",
            state.wrapper,
        )
        report = (
            _format_assignment_report(stack, boxes)
            + "\nDiagnostic mode: pass-through by default. During sampling, logs baseline-vs-LoRA output delta for global_all, global_blocks_only, global_txtfusion_only, and token_masked_all. "
            + "Use the console log lines beginning '[Krea2RegionalMultiLoRA] diagnostic'."
            + f"\nSettings: lora_limit={int(diagnostic_lora_limit)} max_steps={int(max_steps)} return_mode={return_mode}"
            + "\n\nPatch summary:\n"
            + "\n".join(lines)
        )
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
    "Krea2RegionalLoRAApplySimple": Krea2RegionalLoRAApplySimple,
    "Krea2RegionalLoRADiagnostics": Krea2RegionalLoRADiagnostics,
    "Krea2RegionalLoRAPreview": Krea2RegionalLoRAPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2MultiLoRALoader": "Krea2 Multi LoRA Loader",
    "Krea2RegionalLoRAApply": "Krea2 Regional LoRA Apply",
    "Krea2RegionalLoRAApplySimple": "Krea2 Regional LoRA Apply Simple",
    "Krea2RegionalLoRADiagnostics": "Krea2 Regional LoRA Diagnostics",
    "Krea2RegionalLoRAPreview": "Krea2 Regional LoRA Preview",
}
