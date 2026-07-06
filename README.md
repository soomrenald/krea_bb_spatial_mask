# ComfyUI Krea2 Regional LoRA Masks

Custom ComfyUI nodes for applying multiple Krea2 character LoRAs to separate spatial regions in the same generation.

This package is based on the activation-delta masking premise used by Fedor/CliffNodes' Krea2 regional multi-LoRA node, but implements the main safety patches I would want before relying on it:

- masks the LoRA **delta**, not the prompt or attention bias
- targets Krea2's actual `text tokens -> image tokens -> padding` layout when the forward-call tensors expose it
- avoids the fragile assumption that image tokens are always the final `N` sequence tokens
- skips text-fusion/time/final layers by default
- validates LoRA matrix shapes against live Linear layers before installing hooks
- supports manual token-grid overrides when ComfyUI/Krea2 wrappers do not expose enough layout metadata
- includes a mask preview node

## Nodes

### Krea2 Regional LoRA Masks (patched)

Inputs:

- `model`: Krea2 MODEL from `UNETLoader`
- `regions_json`: list of LoRA regions
- `canvas_width`, `canvas_height`: coordinate reference for pixel-space boxes
- `bboxes`: optional `BOUNDING_BOX` input from a box builder node
- `split_mode`: fallback region splitting if no bbox is supplied
- `seam_feather`: soft edge width as a fraction of the token grid
- `outside_strength`: deliberate outside-region leak; keep at `0.0` for identity separation
- `base_strength`: global multiplier over all region strengths
- `token_offset_mode`:
  - `auto_txt_img_pad_safe`: preferred; uses Krea2 text/image layout when inferable
  - `manual`: uses `manual_image_start`
  - `legacy_trailing`: old fallback; assumes the image tokens are the last N tokens
- `image_rows`, `image_cols`: manual image token grid override. Leave `0/0` unless mask placement is wrong.
- `apply_to`:
  - `krea_blocks_only`: recommended default
  - `all_matched_linears`: experimental, more invasive

Output:

- patched `MODEL`
- text `report`

### Krea2 Region Mask Preview

Builds the same rectangular masks at preview resolution so you can check region placement before sampling.

## `regions_json` schema

```json
[
  {
    "name": "left_character",
    "lora": "alice_krea2.safetensors",
    "strength": 1.0,
    "enabled": true,
    "bbox": {"x": 0.05, "y": 0.05, "w": 0.40, "h": 0.85}
  },
  {
    "name": "right_character",
    "lora": "bob_krea2.safetensors",
    "strength": 1.0,
    "enabled": true,
    "bbox": {"x": 0.55, "y": 0.05, "w": 0.40, "h": 0.85}
  }
]
```

Coordinates may be normalized `0..1` or pixels relative to `canvas_width` / `canvas_height`.

If a `BOUNDING_BOX` input is connected, external boxes override JSON boxes by row order.

## Recommended workflow

```text
UNETLoader Krea2 -> Krea2 Regional LoRA Masks (patched) -> KSampler
CLIPLoader krea2 -> prompt conditioning as usual
VAELoader qwen_image_vae -> decode as usual
```

Recommended starting values:

```text
seam_feather: 0.04-0.08
outside_strength: 0.0
base_strength: 1.0
per-region strength: 0.8-1.1
token_offset_mode: auto_txt_img_pad_safe
apply_to: krea_blocks_only
```

Use boxes around face and upper torso. Avoid large overlapping boxes unless you intentionally want blending.

## Installation

Copy this folder into:

```bash
ComfyUI/custom_nodes/ComfyUI-Krea2-Regional-LoRA-Masks
```

Restart ComfyUI.

Requirements are already present in normal ComfyUI installs:

- `torch`
- `safetensors`
- recent ComfyUI with `comfy.patcher_extension` and `ModelPatcher.add_wrapper_with_key`

## Important limitations

This is not guaranteed to work with every Krea2 ComfyUI loader fork. The safest path depends on whether the Krea2 forward wrapper exposes enough information to infer:

```text
txtlen, imglen, padding
```

If generations look like the wrong region is being affected:

1. Turn `debug_logging` on.
2. Set `image_rows` and `image_cols` manually if the inferred token grid is wrong.
3. Use `manual` token offset mode and set `manual_image_start` to the text-token count if auto inference fails.
4. Use `legacy_trailing` only as a compatibility fallback.

## Why this should reduce character bleed

A normal LoRA modifies every matching layer globally. This node adds each LoRA's activation delta only where the spatial token mask is nonzero:

```python
output = base_output + mask * ((x @ down.T) @ up.T) * strength
```

Outside the mask, that LoRA contributes zero to the live activations.

