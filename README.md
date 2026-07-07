# ComfyUI Krea2 Regional Multi-LoRA

Standalone ComfyUI custom nodes for coordinating multiple Krea2 character LoRAs with bbox-driven regional crop/detail/composite workflows.

## What this version does

- standalone implementation
- separate **Multi LoRA Loader** node
- ordered LoRA list with alias, file, strength, and assigned box indices
- one LoRA can target **multiple independent boxes**
- preview node overlays **box numbers and LoRA aliases** so it is easy to see what each drawn box maps to
- provides crop/composite nodes for the recommended isolated per-character LoRA workflow
- includes a bbox-FreeFuse experimental apply node that combines bbox masks with FreeFuse-style token routing
- keeps the direct regional LoRA apply/debug nodes for experimentation and diagnostics

## Nodes

### 1. Krea2 Multi LoRA Loader

This is where the user selects as many LoRAs as desired.

Per LoRA row:

- `enabled`
- `alias`
- `file`
- `strength`
- `boxes`
- `color`

`boxes` uses **1-based box numbers** and supports forms like:

- `1`
- `1,3,4`
- `2-5`
- `1,3-5`

That means one LoRA can be applied to multiple separately drawn boxes.

### 2. Krea2 LoRA Stack Row Model Loader

Loads one selected LoRA row onto a model using ComfyUI's native LoRA loader path. Use this for each isolated crop/detail pass.

Inputs:

- `model`: base Krea2 model
- `lora_stack`: output of **Krea2 Multi LoRA Loader**
- `row_index`: 1-based LoRA row number
- `strength_mode`: use stack strength, override it, or multiply it

### 3. Krea2 Regional LoRA Crop Extract

Extracts the crop and mask for one LoRA row from the bbox assignments.

Inputs:

- `image`: base generated image
- `lora_stack`: output of **Krea2 Multi LoRA Loader**
- `row_index`: 1-based LoRA row number
- `bboxes`: modern `BOUNDING_BOX` input
- `kj_bboxes`: legacy `BBOX` input
- `refine_mask`: optional SAM/person mask to intersect or union with the bbox mask

Outputs:

- `crop_image`: feed this into your crop img2img/detail/inpaint pass
- `crop_mask`: use this as the editable mask
- `crop_info`: metadata for paste-back
- alias, LoRA name, strength, and a report

### 4. Krea2 Regional LoRA Crop Composite

Pastes an edited crop back into the base image using the crop metadata and mask.

Inputs:

- `base_image`: the current full image
- `edited_crop`: decoded result from the crop detail pass
- `crop_info`: output from **Crop Extract**
- `blend_mask`: optional override mask

### 5. Krea2 Regional LoRA Apply

Inputs:

- `model`: Krea2 model from `UNETLoader`
- `lora_stack`: output of **Krea2 Multi LoRA Loader**
- `bboxes`: modern `BOUNDING_BOX` input
- `kj_bboxes`: legacy `BBOX` input
- `ideogram_prompt_json`: fallback prompt JSON input

Experimental direct model-wrapper approach. It uses the ordered LoRA stack plus external boxes to build a masked regional application. Krea2 character LoRAs may still bleed because Krea's transformer mixes text/image information globally.

### 6. Krea2 Regional LoRA Apply BBox FreeFuse

Experimental one-pass approach inspired by FreeFuse. It keeps the same bbox assignments, but adds two mechanisms:

- masked LoRA deltas on Krea image/block layers
- FreeFuse-style attention bias so each subject token is encouraged to attend to its assigned bbox region and discouraged from the other LoRA regions

Inputs:

- `model`: Krea2 model from `UNETLoader`
- `clip`: the same Krea2 CLIP/text encoder used for the prompt
- `positive_prompt`: the exact positive prompt string
- `lora_stack`: output of **Krea2 Multi LoRA Loader**
- `bboxes`: modern `BOUNDING_BOX` input
- `kj_bboxes`: legacy `BBOX` input
- `concepts_json`: optional alias-to-prompt-text map

Important prompt rule: each enabled row's alias, or its `concepts_json` override, must appear in the positive prompt so the node can find the subject tokens to route.

Recommended first test settings:

- `outside_strength=0.0`
- `text_token_lora_strength=1.0`
- `bias_blocks=last_half`
- `bias_scale=5.0`
- `positive_bias_scale=1.0`
- `enable_lora_masking=true`
- `enable_attention_bias=true`

This node skips FreeFuse's automatic similarity-map phase because your external bboxes already provide the spatial masks.

### 7. Krea2 Regional LoRA Preview

Draws the boxes and labels them like:

- `box 1`
- `[1] character_a`
- `[2] character_b`

This helps the user identify which LoRA entry corresponds to which drawn box assignment.

## Typical workflow

```text
bbox-drawing node -> Krea2 Regional LoRA Preview
Krea2 Multi LoRA Loader -> Krea2 Regional LoRA Preview

Base Krea generation with bbox prompt, no character LoRAs -> IMAGE

For row 1:
IMAGE + bboxes + lora_stack -> Krea2 Regional LoRA Crop Extract
UNETLoader Krea2 + lora_stack -> Krea2 LoRA Stack Row Model Loader(row_index=1)
crop_image + crop_mask + row-1-LoRA model -> your img2img/detail/inpaint pass
edited_crop + crop_info + base image -> Krea2 Regional LoRA Crop Composite

Repeat for row 2, row 3, etc.
```

## Example loader configuration

```json
[
  {
    "enabled": true,
    "alias": "alice",
    "lora": "alice_krea2.safetensors",
    "strength": 1.0,
    "boxes": "1,3",
    "color": "#ff5f57"
  },
  {
    "enabled": true,
    "alias": "bob",
    "lora": "bob_krea2.safetensors",
    "strength": 0.95,
    "boxes": "2",
    "color": "#5fb3ff"
  }
]
```

In that example:

- Alice applies to **box 1 and box 3**
- Bob applies to **box 2**

## Notes

- Use `bbox_list_format=xywh` for legacy KJNodes `BBOX`
- Use `BOUNDING_BOX` directly when possible
- For the crop workflow, start with `pad_pixels=96`, `pad_percent=0.15`, `grow_pixels=16`, and `blur_pixels=12`
- If you have a SAM/person mask, try `refine_mask_mode=intersect_refine`
- The older direct regional apply node is useful for diagnostics, but the crop/composite path is the recommended route for reducing identity bleed
- The bbox-FreeFuse node is the most promising one-pass experiment. Make sure aliases or concept overrides appear in the prompt exactly.

## Installation

Copy the folder into:

```bash
ComfyUI/custom_nodes/ComfyUI-Krea2-Regional-LoRA-Masks
```

Restart ComfyUI.
