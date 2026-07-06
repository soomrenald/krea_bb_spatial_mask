# ComfyUI Krea2 Regional Multi-LoRA

Standalone ComfyUI custom nodes for applying multiple Krea2 character LoRAs to distinct spatial regions with a separate ordered loader node.

## What this version does

- standalone implementation
- separate **Multi LoRA Loader** node
- ordered LoRA list with alias, file, strength, and assigned box indices
- one LoRA can target **multiple independent boxes**
- preview node overlays **box numbers and LoRA aliases** so it is easy to see what each drawn box maps to
- applies each LoRA by masking its activation delta on image tokens

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

### 2. Krea2 Regional LoRA Apply

Inputs:

- `model`: Krea2 model from `UNETLoader`
- `lora_stack`: output of **Krea2 Multi LoRA Loader**
- `bboxes`: modern `BOUNDING_BOX` input
- `kj_bboxes`: legacy `BBOX` input
- `ideogram_prompt_json`: fallback prompt JSON input

The node uses the ordered LoRA stack plus external boxes to build the masked regional application.

### 3. Krea2 Regional LoRA Preview

Draws the boxes and labels them like:

- `box 1`
- `[1] character_a`
- `[2] character_b`

This helps the user identify which LoRA entry corresponds to which drawn box assignment.

## Typical workflow

```text
bbox-drawing node -> Krea2 Regional LoRA Preview
Krea2 Multi LoRA Loader -> Krea2 Regional LoRA Preview

UNETLoader Krea2 -> Krea2 Regional LoRA Apply -> KSampler
Krea2 Multi LoRA Loader -> Krea2 Regional LoRA Apply
bbox-drawing node -> Krea2 Regional LoRA Apply
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
- Keep `outside_strength=0.0` for best identity separation
- Start `seam_feather` around `0.04` to `0.08`

## Installation

Copy the folder into:

```bash
ComfyUI/custom_nodes/ComfyUI-Krea2-Regional-LoRA-Masks
```

Restart ComfyUI.
