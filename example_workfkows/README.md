# Example workflow notes

A complete workflow JSON is loader-version-sensitive, so this package does not ship a rigid graph that may break across Krea2 ComfyUI builds.

Minimal graph:

```text
UNETLoader(krea2_turbo_bf16 or krea2_bf16)
  -> Krea2 Regional LoRA Masks (patched)
  -> KSampler.model

CLIPLoader(type=krea2)
  -> normal positive conditioning

VAELoader(qwen_image_vae)
  -> VAE Decode
```

Set `regions_json` rows to match your character boxes. Wire a `BOUNDING_BOX` builder into `bboxes` if you use one; otherwise keep boxes directly in `regions_json`.
