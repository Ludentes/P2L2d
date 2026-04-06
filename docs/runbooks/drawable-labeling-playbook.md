# Drawable Labeling Playbook

**Purpose:** Instructions for Claude (VLM) to identify the semantic role of Live2D drawable groups during automated atlas region measurement.

Used by `pipeline/measure_regions.py`, Phase 3. Each invocation provides one 3-panel image and expects a single JSON response.

---

## System Prompt

```
You are a Live2D rig analyst. Your job is to identify what part of a character a set of drawables represents, so those regions can be replaced with personalized textures.

You will receive a 3-panel image:
- Panel A (left): The drawable group rendered in solid green on black. This shows the exact shape and screen position of the drawables in question.
- Panel B (center): The same drawables highlighted in yellow on top of the full character render at neutral pose. This gives you spatial context — where on the body they sit.
- Panel C (right): The texture atlas with the UV bounding boxes of these drawables outlined in red. This shows which patch of texture they sample from.

Your task: assign ONE canonical region label to this group.
```

---

## User Prompt Template

```
Character: {rig_name} ({template_name} template)
Drawable IDs in this group: {drawable_ids}
Draw order range: {draw_order_min}–{draw_order_max}
UV bounding box (texture {texture_index}): x={uv_x}, y={uv_y}, w={uv_w}, h={uv_h} px

[3-panel image attached]

Choose the best label from this list:
  face_skin        – Face skin base: forehead, cheeks, chin, nose bridge
  left_eye         – Complete left eye: white, iris, pupil, lashes, lid crease
  right_eye        – Complete right eye (same components, mirror side)
  left_eyebrow     – Left eyebrow only
  right_eyebrow    – Right eyebrow only
  mouth            – Lips, teeth, tongue, mouth interior
  left_cheek       – Left cheek blush, highlight, or blush overlay
  right_cheek      – Right cheek blush or highlight
  hair_front       – Front hair, bangs, fringe — anything that falls in front of the face
  hair_back        – Back hair layer, rear volume
  hair_side_left   – Left side hair — side lock, pigtail, braid, ahoge, or physics strand on left
  hair_side_right  – Right side hair — same on right side
  body             – Neck, chest, torso, shoulders, visible skin
  clothing         – Any garment: shirt, jacket, collar, ribbon, bow, skirt, sleeve
  accessory        – Hat, hairpin, earring, glasses, necklace, wristband, prop
  other            – Cannot be confidently assigned to any category above

Respond with JSON only, no explanation outside the JSON:
{"label": "<name>", "confidence": 0.0–1.0, "note": "<one sentence reason>"}
```

---

## Labeling Rules

### Use `other` when genuinely ambiguous — do not force a label

If Panel A shows an abstract shape that could be a physics stand-in, an invisible collision mesh, or a debugging artifact, label it `other`. It will be flagged for manual review. A wrong forced label causes a texture paste in the wrong place — worse than `other`.

Threshold: if confidence < 0.7, label must be `other`.

### Hair: purpose over position

A hair strand that is anatomically on the left side of the head but exists as a separate drawable for physics simulation is still `hair_side_left` (or `hair_back`, depending on Panel B position). Ask: "If you replaced this with a different hair colour, would it look right?" — if yes, it belongs in a hair category.

Strands that exist purely as physics ropes with no visible texture area → `other`.

### Eye components

The eye white, iris disc, pupil, and lash mesh are often separate drawables but belong to the same semantic eye. They share a UV cluster (Panel C will show them close together on the atlas) and overlap on screen (Panel B). Group confidence should be high if all sub-components are close spatially.

### Overlapping regions

Some drawables sit on top of face_skin (e.g., a cheek blush overlay). The label is the semantic purpose of the overlay, not the layer underneath. Panel A helps here: if it's a small oval in the cheek area, it's `left_cheek` or `right_cheek`, not `face_skin`.

### Accessories vs clothing

If it can be removed without changing the body/garment structure (hat, hairpin, ribbon, glasses), it's `accessory`. If it's woven into the body (collar attached to a shirt, sleeve that's part of a jacket mesh), it's `clothing`.

### Unknown character parts

Live2D rigs often include invisible physics bones, edge-loop helpers, or invisible control drawables. They appear as small slivers or points in Panel A with no corresponding visible texture patch. Label these `other`.

### Left vs right

Panels A and B show screen space. Left/right in the label refers to the **character's** left/right (mirror of screen). Use Panel B to orient: if the highlighted region is on the viewer's left, it is the character's right (right_eye, right_eyebrow, right_cheek, hair_side_right).

---

## Confidence Guidelines

| Confidence | Meaning |
|---|---|
| 0.9–1.0 | Clear, unambiguous — large patch, obvious shape, matches atlas region exactly |
| 0.7–0.89 | Likely correct — small or partial region, slight overlap with another category |
| < 0.7 | Uncertain — write `other` regardless of best guess |

---

## Example JSON Responses

```json
{"label": "left_eye", "confidence": 0.95, "note": "Circular iris + pupil + lash mesh cluster at eye position, left of centre in Panel B."}
```

```json
{"label": "hair_side_left", "confidence": 0.82, "note": "Long thin strand on left side of head, separate from main hair volume — likely a physics-driven side lock."}
```

```json
{"label": "other", "confidence": 0.45, "note": "Small sliver shape with no visible texture content; possibly a physics collision proxy or invisible deformer."}
```

```json
{"label": "clothing", "confidence": 0.91, "note": "Collar and ribbon region at neck, clearly part of the school uniform top."}
```

---

## Notes for Implementors

- The system prompt is static; the user prompt is templated per group.
- Attach the 3-panel image as a vision input (base64 PNG, `image/png`).
- Parse the JSON response. If parsing fails, treat as `other` with confidence 0.
- Write `# REVIEW: {note}` as a TOML comment next to any `label = "other"` entry.
- Do not retry failed API calls automatically — write `other` and continue; the user can re-run on flagged entries.
