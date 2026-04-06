# Research: Sprite Atlas Design and Texture Generation for Live2D Rigs

**Date:** 2026-04-06
**Sources:** 14 sources — arXiv papers, GitHub repositories, official Live2D documentation, local rig inspection

---

## Executive Summary

A Live2D texture atlas is "correct" for programmatic texture generation if and only if each swappable semantic concept occupies a predictable, isolated region on a dedicated texture sheet — meaning no two semantically distinct body regions (e.g. face skin and clothing) share UV space on the same sheet. The HaiMeng production rig, which both CartoonAlive [1] and Textoon [2] use, achieves this with 9 × 4096 px sheets each carrying exactly one garment/region category; texture injection is then reduced to pixel-painting at a pre-authored coordinate rectangle, requiring no UV unwrapping at runtime. Hiyori, by contrast, packs everything onto 2 × 2048 px sheets with no semantic partitioning enforced by the atlas layout — a direct consequence of Cubism Editor's auto-layout being packing-efficiency-first rather than semantics-first. For atlas designs like Hiyori's, full-sheet replacement and diffusion inpainting remain viable; per-region paste is achievable if a UV crop table is pre-authored; hue shift is the fallback for mixed-semantic UV regions where skin and clothing cannot be isolated. The correct design principle for any new rig in this pipeline is: one swappable semantic category per texture sheet, semantic English drawable names, and a machine-readable UV crop table committed alongside the rig.

---

## Key Findings

### Live2D Atlas Structure: What "Correct" Looks Like

The Live2D Cubism Editor does not enforce semantic grouping during atlas generation. Its auto-layout operates as a bin-packing optimizer: it places all ArtMesh UVs onto as few texture sheets as possible at as high a scale as possible, using rotation if allowed and a configurable margin to prevent bleeding [3]. As of Cubism 5.2, the editor added a filter to the texture atlas dialog that lets the artist narrow the displayed drawable list by part group [4], but this is a UI display filter only — it does not route drawables to separate sheets. There is no Cubism Editor feature that says "all drawables in PartEye must land on texture_01 and nothing else."

The only mechanism to achieve semantic sheet separation is fully manual: the rigger must move each ArtMesh to the desired texture tab by selecting it, switching tabs, and confirming placement [3]. In practice this means semantic sheet partitioning is a rigger discipline decision, not an editor default. It requires deliberate upfront atlas design before UV packing begins. The consequence is that atlas quality for programmatic texture generation varies entirely by how carefully the rigger planned the layout — rigs built without this intent (the majority of publicly available VTS models) have mixed-semantic sheets.

The HaiMeng rig represents the deliberate approach taken to its logical conclusion: 9 dedicated sheets, each carrying exactly one swappable category [5]. Within each sheet, each component occupies a specific pixel rectangle documented in `model_configuration.json` as explicit `(x, y, w, h)` crop coordinates. The rig's UV mesh in Cubism Editor was authored to map each ArtMesh to exactly that rectangle, so the UV layout and the crop table are in permanent correspondence.

### What Hiyori Gets Wrong (and Why It Happens)

Hiyori's atlas layout is a typical example of what emerges from Cubism Editor's auto-layout without deliberate semantic planning. All 133 drawables are packed onto 2 × 2048 px sheets with no semantic grouping intent. The drawable names are entirely anonymous (`ArtMesh0` through `ArtMesh134`), which is the Cubism Editor default when the rigger does not rename meshes [6]. The result is a set of specific structural problems encountered during manual verification of `manifests/hiyori_atlas.toml` [7]:

**PartArmA contains both body-skin and clothing drawables in the same part.** When suppressing non-target drawables using Live2D's part-opacity mechanism, enabling PartArmA (index 12) for body verification simultaneously makes clothing visible, and vice versa. The part structure does not permit clean isolation of body vs clothing in the arm area.

**ArtMesh102 and ArtMesh104 contain both bare skin and clothing within a single UV bounding box.** The texture region covered by these two drawables has shin/leg skin on one side and sock/clothing on the other. It is impossible to recolor clothing without also affecting skin in this region without a semantic pixel-level mask. This was designated the `cloth_and_body` region with `swap_strategy = "hue_shift"` as a placeholder [7].

**PartEye contains all eye drawables without left/right subdivision.** Left and right eye drawables live in the same part, so part-level isolation cannot distinguish them. Atlas verification for left_eye and right_eye had to rely on drawable UV coordinates rather than rendered isolation [7].

These are not bugs in Hiyori's rig for its intended use (real-time animation via VTube Studio). They are neutral design decisions that only become problems when the rig is used for programmatic texture generation, which was never an authoring goal. They are also common in commercial VTuber rigs because the packing efficiency gain from mixed-semantic sheets is real and visible as reduced draw calls and memory footprint in the runtime.

The root cause is that Cubism Editor's auto-layout does not know about semantic intent. It only knows about pixel area and packing efficiency. A rigger who wants semantic separation must override the auto-layout entirely, working manually sheet by sheet.

### CartoonAlive and Textoon: How They Handle Textures

Both CartoonAlive and Textoon are built by the same research group (Human3DAIGC) and both target the HaiMeng rig as their base template. They assume the rig is already semantically partitioned across dedicated sheets.

CartoonAlive [1] handles portrait-to-texture transfer in four sequential stages. First, the input portrait is eye-rotation-normalized and facial landmarks are detected for eyes, nose, mouth, eyebrows, and contour. An affine transform is computed between detected landmarks and the template's known landmark positions. The aligned image is warped and its semantic regions (face skin, eyes, brows, nose, mouth) are extracted and pasted into their pre-known UV destinations on the atlas. This is not neural texture synthesis — it is affine warp plus pixel crop-and-paste at a fixed coordinate. Second, with facial features temporarily removed from the texture, the rig renders using only the underlying face layer; a 4-layer MLP (trained on 100,000 synthetic renders at 1024 × 1024 px) maps facial keypoints on this stripped render to Live2D positional parameters for each feature component. Third, using the inferred parameters, the pipeline renders binary masks of each facial feature at its predicted position, and repaints the underlying face texture in those masked areas to eliminate animation artefacts where moving features would reveal uncleaned skin beneath. Fourth, hair is extracted separately via a dedicated hair segmentation model; if bangs occlude the brows, a GAN-based hair-removal model (HairMapper) is applied before eyebrow extraction and parameter prediction [5].

Textoon [2] uses SDXL (specifically `realcartoonXL_v7.safetensors` and a fine-tuned SDXL model) via ComfyUI to generate a character image from a text description. The critical implementation is in `utils/transfer_part_texture.py::extract_part_to_texture()`. For each clothing/hair/face piece, `model_configuration.json` stores explicit pixel-coordinate crop rectangles — for example, `"skirt": {"x": 72, "y": 61, "w": 1268, "h": 2554, "name": "texture_03"}`. The pipeline renders a full 3360 × 5040 px character composition from PSD, then uses these coordinates to crop each semantic piece and write it into the corresponding 4096 × 4096 texture sheet at the known destination [5]. No UV unwrapping occurs at runtime. For regions hidden behind other parts (e.g. the back of a shirt arm hidden by a skirt), Textoon fills with pixels from visible regions, then applies image-to-image ControlNet for refinement [2].

Neither system attempts full-sheet neural texture generation. Both rely on the atlas being semantically clean: without the HaiMeng rig's sheet-per-category design, neither pipeline's coordinate lookup tables would point to semantically coherent pixel regions.

### Transformation Strategies: Clean vs Dirty Atlas

The following strategies are discussed in roughly increasing order of atlas requirement cleanliness.

**Hue shift** works on any atlas regardless of semantic mixing. Applying a hue rotation to the entire texture sheet changes all colours uniformly. For a sheet containing mixed semantics (skin + clothing), this is usually wrong unless skin tones and clothing colours happen to be far apart in hue. It can be applied selectively using a colour-distance mask (reject pixels near skin-tone ranges from the shift), which is the approach planned for Hiyori's `cloth_and_body` region [7]. Hue shift is the lowest-fidelity method and produces cartoonish colour changes without any shape transfer.

**Full-sheet replacement** replaces the entire texture image with a newly generated one. This works when the entire sheet carries one semantic concept (e.g. HaiMeng's `texture_03` is all skirt), or when the input image is already styled to match the rig's layout. Textoon's SDXL generation produces a full-body character image and then crops it into per-sheet pieces; that is not really "full-sheet replacement" as the sheets are never independently generated. True full-sheet replacement requires the generator to know the UV layout of the sheet's contents, which is only tractable when the sheet has one semantic category.

**Per-region paste** pastes a warped crop from the source image into a known UV rectangle on the atlas. This is what both CartoonAlive and Textoon do, and it is the most practical strategy for portrait-driven generation. It requires a pre-authored UV crop table specifying the pixel rectangle on each texture sheet where each named region lives. It does not require the sheet to be semantically clean at the whole-sheet level — only the target rectangle must be semantically coherent. Hiyori's face_skin region on tex0 qualifies for this approach despite the sheet's overall mixed content.

**Diffusion inpainting** can regenerate a sub-region of the atlas in place, using the surrounding context as conditioning. This is the most powerful strategy for handling mixed-semantic UV regions: a mask isolates the target pixels (e.g. just the clothing portion of a cloth_and_body region), and inpainting fills those pixels in a style consistent with the new character while leaving adjacent skin untouched. This requires a per-pixel semantic mask (not just a bounding box), and is significantly more compute-intensive than paste operations. No sources reviewed actually use inpainting at atlas-paste time — it appears only as an occlusion-completion step in Textoon (to fill hidden body parts) and as face-skin repainting in CartoonAlive (to clean artefacts). Diffusion inpainting at generation time on a mixed-semantic UV region is a viable approach but adds substantial complexity and latency.

For Hiyori specifically: face_skin, left_eye, right_eye, left_eyebrow, right_eyebrow, mouth, and individual hair regions can all receive per-region paste. The body and clothing regions on tex1 can receive hue shift. The cloth_and_body region requires either inpainting with a pixel mask or is left unmodified. Full-sheet replacement on either of Hiyori's two sheets would require generating a correctly UV-mapped full image, which is impractical without re-rigging.

### Correct Rig Design for Programmatic Texture Generation

The architectural principle demonstrated by HaiMeng and confirmed by both CartoonAlive and Textoon is: **one swappable semantic category per texture sheet**. This is the only design that makes texture injection a trivial pixel-paste operation at generation time.

For the portrait-to-live2d pipeline, the following rig authoring requirements should be enforced for any new generated or commissioned rig:

Use semantic English drawable naming (Type B convention) for all ArtMesh IDs — e.g. `face_skin_main`, `eye_iris_left`, `hair_back_01`, `shirt_body_left`. This enables automated atlas config generation by keyword matching without any user labeling step [6]. Anonymous names like `ArtMesh0` require the UV-extraction-plus-manual-labeling workflow and have no machine-readable semantics.

Partition texture sheets by swappable category. A minimum viable partition for a humanoid avatar is: sheet 0 — face skin + eyes + brows + mouth + cheeks; sheet 1 — all hair variants; sheet 2 — clothing (or further split clothing into tops, bottoms, shoes if garment swapping is required). The face/hair split is the highest-priority partition because it enables portrait-face transfer without touching clothing at all.

Do not mix semantically distinct swappable regions within a single part's drawables. If body skin and clothing share a part, the part-opacity suppression mechanism cannot isolate them during verification or rendering. Each part should contain only one semantic category.

Commit a machine-readable UV crop table alongside the rig at authoring time. The format used by this project (`atlas_config.toml` with `[[regions]]` blocks giving `name`, `texture_index`, and pixel `(x, y, w, h)`) is suitable. The equivalent in Textoon is `model_configuration.json`. This table must be regenerated if the atlas is re-packed in Cubism Editor — it is invalidated by any auto-layout operation that moves drawables.

For HaiMeng specifically: the 9 × 4096 sheet design is already correct. The remaining unknown is the exact UV crop table (`model_configuration.json` from the Textoon repo), which requires EULA-gated access to the HaiMeng assets. Once that file is available, the texture injection pipeline for HaiMeng follows CartoonAlive/Textoon directly with no additional atlas analysis required.

---

## Open Questions

No sources directly address whether Cubism Editor can be scripted or batch-commanded to route specific drawables to specific texture sheets — the documentation covers only the manual UI workflow. If scripting is possible via the Cubism Editor plugin API, atlas re-partitioning of existing rigs (such as Hiyori) could be automated, though re-partitioning requires a re-export of the `.moc3` file which in turn invalidates any existing UV crop tables.

The Textoon paper [2] and CartoonAlive paper [1] both omit implementation details about how the contour boundary (the UV-space silhouette of each semantic region) is represented or whether it is derived from mesh vertex UVs or from a separate mask image. It is assumed to be a pixel-coordinate bounding box from `model_configuration.json`, but the actual UV mesh may use non-rectangular shapes (e.g. L-shaped clothing regions), in which case the bounding box approach would include unwanted background pixels and the paste operation would require an alpha channel or mesh-derived pixel mask.

No sources address whether diffusion inpainting at atlas generation time — using the existing atlas contents as context conditioning — produces results that are spatially coherent with the rig's UV layout. This is a practical question for the cloth_and_body problem in Hiyori and would require experimentation.

---

## Sources

[1] CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits — arXiv:2507.17327 — https://arxiv.org/abs/2507.17327

[2] Textoon: Generating Vivid 2D Cartoon Characters from Text Descriptions — arXiv:2501.10020 — https://arxiv.org/abs/2501.10020

[3] Live2D Cubism Editor Manual — Edit Texture Atlas — https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/

[4] Live2D Dev Team on X — Cubism Editor 5.2.00 beta1 part-filter feature — https://x.com/live2d_dev/status/1871028203608752159

[5] Local prior research — CartoonAlive Texture Editability — `docs/research/2026-04-06-cartoonalive-texture-editability.md`

[6] Local prior research — Multi-Rig UV Extraction Generalization — `docs/research/2026-04-06-multi-rig-uv-extraction-generalization.md`

[7] Local prior research — Atlas Region Verification Lessons — `docs/research/2026-04-06-atlas-region-verification-lessons.md`

[8] Local prior research — Texture Atlas Tools and UV Extraction — `docs/research/2026-04-06-texture-atlas-tools-and-uv-extraction.md`

[9] GitHub — Human3DAIGC/Textoon repository README — https://github.com/Human3DAIGC/Textoon

[10] GitHub — Human3DAIGC/CartoonAlive repository — https://github.com/Human3DAIGC/CartoonAlive

[11] CartoonAlive HTML paper — https://arxiv.org/html/2507.17327v1

[12] Textoon HTML paper — https://arxiv.org/html/2501.10020v1

[13] CartoonAlive project webpage — https://human3daigc.github.io/CartoonAlive_webpage/

[14] Local rig inspection — Hiyori drawable IDs, part map, UV bounding boxes — `manifests/hiyori_atlas.toml`, `manifests/hiyori_drawables_raw.json`, `manifests/hiyori_part_map.json`
