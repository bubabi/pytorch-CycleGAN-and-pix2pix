# generate_overview_grid.py — Functional Specification

## Overview
- Compose a sequence of frames that showcase, per ID:
  - Top: the original image.
  - Bottom: a grid of generated images from multiple "domains" (e.g., different models/variants), each with a caption.
- Optionally assemble these frames into an animation (video and/or GIF). No format/codec specifics are covered in this document.
- Designed around CycleGAN-style result folders using `<id>_real.png` and `<id>_fake.png` naming.

## Input data model and naming conventions
- Domains are passed as repeated arguments in the form `Label:/abs/path/to/images`.
- Each domain directory should contain files following the naming convention:
  - Original (real): `<id><real_suffix>` (default: `_real.png`)
  - Generated (fake): `<id><fake_suffix>` (default: `_fake.png`)
- A pair exists when both real and fake for the same `<id>` are present in a domain directory.
- The script computes the intersection of `<id>` values across all domains and processes those IDs.
- Originals source:
  - If `--original_dir` is specified, originals are taken from that directory.
  - Otherwise, originals are taken from the first domain’s real images.

## Command line interface (grouped)
- I/O selection
  - `--domain Label:/path` (repeatable) — domain label and directory.
  - `--original_dir /path` — optional explicit folder for originals.
  - `--real_suffix`, `--fake_suffix` — file suffixes (defaults: `_real.png`, `_fake.png`).
  - `--output_mp4 /path/out.mp4` — optional path to save video.
  - `--output_gif /path/out.gif` — optional path to save GIF.
  - `--fps <float>` — animation timing for both video and GIF.
  - `--max_frames <int>` — limit number of frames (0 = all).
- Layout & rendering
  - `--width <int>` — canvas width; height is computed.
  - `--margin <int>` — outer margin in pixels.
  - `--gap <int>` — gap between tiles and sections.
  - `--columns <int>` — number of columns for the bottom grid (0 = one row with all domains).
  - `--tile_height <int>` — force tile height (0 = auto based on aspect of domain images).
  - `--orig_width_ratio <float>` — fraction of content width used by the top original (0.3–1.0).
  - `--bg <#hex>` — background color.
  - `--caption_color <#hex>` — caption text color.
  - `--font /path/to.ttf`, `--font_size <int>` — caption font configuration (fallbacks provided).
  - `--caption_align left|center` — text alignment in caption boxes.
  - `--label_max_lines <int>` — max caption lines per tile (ellipsis if exceeded).
- Captions from JSON
  - `--labels_json /path.json` — optional JSON source for captions to override the domain labels.
  - `--labels_json_key key` — key extracted from each JSON item (default: `main_prompt`).
  - `--labels_keys k1 k2 ...` — explicit ordering of keys when JSON is an object.

## Processing pipeline
1. Parse and validate domains (format and directory existence).
2. For each domain, build a mapping `base_id -> (real_path, fake_path)` using the suffix rules.
3. Compute the intersection of `base_id`s across domains; sort for deterministic processing; optionally truncate by `--max_frames`.
4. If `--labels_json` is given, extract a captions list matching the number of domains:
   - JSON list: take first N items; if an item is an object, extract `--labels_json_key`, else stringify.
   - JSON object: use `--labels_keys` if provided, else auto-sort keys (numeric-like keys numerically, others lexicographically); then extract values similarly.
   - Validate length = number of domains.
5. Prepare font and colors (robust font fallback; hex color parsing).
6. Determine layout metrics:
   - Load a reference original for sizing.
   - If `--tile_height` is 0, infer tile height from the first domain image’s aspect ratio applied to the computed tile width.
7. For each `base_id`:
   - Load the original (from `--original_dir` or the first domain’s real).
   - Load the fake image for each domain.
   - Compose a frame via the layout algorithm (see next section).
   - If output paths were provided, append the frame to the selected outputs.
8. Finalize and save outputs; print progress every 10 frames.

## Layout/composition details
- Canvas width = `--width`; total height is computed.
- Content width = `width - 2*margin`.
- Grid configuration:
  - Columns = `--columns` if > 0, else number of domains.
  - Rows = ceil(N / columns).
  - Tile width = `(content_w - (cols - 1)*gap) // cols`.
  - Tile height:
    - If forced via `--tile_height` (> 0), use it.
    - Else, preserve aspect based on the first domain image.
- Original (top) section:
  - Allocated width = `orig_width_ratio * content_w` (clamped to 0.3–1.0).
  - Height preserves original’s aspect ratio.
  - Original image is resized and centered horizontally; a light separator line is drawn below it.
- Domain tiles (bottom grid):
  - Each domain image is letterboxed to `tile_w x tile_h` on the background color.
  - Under each tile, a caption box (black) is drawn.
  - Caption text (from JSON or domain label) is wrapped with padding, alignment rules, and a maximum line count; ellipsis when truncated.
  - Caption box height is computed from the maximum number of wrapped lines across all captions to keep the grid uniform.

## Error handling & assumptions
- Invalid `--domain` format or non-existent directories → error.
- No pairs found for a domain given the suffix rules → error.
- No common `base_id`s across domains → error.
- Missing original in `--original_dir` for any `base_id` → error.
- `labels_json` path missing, wrong type, or item count mismatch → error.

## Dependencies (functional)
- Required: Pillow (PIL), NumPy.
- Optional: OpenCV (for MP4 output if requested), imageio (for `--gif_method imageio`).

## Outputs
- A sequence of composed frames in the order of sorted `base_id`s (optionally truncated).
- Optional MP4 and/or GIF animation; timing governed by `--fps`.

## Example usage (format-agnostic)
```bash
python generate_overview_grid.py \
  --domain "D1:/abs/path/to/domain1/images" \
  --domain "D2:/abs/path/to/domain2/images" \
  --domain "D3:/abs/path/to/domain3/images" \
  --original_dir /abs/path/to/originals \
  --labels_json /abs/path/to/labels.json \
  --labels_json_key main_prompt \
  --labels_keys 1 2 3 \
  --columns 3 \
  --width 1920 \
  --fps 8 \
  --max_frames 300 \
  --label_max_lines 0 \
  --caption_align center \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --output_mp4 /abs/path/to/overview.mp4 \
  --output_gif /abs/path/to/overview.gif
```
