import os
import sys
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional imports
try:
    import cv2  # for MP4 output
except Exception:  # pragma: no cover
    cv2 = None

try:
    import imageio.v2 as imageio  # for GIF output
except Exception:  # pragma: no cover
    imageio = None


# -----------------------------
# Utilities
# -----------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_domain_arg(s: str) -> Tuple[str, str]:
    # Expect format Label:/abs/path
    if ":" not in s:
        raise ValueError(f"Invalid --domain '{s}'. Expected 'Label:/abs/path'.")
    label, path = s.split(":", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise ValueError(f"Invalid --domain '{s}': empty label.")
    if not os.path.isdir(path):
        raise ValueError(f"Invalid --domain '{s}': directory does not exist: {path}")
    return label, path


def parse_hex_color(s: str, default=(0, 0, 0)) -> Tuple[int, int, int]:
    if not s:
        return default
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 3:
        r, g, b = s
        s = f"{r}{r}{g}{g}{b}{b}"
    if len(s) != 6:
        return default
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return default


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    # Robust fallback across systems
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    # Try common fonts
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception:
                continue
    # Default bitmap font
    return ImageFont.load_default()


def safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    # Returns (x0, y0, x1, y1)
    return draw.textbbox((0, 0), text, font=font)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    x0, y0, x1, y1 = text_bbox(draw, text, font)
    return x1 - x0


def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    x0, y0, x1, y1 = text_bbox(draw, text, font)
    return y1 - y0


def wrap_text_to_width(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    # Simple greedy word wrap on spaces. If a single word exceeds max_w, we hard-split it.
    words = text.replace('\n', ' ').split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = current + ' ' + w
        if text_width(draw, candidate, font) <= max_w:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)

    # Handle extremely long single tokens
    processed: List[str] = []
    for line in lines:
        if text_width(draw, line, font) <= max_w:
            processed.append(line)
        else:
            # hard wrap inside the long token
            token = line
            chunk = ""
            for ch in token:
                if text_width(draw, chunk + ch, font) <= max_w:
                    chunk += ch
                else:
                    if chunk:
                        processed.append(chunk)
                    chunk = ch
            if chunk:
                processed.append(chunk)
    return processed if processed else [""]


def truncate_lines_with_ellipsis(lines: List[str], draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_w: int, max_lines: int) -> List[str]:
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    truncated = lines[:max_lines]
    # ensure last line fits with ellipsis
    last = truncated[-1]
    ell = '…'
    if text_width(draw, last + ell, font) <= max_w:
        truncated[-1] = last + ell
    else:
        # remove characters until fits
        s = last
        while s and text_width(draw, s + ell, font) > max_w:
            s = s[:-1]
        truncated[-1] = (s + ell) if s else ell
    return truncated


def letterbox(image: Image.Image, target_w: int, target_h: int, bg: Tuple[int, int, int]) -> Image.Image:
    w, h = image.size
    if w == 0 or h == 0:
        canvas = Image.new('RGB', (target_w, target_h), color=bg)
        return canvas
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), resample=Image.LANCZOS)
    canvas = Image.new('RGB', (target_w, target_h), color=bg)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


# -----------------------------
# Domain scanning & captions
# -----------------------------

def build_domain_mapping(dir_path: str, real_suffix: str, fake_suffix: str) -> Dict[str, Tuple[str, str]]:
    files = os.listdir(dir_path)
    real_ids = set()
    fake_ids = set()
    for f in files:
        if f.endswith(real_suffix):
            base = f[: -len(real_suffix)]
            real_ids.add(base)
        elif f.endswith(fake_suffix):
            base = f[: -len(fake_suffix)]
            fake_ids.add(base)
    pair_ids = sorted(real_ids & fake_ids)
    mapping: Dict[str, Tuple[str, str]] = {}
    for bid in pair_ids:
        mapping[bid] = (
            os.path.join(dir_path, bid + real_suffix),
            os.path.join(dir_path, bid + fake_suffix),
        )
    if not mapping:
        raise RuntimeError(f"No (real,fake) pairs found in {dir_path} with suffixes '{real_suffix}', '{fake_suffix}'.")
    return mapping


def load_captions_from_json(json_path: str, num_domains: int, labels_json_key: str, labels_keys: Optional[List[str]]) -> List[str]:
    if not os.path.isfile(json_path):
        raise RuntimeError(f"labels_json does not exist: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def item_to_text(item):
        if isinstance(item, dict):
            val = item.get(labels_json_key, None)
            if val is None:
                # fallback: try any first string-like value
                for v in item.values():
                    if isinstance(v, str):
                        return v
                return json.dumps(item, ensure_ascii=False)
            return str(val)
        return str(item)

    if isinstance(data, list):
        items = data[:num_domains]
        if len(items) != num_domains:
            raise RuntimeError(f"labels_json list has {len(items)} items; expected {num_domains}.")
        return [item_to_text(it) for it in items]

    if isinstance(data, dict):
        # determine ordering
        keys: List[str]
        if labels_keys:
            keys = [str(k) for k in labels_keys]
        else:
            # auto-sort: numeric-like keys numerically, others lexicographically
            def key_sorter(k: str):
                try:
                    return (0, int(k))
                except Exception:
                    return (1, str(k))
            keys = sorted(data.keys(), key=key_sorter)
        if len(keys) < num_domains:
            raise RuntimeError(f"labels_json object provides {len(keys)} keys; expected at least {num_domains}.")
        keys = keys[:num_domains]
        return [item_to_text(data[k]) for k in keys]

    raise RuntimeError("labels_json must be a list or object.")


# -----------------------------
# Layout computation
# -----------------------------

class Layout:
    def __init__(self,
                 canvas_w: int,
                 canvas_h: int,
                 margin: int,
                 gap: int,
                 content_w: int,
                 columns: int,
                 rows: int,
                 tile_w: int,
                 tile_h: int,
                 caption_box_h: int,
                 orig_alloc_w: int,
                 orig_h: int,
                 sep_line_th: int,
                 orig_x: int,
                 top_y: int):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.margin = margin
        self.gap = gap
        self.content_w = content_w
        self.columns = columns
        self.rows = rows
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.caption_box_h = caption_box_h
        self.orig_alloc_w = orig_alloc_w
        self.orig_h = orig_h
        self.sep_line_th = sep_line_th
        self.orig_x = orig_x
        self.top_y = top_y


def compute_layout(
    width: int,
    margin: int,
    gap: int,
    columns: int,
    num_domains: int,
    tile_h_forced: int,
    orig_width_ratio: float,
    bg_color: Tuple[int, int, int],
    captions_lines: List[List[str]],
    font: ImageFont.ImageFont,
    line_spacing: int,
    ref_orig_img: Image.Image,
    ref_domain_fake_img: Image.Image,
) -> Layout:
    # content width
    content_w = width - 2 * margin
    if content_w <= 0:
        raise RuntimeError("Canvas width too small for given margin.")

    # grid geometry
    cols = columns if columns and columns > 0 else num_domains
    rows = int(math.ceil(num_domains / cols))
    tile_w = (content_w - (cols - 1) * gap) // cols
    if tile_w <= 0:
        raise RuntimeError("Tile width computed as <= 0. Reduce columns or gap or increase width/margin.")

    if tile_h_forced and tile_h_forced > 0:
        tile_h = tile_h_forced
    else:
        w, h = ref_domain_fake_img.size
        aspect = h / max(1, w)
        tile_h = int(round(tile_w * aspect))

    # caption box height from provided wrapped lines (max lines length)
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    line_h = text_height(dummy_draw, "Ag", font)
    max_lines = max(len(lines) for lines in captions_lines) if captions_lines else 1
    caption_pad_y = 8
    caption_box_h = caption_pad_y * 2 + max_lines * line_h + (max_lines - 1) * line_spacing

    # original allocation
    orig_width_ratio = float(max(0.3, min(1.0, orig_width_ratio)))
    orig_alloc_w = int(round(orig_width_ratio * content_w))
    # reference original aspect to compute a consistent height across frames
    ow, oh = ref_orig_img.size
    ref_top_h = int(round(orig_alloc_w * (oh / max(1, ow))))

    sep_line_th = 1
    # grid total height (tiles + caption boxes + gaps between rows)
    grid_h = rows * (tile_h + caption_box_h) + (rows - 1) * gap

    canvas_h = margin + ref_top_h + sep_line_th + gap + grid_h + margin

    # original X such that centered within content width
    orig_x = margin + (content_w - orig_alloc_w) // 2
    top_y = margin

    return Layout(
        canvas_w=width,
        canvas_h=canvas_h,
        margin=margin,
        gap=gap,
        content_w=content_w,
        columns=cols,
        rows=rows,
        tile_w=tile_w,
        tile_h=tile_h,
        caption_box_h=caption_box_h,
        orig_alloc_w=orig_alloc_w,
        orig_h=ref_top_h,
        sep_line_th=sep_line_th,
        orig_x=orig_x,
        top_y=top_y,
    )


# -----------------------------
# Rendering
# -----------------------------

def compose_frame(
    base_id: str,
    layout: Layout,
    bg_color: Tuple[int, int, int],
    caption_color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    captions_lines: List[List[str]],
    captions_align: str,
    caption_pad_x: int,
    line_spacing: int,
    original_path: str,
    fake_paths_per_domain: List[str],
) -> Image.Image:
    # Base canvas
    canvas = Image.new('RGB', (layout.canvas_w, layout.canvas_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    # Original image area (letterbox into orig_alloc_w x orig_h)
    try:
        orig_img = safe_open_rgb(original_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open original for id '{base_id}': {original_path}: {e}")

    orig_lb = letterbox(orig_img, layout.orig_alloc_w, layout.orig_h, bg_color)
    canvas.paste(orig_lb, (layout.orig_x, layout.top_y))

    # Separator line below original
    sep_y = layout.top_y + layout.orig_h
    # A light line: blend towards white-ish
    line_color = tuple(min(255, c + 40) for c in bg_color)
    draw.rectangle([layout.margin, sep_y, layout.margin + layout.content_w, sep_y + layout.sep_line_th - 1], fill=line_color)

    # Grid origin (top-left of first tile)
    grid_origin_y = sep_y + layout.sep_line_th + layout.gap

    # Draw tiles + captions
    for idx, fake_path in enumerate(fake_paths_per_domain):
        col = idx % layout.columns
        row = idx // layout.columns
        x = layout.margin + col * (layout.tile_w + layout.gap)
        y = grid_origin_y + row * (layout.tile_h + layout.caption_box_h + layout.gap)

        try:
            fake_img = safe_open_rgb(fake_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open fake for id '{base_id}': {fake_path}: {e}")

        tile_img = letterbox(fake_img, layout.tile_w, layout.tile_h, bg_color)
        canvas.paste(tile_img, (x, y))

        # Caption box
        cap_x0, cap_y0 = x, y + layout.tile_h
        cap_x1, cap_y1 = x + layout.tile_w, cap_y0 + layout.caption_box_h
        draw.rectangle([cap_x0, cap_y0, cap_x1, cap_y1], fill=(0, 0, 0))

        # Draw wrapped lines
        lines = captions_lines[idx] if idx < len(captions_lines) else [""]
        cy = cap_y0 + 8  # vertical padding (matches compute_layout)
        for line in lines:
            if not line:
                lh = text_height(draw, "Ag", font)
                cy += lh + line_spacing
                continue
            if captions_align == 'center':
                tw = text_width(draw, line, font)
                cx = x + (layout.tile_w - tw) // 2
            else:  # left
                cx = x + caption_pad_x
            draw.text((cx, cy), line, font=font, fill=caption_color)
            lh = text_height(draw, line, font)
            cy += lh + line_spacing

    return canvas


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Compose an overview grid video/GIF with top original and bottom domain tiles.")

    # I/O selection
    parser.add_argument('--domain', action='append', required=True,
                        help="Repeatable: 'Label:/abs/path'.")
    parser.add_argument('--original_dir', type=str, default=None,
                        help='Optional explicit folder for originals.')
    parser.add_argument('--real_suffix', type=str, default='_real.png')
    parser.add_argument('--fake_suffix', type=str, default='_fake.png')
    parser.add_argument('--output_mp4', type=str, default=None)
    parser.add_argument('--output_gif', type=str, default=None)
    parser.add_argument('--output_webm', type=str, default=None)
    parser.add_argument('--output_frames_dir', type=str, default=None)
    parser.add_argument('--fps', type=float, default=8.0)
    parser.add_argument('--max_frames', type=int, default=0, help='0 = all frames')

    # Layout & rendering
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--margin', type=int, default=60)
    parser.add_argument('--gap', type=int, default=20)
    parser.add_argument('--columns', type=int, default=0, help='0 = one row with all domains')
    parser.add_argument('--tile_height', type=int, default=0, help='0 = auto from aspect')
    parser.add_argument('--orig_width_ratio', type=float, default=0.8)
    parser.add_argument('--bg', type=str, default='#111111')
    parser.add_argument('--caption_color', type=str, default='#FFFFFF')
    parser.add_argument('--font', type=str, default=None)
    parser.add_argument('--font_size', type=int, default=20)
    parser.add_argument('--caption_align', type=str, choices=['left', 'center'], default='center')
    parser.add_argument('--label_max_lines', type=int, default=0, help='0 = unlimited')

    # Captions from JSON
    parser.add_argument('--labels_json', type=str, default=None)
    parser.add_argument('--labels_json_key', type=str, default='main_prompt')
    parser.add_argument('--labels_keys', type=str, nargs='*', default=None)

    args = parser.parse_args()

    # Parse domains
    try:
        domains: List[Tuple[str, str]] = [parse_domain_arg(s) for s in args.domain]
    except Exception as e:
        eprint(f"Error: {e}")
        sys.exit(1)

    # Build mappings
    domain_maps: List[Dict[str, Tuple[str, str]]] = []
    for label, dpath in domains:
        try:
            mapping = build_domain_mapping(dpath, args.real_suffix, args.fake_suffix)
        except Exception as e:
            eprint(f"Error: {e}")
            sys.exit(1)
        domain_maps.append(mapping)

    # Compute intersection of base IDs
    id_sets = [set(m.keys()) for m in domain_maps]
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if not common_ids:
        eprint("Error: No common base IDs across provided domains.")
        sys.exit(1)

    # Truncate by max_frames if requested
    if args.max_frames and args.max_frames > 0:
        common_ids = common_ids[: args.max_frames]

    # Determine source of originals
    originals_dir = args.original_dir
    if originals_dir is not None:
        if not os.path.isdir(originals_dir):
            eprint(f"Error: --original_dir does not exist: {originals_dir}")
            sys.exit(1)
    else:
        # Use first domain's real images
        originals_dir = None  # signal to use paths from first domain mapping

    # Captions
    if args.labels_json:
        try:
            captions = load_captions_from_json(args.labels_json, len(domains), args.labels_json_key, args.labels_keys)
        except Exception as e:
            eprint(f"Error: {e}")
            sys.exit(1)
    else:
        captions = [label for (label, _) in domains]

    # Prepare font and colors
    font = load_font(args.font, args.font_size)
    bg_color = parse_hex_color(args.bg, default=(17, 17, 17))
    caption_color = parse_hex_color(args.caption_color, default=(255, 255, 255))

    # Prepare a dummy draw for wrapping
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))

    # Wrap captions now to compute caption box height; respect label_max_lines (0 = unlimited)
    caption_pad_x = 12
    line_spacing = 4

    # Reference images for layout
    ref_id = common_ids[0]
    # ref original
    if args.original_dir:
        ref_orig_path = os.path.join(args.original_dir, ref_id + args.real_suffix)
    else:
        ref_orig_path = domain_maps[0][ref_id][0]
    try:
        ref_orig_img = safe_open_rgb(ref_orig_path)
    except Exception as e:
        eprint(f"Error opening reference original: {ref_orig_path}: {e}")
        sys.exit(1)

    # ref domain fake (first domain)
    ref_fake_path = domain_maps[0][ref_id][1]
    try:
        ref_fake_img = safe_open_rgb(ref_fake_path)
    except Exception as e:
        eprint(f"Error opening reference fake: {ref_fake_path}: {e}")
        sys.exit(1)

    # compute a provisional tile width to wrap captions appropriately (depends on layout)
    # We'll compute layout twice: first, assume captions_lines for an initial tile_w guess using columns config; then recompute
    cols_initial = args.columns if args.columns and args.columns > 0 else len(domains)
    content_w_initial = args.width - 2 * args.margin
    tile_w_initial = (content_w_initial - (cols_initial - 1) * args.gap) // cols_initial
    if tile_w_initial <= 0:
        eprint("Error: Tile width computed as <= 0 with current settings. Adjust --width/--margin/--gap/--columns.")
        sys.exit(1)

    max_text_w = max(1, tile_w_initial - 2 * caption_pad_x)

    wrapped_captions = [wrap_text_to_width(captions[i], dummy_draw, font, max_text_w) for i in range(len(captions))]
    if args.label_max_lines and args.label_max_lines > 0:
        wrapped_captions = [truncate_lines_with_ellipsis(lines, dummy_draw, font, max_text_w, args.label_max_lines)
                            for lines in wrapped_captions]

    # Compute final layout with accurate caption box height
    layout = compute_layout(
        width=args.width,
        margin=args.margin,
        gap=args.gap,
        columns=args.columns,
        num_domains=len(domains),
        tile_h_forced=args.tile_height,
        orig_width_ratio=args.orig_width_ratio,
        bg_color=bg_color,
        captions_lines=wrapped_captions,
        font=font,
        line_spacing=line_spacing,
        ref_orig_img=ref_orig_img,
        ref_domain_fake_img=ref_fake_img,
    )

    # Re-wrap captions in case tile_w changed (rare if same as initial)
    max_text_w_final = max(1, layout.tile_w - 2 * caption_pad_x)
    wrapped_captions = [wrap_text_to_width(captions[i], dummy_draw, font, max_text_w_final) for i in range(len(captions))]
    if args.label_max_lines and args.label_max_lines > 0:
        wrapped_captions = [truncate_lines_with_ellipsis(lines, dummy_draw, font, max_text_w_final, args.label_max_lines)
                            for lines in wrapped_captions]

    # Setup writers
    write_mp4 = args.output_mp4 is not None and len(args.output_mp4) > 0
    write_gif = args.output_gif is not None and len(args.output_gif) > 0
    write_webm = args.output_webm is not None and len(args.output_webm) > 0
    write_frames = args.output_frames_dir is not None and len(args.output_frames_dir) > 0

    if write_mp4:
        if cv2 is None:
            eprint("Error: OpenCV (cv2) is required for MP4 output but is not available.")
            sys.exit(1)
        mp4_dir = os.path.dirname(args.output_mp4)
        if mp4_dir:
            os.makedirs(mp4_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_mp4, fourcc, float(args.fps), (layout.canvas_w, layout.canvas_h))
        if not video_writer.isOpened():
            eprint(f"Error: Failed to open video writer for {args.output_mp4}")
            sys.exit(1)
    else:
        video_writer = None

    webm_writer = None
    if write_webm:
        if imageio is None:
            eprint("Error: imageio is required for WEBM output but is not available.")
            if video_writer is not None:
                video_writer.release()
            sys.exit(1)
        webm_dir = os.path.dirname(args.output_webm)
        if webm_dir:
            os.makedirs(webm_dir, exist_ok=True)
        # Try VP9, then VP8
        try:
            webm_writer = imageio.get_writer(
                args.output_webm,
                fps=float(args.fps),
                codec='libvpx-vp9',
                quality=8,
                macro_block_size=None,
                ffmpeg_params=['-pix_fmt', 'yuv420p']
            )
        except Exception as e1:
            try:
                webm_writer = imageio.get_writer(
                    args.output_webm,
                    fps=float(args.fps),
                    codec='libvpx',
                    quality=8,
                    macro_block_size=None,
                    ffmpeg_params=['-pix_fmt', 'yuv420p']
                )
            except Exception as e2:
                eprint(f"Error: Failed to open WEBM writer for {args.output_webm}: {e1} | {e2}")
                if video_writer is not None:
                    video_writer.release()
                sys.exit(1)

    if write_frames:
        os.makedirs(args.output_frames_dir, exist_ok=True)

    gif_frames: List[Image.Image] = []
    if write_gif and imageio is None:
        eprint("Warning: imageio not available; GIF output will be skipped.")
        write_gif = False

    # Iterate frames
    for idx, bid in enumerate(common_ids):
        # Original path for this frame
        if args.original_dir:
            orig_path = os.path.join(args.original_dir, bid + args.real_suffix)
            if not os.path.isfile(orig_path):
                eprint(f"Error: Missing original for base_id '{bid}' in --original_dir: {orig_path}")
                if video_writer is not None:
                    video_writer.release()
                sys.exit(1)
        else:
            orig_path = domain_maps[0][bid][0]

        # Fake paths per domain in given order
        fake_paths = [domain_maps[i][bid][1] for i in range(len(domains))]

        frame_img = compose_frame(
            base_id=bid,
            layout=layout,
            bg_color=bg_color,
            caption_color=caption_color,
            font=font,
            captions_lines=wrapped_captions,
            captions_align=args.caption_align,
            caption_pad_x=caption_pad_x,
            line_spacing=line_spacing,
            original_path=orig_path,
            fake_paths_per_domain=fake_paths,
        )

        if write_mp4:
            frame_bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        if write_webm:
            webm_writer.append_data(np.array(frame_img))

        if write_gif:
            gif_frames.append(frame_img)

        if write_frames:
            out_path = os.path.join(args.output_frames_dir, f"{idx:06d}_{bid}.png")
            frame_img.save(out_path)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(common_ids)} frames...")

    # Finalize outputs
    if write_mp4 and video_writer is not None:
        video_writer.release()
        print(f"Saved MP4 to {args.output_mp4}")

    if write_webm and webm_writer is not None:
        webm_writer.close()
        print(f"Saved WEBM to {args.output_webm}")

    if write_gif and gif_frames:
        os.makedirs(os.path.dirname(args.output_gif), exist_ok=True)
        # duration per frame in seconds
        duration = 1.0 / max(0.1, float(args.fps))
        imageio.mimsave(args.output_gif, [np.array(f) for f in gif_frames], duration=duration)
        print(f"Saved GIF to {args.output_gif}")

    if write_frames:
        print(f"Saved frames to {args.output_frames_dir}")

    print("Done.")


if __name__ == '__main__':
    main()
