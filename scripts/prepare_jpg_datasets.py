#!/usr/bin/env python3
"""
Prepare CycleGAN or pix2pix datasets from a jpg_dataset tree.

Source layout (fixed):
  <root>/jpg_dataset/
    original/<camera>/
    augmented/<domain_name>/<camera>/   # domain_name examples: domain_0, domain_1, domain_2, ...

This script creates, for each domain_*, dataset folders under --output-root,
without symlinks.

For CycleGAN (unaligned):
  <output-root>/<name>_cyclegan/
    trainA/, trainB/, testA/, testB/

For pix2pix (aligned AB):
  <output-root>/<name>_pix2pix/
    train/, test/    # each contains horizontally-concatenated AB images

Defaults:
  - camera: rgb_front
  - match only files present in both A (original) and B (augmented/domain_*)
  - split: train 80%, test 20%
  - concatenation done with Pillow (no OpenCV dependency)

Example:
  python scripts/prepare_jpg_datasets.py \
    --source-root jpg_dataset \
    --output-root datasets \
    --camera rgb_front \
    --mode cyclegan pix2pix
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_image_files(dir_path: Path) -> List[str]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    names = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            names.append(p.name)
    return sorted(names)


def intersect_filenames(a: List[str], b: List[str]) -> List[str]:
    aset = set(a)
    bset = set(b)
    common = sorted(list(aset & bset))
    return common


def train_test_split(items: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    n_train = int(len(items) * train_ratio)
    return items[:n_train], items[n_train:]


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def copy_files(names: List[str], src: Path, dst: Path):
    ensure_dir(dst)
    for i, name in enumerate(names, 1):
        shutil.copy2(src / name, dst / name)
        if i % 1000 == 0:
            print(f"  copied {i}/{len(names)} -> {dst}")


def concat_AB_and_save(names: List[str], A_dir: Path, B_dir: Path, out_dir: Path, resize: str = "none"):
    """
    Concatenate A (left) and B (right). If sizes differ:
      - resize == 'A': resize B to A size
      - resize == 'B': resize A to B size
      - resize == 'none': skip mismatched pairs
    """
    ensure_dir(out_dir)
    skipped = 0
    saved = 0
    for i, name in enumerate(names, 1):
        a_path = A_dir / name
        b_path = B_dir / name
        try:
            A = Image.open(a_path).convert("RGB")
            B = Image.open(b_path).convert("RGB")
        except Exception as e:
            print(f"  [read-error] {name}: {e}")
            skipped += 1
            continue
        if A.size != B.size:
            if resize == "A":
                B = B.resize(A.size, Image.BICUBIC)
            elif resize == "B":
                A = A.resize(B.size, Image.BICUBIC)
            else:
                skipped += 1
                if skipped <= 10:
                    print(f"  [skip-size-mismatch] {name}: A{A.size} vs B{B.size}")
                continue
        W, H = A.size
        out = Image.new("RGB", (W + B.size[0], H))
        out.paste(A, (0, 0))
        out.paste(B, (W, 0))
        out.save(out_dir / name)
        saved += 1
        if i % 1000 == 0:
            print(f"  saved {i}/{len(names)} AB -> {out_dir}")
    print(f"  AB saved: {saved}, skipped: {skipped} -> {out_dir}")


def derive_dataset_name(src_domain_dir: Path) -> str:
    # src_domain_dir.name like 'domain_2' -> 'domain2'
    nm = src_domain_dir.name.replace("_", "")  # domain_2 -> domain2
    return f"driving_rgb2{nm}"


def main():
    p = argparse.ArgumentParser(description="Prepare CycleGAN/pix2pix datasets from jpg_dataset")
    p.add_argument("--source-root", type=str, default="jpg_dataset", help="Path to jpg_dataset root")
    p.add_argument("--output-root", type=str, default="datasets", help="Where to create datasets")
    p.add_argument("--camera", type=str, default="rgb_front", help="Camera subfolder name under each domain")
    p.add_argument("--domains", type=str, nargs="*", default=None, help="Specific domain_* to process. If omitted, auto-detect all under augmented/")
    p.add_argument("--mode", type=str, nargs="+", choices=["cyclegan", "pix2pix"], default=["cyclegan", "pix2pix"], help="Which dataset formats to create")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (rest goes to test)")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p.add_argument("--pix2pix-resize", type=str, choices=["none", "A", "B"], default="none", help="If A/B sizes differ when making AB, how to resize")
    p.add_argument("--same-train-test", action="store_true", help="Use all common images as train and the same set as test (no split)")
    args = p.parse_args()

    src_root = Path(args.source_root)
    out_root = Path(args.output_root)

    A_dir = src_root / "original" / args.camera
    if not A_dir.exists():
        raise SystemExit(f"Missing original camera dir: {A_dir}")

    aug_root = src_root / "augmented"
    if not aug_root.exists():
        raise SystemExit(f"Missing augmented root: {aug_root}")

    if args.domains:
        domain_dirs = [aug_root / d for d in args.domains]
    else:
        domain_dirs = [d for d in aug_root.iterdir() if d.is_dir() and d.name.startswith("domain_")]
    domain_dirs = sorted(domain_dirs, key=lambda p: p.name)
    if not domain_dirs:
        raise SystemExit("No domain_* directories found under augmented/")

    A_names = list_image_files(A_dir)
    if not A_names:
        raise SystemExit(f"No images found in {A_dir}")

    print(f"Found {len(A_names)} images in A (original/{args.camera})")
    print(f"Domains to process: {[d.name for d in domain_dirs]}")

    for domain_dir in domain_dirs:
        B_dir = domain_dir / args.camera
        B_names = list_image_files(B_dir)
        if not B_names:
            print(f"[warn] No images in {B_dir}, skipping domain {domain_dir.name}")
            continue
        common = intersect_filenames(A_names, B_names)
        if not common:
            print(f"[warn] No common filenames between {A_dir} and {B_dir}, skipping")
            continue
        if args.same_train_test:
            train_files, test_files = common, common
        else:
            train_files, test_files = train_test_split(common, args.train_ratio, args.seed)
        name_base = derive_dataset_name(domain_dir)
        print(f"\n== {domain_dir.name} -> {name_base} ==")
        print(f"  common files: {len(common)} | train: {len(train_files)} | test: {len(test_files)}")

        if "cyclegan" in args.mode:
            dest = out_root / f"{name_base}_cyclegan"
            ta, tb, va, vb = dest / "trainA", dest / "trainB", dest / "testA", dest / "testB"
            print(f"  [CycleGAN] -> {dest}")
            copy_files(train_files, A_dir, ta)
            copy_files(train_files, B_dir, tb)
            copy_files(test_files, A_dir, va)
            copy_files(test_files, B_dir, vb)

        if "pix2pix" in args.mode:
            dest = out_root / f"{name_base}"
            tr, te = dest / "train", dest / "test"
            print(f"  [pix2pix] -> {dest}")
            concat_AB_and_save(train_files, A_dir, B_dir, tr, resize=args.pix2pix_resize)
            concat_AB_and_save(test_files, A_dir, B_dir, te, resize=args.pix2pix_resize)

    print("\nDone.")


if __name__ == "__main__":
    main()
