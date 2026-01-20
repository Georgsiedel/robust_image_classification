#!/usr/bin/env python3
"""
Convert TIFF ImageFolder -> HDF5 with TRANSCODING (TIFF -> PNG/JPEG).

Solves two problems:
1. Compatibility: Converts TIFF (hard to decode fast) to PNG/JPEG (fast C++ decoding).
2. Bandwidth: Stores compressed bytes (PNG/JPEG) instead of raw arrays.

Usage:
  python convert_tiff_transcode.py --src /path/to/data --format png
"""
import argparse
import json
import io
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
from typing import List, Tuple, Dict, Set

# Increase max pixels just in case (Satellite images can be huge)
Image.MAX_IMAGE_PIXELS = None

def read_list_file(list_path: Path) -> Tuple[Set[str], Set[str]]:
    full_paths = set()
    basenames = set()
    if not list_path.exists():
        return full_paths, basenames
    with list_path.open("r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            full_paths.add(p.as_posix())
            basenames.add(p.name)
    return full_paths, basenames

def find_all_image_files(root: Path, exts_str: str = ".tif,.tiff"):
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    input_exts = [e.strip() for e in exts_str.split(",")]
    valid_exts = set()
    for e in input_exts:
        if not e.startswith("."): e = "." + e
        valid_exts.add(e.lower())
        valid_exts.add(e.upper())

    files = []
    labels = []
    for c in classes:
        class_dir = root / c
        class_files = []
        for ext in valid_exts:
            class_files.extend(list(class_dir.rglob(f"*{ext}")))
        class_files.sort()
        for p in class_files:
            files.append(p)
            labels.append(class_to_idx[c])
    return files, labels, class_to_idx

def select_split(files, labels, root, split_full, split_base):
    selected_files = []
    selected_labels = []
    for p, lab in zip(files, labels):
        try:
            rel_posix = p.relative_to(root).as_posix()
        except Exception:
            rel_posix = p.name
        
        if rel_posix in split_full:
            selected_files.append(p)
            selected_labels.append(lab)
        elif p.name in split_base:
            selected_files.append(p)
            selected_labels.append(lab)
    return selected_files, selected_labels

def transcode_image_to_bytes(path: Path, target_format: str = "PNG", quality: int = 95) -> bytes:
    """
    Opens TIFF (or any image), converts to RGB, and saves as compressed PNG/JPEG bytes.
    """
    with Image.open(path) as img:
        # Ensure RGB (strips Alpha if present, handles Grayscale, handles 16-bit TIFF -> 8-bit RGB)
        # If strict scientific 16-bit retention is needed, PNG works but 'convert("RGB")' downscales to 8-bit.
        # For ResNet training, 8-bit RGB is standard.
        img = img.convert("RGB")
        
        buf = io.BytesIO()
        if target_format.upper() == "JPEG":
            img.save(buf, format="JPEG", quality=quality)
        else:
            # PNG (optimize=True makes it smaller but slower to write, usually worth it)
            img.save(buf, format="PNG", optimize=True)
            
        return buf.getvalue()

def write_h5(files: List[Path], labels: List[int], class_to_idx: Dict[str,int],
             root: Path, out_path: Path, target_format: str):
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(files)
    if N == 0:
        return

    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    
    # We do NOT use HDF5 compression (LZF) here because PNG/JPEG are already compressed.
    # Double compression wastes CPU.
    
    with h5py.File(out_path, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8)
        dset_labels = f.create_dataset("labels", data=np.array(labels, dtype=np.int32))

        rel_fnames = []
        for p in files:
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = p.name
            rel_fnames.append(rel)
        
        f.attrs["filenames"] = json.dumps(rel_fnames)
        f.attrs["class_to_idx"] = json.dumps(class_to_idx)
        # Store metadata about what we did
        f.attrs["original_format"] = "TIFF"
        f.attrs["stored_format"] = target_format.upper()

        for i, p in enumerate(tqdm(files, desc=f"Writing {out_path.name} ({target_format})", unit="img")):
            # Transcode here
            img_bytes = transcode_image_to_bytes(p, target_format=target_format)
            
            # Store bytes
            dset_imgs[i] = np.frombuffer(img_bytes, dtype=np.uint8)
            
        f.flush()

    print(f"✅ Wrote {out_path} ({N} images). Format: {target_format}")

def main(args):
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source root {src} does not exist")

    train_full, train_base = read_list_file(Path(args.train_list)) if args.train_list else (set(), set())
    test_full, test_base = read_list_file(Path(args.test_list)) if args.test_list else (set(), set())

    files, labels, class_to_idx = find_all_image_files(src, exts_str=args.ext)
    print(f"Found {len(files)} images. Transcoding to {args.format.upper()}...")

    train_files, train_labels = select_split(files, labels, src, train_full, train_base)
    test_files, test_labels = select_split(files, labels, src, test_full, test_base)

    if args.train_list is None and args.test_list is None:
        raise SystemExit("No lists provided.")

    if args.train_list:
        write_h5(train_files, train_labels, class_to_idx, src, Path(args.out_train), args.format)
    if args.test_list:
        write_h5(test_files, test_labels, class_to_idx, src, Path(args.out_test), args.format)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/TreeSAT", help="root folder")
    ap.add_argument("--train-list", default="../data/TreeSAT/train_filenames.lst")
    ap.add_argument("--test-list", default="../data/TreeSAT/test_filenames.lst")
    ap.add_argument("--out-train", default="../data/TreeSAT/TreeSAT_train_raw.h5")
    ap.add_argument("--out-test", default="../data/TreeSAT/TreeSAT_test_raw.h5")
    ap.add_argument("--ext", default=".tif,.tiff")
    # New argument to control format
    ap.add_argument("--format", default="png", choices=["png", "jpeg"], 
                    help="Target format to store in HDF5. PNG is lossless/safer. JPEG is faster.")
    args = ap.parse_args()
    main(args)