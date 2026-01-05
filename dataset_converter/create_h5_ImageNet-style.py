#!/usr/bin/env python3
"""
Convert ImageFolder (class-subfolders) into a single HDF5 file
with variable-size NumPy arrays stored as bytes (lazy-loadable).

Features:
- Supports variable-size images (no pre-resizing needed)
- Lazy loading via h5py
- Optional LZF compression
- Only processes JPEG files (*.JPEG, *.jpg, *.jpeg)
"""

import argparse
import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm

def find_image_files(root):
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    files, labels = [], []
    for c in classes:
        class_dir = root / c
        # Only JPEGs
        for p in sorted(class_dir.glob("*.jpg")):
            files.append(p)
            labels.append(class_to_idx[c])
        for p in sorted(class_dir.glob("*.jpeg")):
            files.append(p)
            labels.append(class_to_idx[c])
        for p in sorted(class_dir.glob("*.png")):
            files.append(p)
            labels.append(class_to_idx[c])
    return files, labels, class_to_idx

def image_to_npy_bytes(path):
    """Open image, convert to RGB numpy array, serialize as .npy bytes and return bytes."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)  # H,W,3 uint8
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def main(args):
    src = Path(args.src)
    out = Path(args.out)

    files, labels, class_to_idx = find_image_files(src)
    N = len(files)
    print(f"Found {N} images, {len(class_to_idx)} classes.")

    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    compression = "lzf" if args.compression else None

    with h5py.File(out, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8, compression=compression)
        dset_labels = f.create_dataset("labels", data=np.array(labels, dtype=np.int32))

        # store filenames for reference
        f.attrs["filenames"] = json.dumps([str(p.relative_to(src)) for p in files])
        f.attrs["class_to_idx"] = json.dumps(class_to_idx)

        for i, p in enumerate(tqdm(files, desc="Converting")):
            b = image_to_npy_bytes(p)
            arr = np.frombuffer(b, dtype=np.uint8)  # convert bytes -> 1D uint8 array
            dset_imgs[i] = arr

        f.flush()

    print(f"✅ Wrote {out} ({N} images).")
    print(f"Compression used: {compression}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/Casting-Product-Quality/test", help="ImageFolder root")
    ap.add_argument("--out", default="../data/Casting-Product-Quality/test.h5", help="Output HDF5 file path")
    ap.add_argument("--compression", action="store_true", help="Use LZF compression for the dataset (fast).")
    args = ap.parse_args()
    main(args)
