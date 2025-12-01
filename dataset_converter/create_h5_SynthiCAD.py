#!/usr/bin/env python3
"""
Convert a flat folder of images with class names embedded in filenames
(e.g. adapter_plate_triangular000000.jpg -> class "adapter_plate_triangular")
into a single HDF5 file with variable-size NumPy arrays stored as bytes.

Features:
- Supports variable-size images (no pre-resizing needed)
- Lazy loading via h5py
- Optional LZF compression
- Only processes JPEG/PNG files (*.jpg, *.jpeg, *.png) (case-insensitive)
- Extracts class name by stripping the trailing numeric ID before the extension.
  If no trailing digits are found, the filename stem is used as class name.
"""
import argparse
import io
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def extract_class_from_filename(path: Path) -> str:
    """
    Given a Path like 'adapter_plate_triangular000000.jpg',
    returns 'adapter_plate_triangular' by removing trailing digits just before the extension.
    If no trailing digits are found, return the stem (filename without extension).
    """
    name = path.name
    # match: (anything minimal)(one or more digits).extension
    m = re.match(r"^(.+?)(\d+)(\.[^.]+)$", name)
    if m:
        return m.group(1)
    # fallback: use the stem (no extension)
    return path.stem

def find_image_files_flat(root):
    """
    Finds image files in a single folder (non-recursive).
    Returns: (files_list_sorted, labels_list, class_to_idx)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {root}")

    # gather image files in the directory (non-recursive)
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if len(files) == 0:
        raise RuntimeError(f"No image files with extensions {VALID_EXTS} found in {root}")

    # extract classes from filenames
    classes = [extract_class_from_filename(p) for p in files]
    unique_classes = sorted(set(classes))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    labels = [class_to_idx[c] for c in classes]
    return files, labels, class_to_idx

def image_to_npy_bytes(path):
    """Open image, convert to RGB numpy array, serialize as .npy bytes and return bytes."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)  # H,W,3 uint8
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def main(args):
    src = Path(args.src)
    out = Path(args.out)

    files, labels, class_to_idx = find_image_files_flat(src)
    N = len(files)
    print(f"Found {N} images, {len(class_to_idx)} classes.")

    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    compression = "lzf" if args.compression else None

    with h5py.File(out, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8, compression=compression)
        dset_labels = f.create_dataset("labels", data=np.array(labels, dtype=np.int32))

        # store filenames for reference (relative to src)
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
    ap.add_argument("--src", default="../data/SynthiCAD/Valid_Dataset/images", help="ImageFolder root")
    ap.add_argument("--out", default="../data/SynthiCAD/SynthiCAD_valid.h5", help="Output HDF5 file path")
    ap.add_argument("--compression", action="store_true", help="Use LZF compression for the dataset (fast).")
    args = ap.parse_args()
    main(args)
