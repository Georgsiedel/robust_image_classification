#!/usr/bin/env python3
"""
Convert ImageFolder (class-subfolders) into a single HDF5 file
containing raw JPEG bytes (compressed).
"""

import argparse
import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

def find_image_files(root):
    root = Path(root)
    # Sort for determinism
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    files = []
    labels = []
    
    # Extensions to look for (Explicitly include Uppercase for Linux case-sensitivity)
    exts = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP"
    ]
    
    for c in classes:
        class_dir = root / c
        class_idx = class_to_idx[c]
        
        # Gather all images in this class
        class_files = []
        for ext in exts:
            class_files.extend(sorted(class_dir.glob(ext)))
        
        # Sort ONCE after gathering all extensions to ensure deterministic order 
        # (e.g. if you have mixed .jpg and .JPEG in the same folder)
        class_files.sort()
        
        # Add to main lists
        for p in class_files:
            files.append(p)
            labels.append(class_idx)
            
    return files, labels, class_to_idx

def main(args):
    src = Path(args.src)
    out = Path(args.out)

    files, labels, class_to_idx = find_image_files(src)
    N = len(files)
    print(f"Found {N} images in {len(class_to_idx)} classes.")
    
    # HDF5 Variable Length Unsigned Int8 (for storing raw bytes)
    dt = h5py.vlen_dtype(np.uint8)

    with h5py.File(out, "w") as f:
        # Create datasets
        # Note: We usually don't need 'compression="lzf"' for JPEGs as they are already compressed
        dset_imgs = f.create_dataset("images", (N,), dtype=dt)
        dset_labels = f.create_dataset("labels", data=np.array(labels, dtype=np.int32))

        # Metadata
        f.attrs["filenames"] = json.dumps([str(p.relative_to(src)) for p in files])
        f.attrs["class_to_idx"] = json.dumps(class_to_idx)

        print(f"Writing to {out}...")
        for i, p in enumerate(tqdm(files)):
            # Read Raw Bytes (Fast, no decoding)
            with open(p, "rb") as img_f:
                binary_data = img_f.read()
            
            # Convert to numpy uint8 array (still raw bytes)
            dset_imgs[i] = np.frombuffer(binary_data, dtype=np.uint8)

    print(f"✅ Success. Saved to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/Casting-Product-Quality/test", help="ImageFolder root")
    ap.add_argument("--out", default="../data/Casting-Product-Quality/Casting-Product-Quality_test_raw.h5", help="Output .h5 file path")
    args = ap.parse_args()
    main(args)
