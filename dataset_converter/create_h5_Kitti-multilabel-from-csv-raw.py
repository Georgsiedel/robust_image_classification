#!/usr/bin/env python3
"""
Convert a flat folder of PNGs + a CSV of multi-labels into a single HDF5 file.

Stores RAW image bytes (PNG) directly.
Compatible with HDF5ImageDataset using torchvision.io.decode_image.

CSV format expected (per-line examples):
  000000.txt,0,0,1,1,0
  000001.png,1,0,0,0,0
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import sys

DEFAULT_LABEL_NAMES = [
    "non_vulnerable_present",
    "non_vulnerable_nearby",
    "vulnerable_present",
    "vulnerable_nearby",
    "crowded_critical",
]

def parse_csv_labels(csv_path: Path, expected_num_labels: int = None):
    """
    Read CSV and return dict: basename_no_ext -> list[int] (binary label vector).
    """
    mapping = {}
    with csv_path.open("r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # strip whitespace and ignore empty cells
            row = [c.strip() for c in row if c is not None and c.strip() != ""]
            if len(row) < 2:
                continue
            name = row[0]
            # If the name contains a path, take the stem
            stem = Path(name).stem
            # parse label columns as ints
            try:
                labels = [int(float(x)) for x in row[1:]]
            except Exception:
                continue
            
            if expected_num_labels is not None and len(labels) != expected_num_labels:
                raise ValueError(f"Row for {name} has {len(labels)} labels but expected {expected_num_labels}")
            mapping[stem] = labels
    return mapping

def main():
    ap = argparse.ArgumentParser(description="Convert flat PNG folder + CSV labels -> single HDF5 (raw bytes).")
    ap.add_argument("--images-dir", default="../data/KITTI_Distance_Multiclass/train/", help="Directory containing PNG images (flat).")
    ap.add_argument("--labels-csv", default="../data/KITTI_Distance_Multiclass/multilabel_annotations.csv", help="CSV file with per-image multi-labels.")
    ap.add_argument("--out-h5", default="../data/KITTI_Distance_Multiclass/KITTI_Distance_Multiclass_raw.h5", help="Output HDF5 filepath.")
    ap.add_argument("--ext", default=".png", help="Image extension to look for (default '.png').")
    ap.add_argument("--label-names", "-n", default=None,
                    help="Comma-separated label names or path to JSON. "
                         f"Default: {','.join(DEFAULT_LABEL_NAMES)}")
    # Note: Compression is usually not needed for raw PNG/JPEG bytes as they are already compressed, 
    # but LZF is fast and harmless if you want to keep it.
    ap.add_argument("--compression", action="store_true", help="Use LZF compression for the images dataset.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    csv_path = Path(args.labels_csv)
    out_h5 = Path(args.out_h5)
    ext = args.ext if args.ext.startswith('.') else '.' + args.ext

    if not images_dir.exists():
        raise SystemExit(f"Images dir {images_dir} does not exist")
    if not csv_path.exists():
        raise SystemExit(f"CSV labels file {csv_path} does not exist")

    # Load label names
    if args.label_names:
        ln = args.label_names
        try:
            p = Path(ln)
            if p.exists():
                label_names = json.loads(p.read_text())
            else:
                label_names = [s.strip() for s in ln.split(",") if s.strip()]
        except Exception:
            raise SystemExit("Failed to parse --label-names (expect JSON file or comma-separated list)")
    else:
        label_names = DEFAULT_LABEL_NAMES.copy()

    num_labels = len(label_names)
    mapping = parse_csv_labels(csv_path, expected_num_labels=num_labels)
    if not mapping:
        raise SystemExit(f"No valid rows parsed from {csv_path}")

    # Match files to CSV entries
    found_files = []
    multilabels = []
    basenames = []
    missing = []

    print(f"Matching {len(mapping)} CSV entries to files in {images_dir}...")

    for stem, labels in mapping.items():
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            found_files.append(candidate)
            multilabels.append(labels)
            basenames.append(candidate.name)
        else:
            # Fallback: look for any file with this stem
            found_any = None
            for p in images_dir.iterdir():
                if p.is_file() and p.stem == stem:
                    found_any = p
                    break
            if found_any:
                found_files.append(found_any)
                multilabels.append(labels)
                basenames.append(found_any.name)
            else:
                missing.append(stem)

    if missing:
        print(f"[warning] {len(missing)} CSV entries did not match any image file (skipped).", file=sys.stderr)

    N = len(found_files)
    if N == 0:
        raise SystemExit("No images found to write.")

    # Prepare HDF5
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    compression = "lzf" if args.compression else None

    # Convert labels to uint8 numpy array (NxC)
    multilabels_arr = np.array(multilabels, dtype=np.uint8)

    with h5py.File(out_h5, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8, compression=compression)
        dset_labels = f.create_dataset("labels", data=multilabels_arr)

        # Attributes
        f.attrs["filenames"] = json.dumps(basenames)
        f.attrs["label_names"] = json.dumps(label_names)
        f.attrs["multilabel_format"] = "binary_matrix"

        # Write RAW BYTES
        for i, p in enumerate(tqdm(found_files, desc=f"Writing {out_h5.name}", unit="img")):
            with open(p, "rb") as img_f:
                binary_data = img_f.read()
            
            # Store as numpy uint8 array of bytes
            dset_imgs[i] = np.frombuffer(binary_data, dtype=np.uint8)

    print(f"✅ Wrote {out_h5} ({N} images).")

if __name__ == "__main__":
    main()