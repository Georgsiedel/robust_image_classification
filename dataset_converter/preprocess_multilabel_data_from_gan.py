#!/usr/bin/env python3
"""
convert_gan_to_multilabel_layout.py

Given:
  - a dataset name (so we can load the original dataset and build the CompactMultilabelEncoder)
  - a GAN images folder in ImageFolder layout where class folders are numeric compact IDs
Produce:
  - a subfolder inside the GAN folder containing the converted ImageFolder layout where
    class folder names are binary multilabel strings like "0_0_1_1".
  - dataset.json (NVIDIA style) inside the converted folder
  - class_mapping.json (human readable) inside the converted folder
"""

import argparse
import os
import sys
import json
import shutil
from tqdm import tqdm

# Put parent dir on sys.path so experiments.* imports work (same as your helper code)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Paste or import your helper utilities here ---
# The helper code you gave included:
# - CompactMultilabelEncoder
# - load_user_dataset(...)  (depends on experiments.data etc.)
#
# For clarity we import them as if they are available in the same package.
# If you only have them inline, paste their definitions above this script.
from preprocess_data_for_gan import CompactMultilabelEncoder, load_user_dataset
# Replace 'your_module_with_helpers' with the module path where you put the helper code.
# Alternatively, if the helper classes are in the same file, remove the import and keep the definitions.

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTS)

def convert_layout(dataset_name: str, gan_src_dir: str, out_subdir_name: str = "multilabel_converted", copy_files: bool = True):
    """
    Convert GAN images layout (numeric folder names) to multilabel folder names.
    - dataset_name: dataset used to build encoder (passed to load_user_dataset)
    - gan_src_dir: folder containing numeric subfolders (ImageFolder layout)
    - out_subdir_name: name of subfolder to create inside gan_src_dir for converted layout
    - copy_files: if False, attempt to move files (default True => copy)
    """
    if not os.path.isdir(gan_src_dir):
        raise FileNotFoundError(f"GAN source directory not found: {gan_src_dir}")

    # Load original dataset to build mapping (this may be heavy but required to recover mapping)
    print(f"Loading dataset '{dataset_name}' to build multilabel encoder...")
    user_trainset, image_size, multilabel, class_to_idx = load_user_dataset(dataset_name)

    if not multilabel:
        raise ValueError(f"Dataset '{dataset_name}' is not multilabel according to load_user_dataset(). Aborting.")

    num_binary_classes = class_to_idx
    print(f"Detected multilabel dataset with {num_binary_classes} binary classes. Building encoder...")
    id_encoder = CompactMultilabelEncoder(user_trainset, num_binary_classes)

    # Prepare output directory
    out_root = os.path.join(gan_src_dir, out_subdir_name)
    os.makedirs(out_root, exist_ok=True)
    print(f"Output will be saved to: {out_root}")

    # We'll write NVIDIA dataset.json inside out_root
    dataset_json_labels = []

    # Build mapping from numeric id -> multilabel string
    class_id_to_multilabel_str = {}

    # Walk numeric-class subfolders in gan_src_dir
    # Only consider immediate child directories (ImageFolder layout)
    subfolders = [d for d in sorted(os.listdir(gan_src_dir)) if os.path.isdir(os.path.join(gan_src_dir, d))]
    # Remove the output folder itself if found among children (avoid reprocessing)
    subfolders = [s for s in subfolders if s != out_subdir_name]

    # Validate that subfolder names are numeric (GAN produced compact ids)
    numeric_folders = []
    for folder in subfolders:
        try:
            int(folder)
            numeric_folders.append(folder)
        except ValueError:
            # ignore non-numeric folder (could be stray files)
            print(f"Skipping non-numeric folder in GAN folder: {folder}")

    if not numeric_folders:
        raise RuntimeError("No numeric class subfolders found in GAN folder. Expecting ImageFolder layout with numeric folder names.")

    # For each numeric folder, compute destination multilabel folder name
    for folder in numeric_folders:
        class_id = int(folder)
        if class_id not in id_encoder.reverse_mapping:
            raise KeyError(f"Class ID {class_id} (GAN folder '{folder}') not found in encoder reverse mapping. "
                           "Make sure the encoder built from your original dataset contains this ID.")
        multilabel_tensor = id_encoder.id_to_multilabel(class_id)  # torch tensor of shape (num_binary_classes,)
        # Format e.g. "0_0_1_1"
        multilabel_list = [str(int(x)) for x in multilabel_tensor.tolist()]
        multilabel_str = "_".join(multilabel_list)
        class_id_to_multilabel_str[class_id] = multilabel_str

    # Copy files
    total_files = 0
    for folder in numeric_folders:
        folder_path = os.path.join(gan_src_dir, folder)
        files = [f for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f)) and is_image_file(f)]
        total_files += len(files)
    print(f"Found {len(numeric_folders)} numeric class folders with {total_files} image files total.")

    pbar = tqdm(total=total_files, desc="Converting images")
    for folder in numeric_folders:
        class_id = int(folder)
        multilabel_str = class_id_to_multilabel_str[class_id]
        src_folder = os.path.join(gan_src_dir, folder)
        dst_folder = os.path.join(out_root, multilabel_str)
        os.makedirs(dst_folder, exist_ok=True)

        # copy/move each image file
        for fname in sorted(os.listdir(src_folder)):
            src_path = os.path.join(src_folder, fname)
            if not os.path.isfile(src_path) or not is_image_file(fname):
                continue

            # Destination filename keeps original name
            dst_path = os.path.join(dst_folder, fname)
            if copy_files:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            # Add to dataset.json list (relative to out_root)
            relative_path = f"{multilabel_str}/{fname}"
            dataset_json_labels.append([relative_path, class_id])

            pbar.update(1)
    pbar.close()

    # Save dataset.json inside out_root
    dataset_json_path = os.path.join(out_root, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump({"labels": dataset_json_labels}, f)
    print(f"Saved NVIDIA-style dataset.json -> {dataset_json_path}")

    # Save class mapping: numeric id -> multilabel_str and also numeric id -> list
    mapping = {}
    mapping_lists = {}
    for class_id, multilabel_str in class_id_to_multilabel_str.items():
        mapping[str(class_id)] = multilabel_str
        # also save list of ints
        multilabel_tensor = id_encoder.id_to_multilabel(class_id)
        mapping_lists[str(class_id)] = [int(x) for x in multilabel_tensor.tolist()]

    mapping_json_path = os.path.join(out_root, "class_mapping.json")
    with open(mapping_json_path, "w") as f:
        json.dump({"id_to_multilabel_str": mapping, "id_to_multilabel_list": mapping_lists}, f, indent=2)
    print(f"Saved class mapping -> {mapping_json_path}")

    print("Conversion complete.")
    return out_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GAN ImageFolder numeric labels -> multilabel string folder names.")
    parser.add_argument("--dataset_name", type=str, default='WaferMap',
                        help="Name of the dataset used to build encoder (passed to load_user_dataset).")
    parser.add_argument("--gan_src_dir", type=str, default='../data/WaferMap_GAN',
                        help="Path to GAN images folder (ImageFolder layout with numeric class subfolders).")
    parser.add_argument("--out_subdir", type=str, default="multilabel_converted",
                        help="Name of subfolder to create inside gan_src_dir for converted layout.")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="Copy files (default). If you prefer moving files set --copy False in code.")
    args = parser.parse_args()

    # Note: argparse doesn't allow boolean flags easily; keep copy True by default. To move, set copy_files in call below.
    convert_layout(args.dataset_name, args.gan_src_dir, args.out_subdir, copy_files=True)
