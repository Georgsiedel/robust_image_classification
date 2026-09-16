import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

CURRENT_DIR = Path(__file__).resolve().parent    # style_detection/
SUBMODULE_ROOT = CURRENT_DIR / "MicroAST"       # style_detection/micro_ast/

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

from MicroAST.net_microAST import Encoder, Modulator

def extract_channel_stats(feat: torch.Tensor):
    # Computes channel-wise mean and standard deviation vectors
    mean = feat.mean(dim=(2, 3)) # [1, 64]
    std = feat.std(dim=(2, 3), unbiased=False) + 1e-5 # [1, 64]
    return mean, std

def precompute_style_distribution(image_dir: str, output_npz: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    style_encoder = Encoder().to(device).eval()
    modulator = Modulator().to(device).eval()

    models_dir = SUBMODULE_ROOT / "models"
    style_encoder.load_state_dict(torch.load(models_dir / "style_encoder_iter_160000.pth.tar", map_location=device))
    modulator.load_state_dict(torch.load(models_dir / "modulator_iter_160000.pth.tar", map_location=device))

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [Path(image_dir) / f for f in os.listdir(image_dir) if Path(f).suffix.lower() in img_extensions]
    print(f"Extracting style statistics from {len(img_paths)} images...")

    all_style_vectors = []

    with torch.no_grad():
        for idx, path in enumerate(img_paths):
            if (idx % 10==0):
                print('Image number:', idx)
            try:
                img = Image.open(path).convert("RGB")
                tensor = img_transform(img).unsqueeze(0).to(device)

                s = style_encoder(tensor) # [s0, s1]
                w, b = modulator(s)       # ([w0, w1], [b0, b1])

                # Collapse spatial details into channel statistics
                s0_mean, s0_std = extract_channel_stats(s[0])
                s1_mean, s1_std = extract_channel_stats(s[1])

                # Reshape modulator features [1, 64, 1, 1] -> [1, 64]
                w0, w1 = w[0].view(1, -1), w[1].view(1, -1)
                b0, b1 = b[0].view(1, -1), b[1].view(1, -1)

                # Concatenate all 8 statistics arrays into a single 512-dim vector
                style_vector = torch.cat([s0_mean, s0_std, s1_mean, s1_std, w0, w1, b0, b1], dim=1)
                all_style_vectors.append(style_vector.cpu().numpy())

            except Exception as e:
                print(f"Skipping corrupt asset {path.name}: {e}")

    # Convert to a single large matrix [Num_Images, 512]
    data_matrix = np.concatenate(all_style_vectors, axis=0)

    # Compute continuous Multivariate Distribution metrics
    mean_vector = np.mean(data_matrix, axis=0)
    covariance_matrix = np.cov(data_matrix, rowvar=False)

    # Add a tiny diagonal ridge stabilization factor to ensure mathematical positive-definiteness
    covariance_matrix += 1e-5 * np.eye(512)

    np.savez_compressed(output_npz, mean=mean_vector, covariance=covariance_matrix)
    print(f"Successfully baked style distribution package to: {output_npz}")

if __name__ == "__main__":
    precompute_style_distribution("../datasets/painter-by-numbers-train-1", "../datasets/style_distribution.npz")