import sys
import math
import random
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np

# --- Path Resolution & Submodule Injection ---
CURRENT_DIR = Path(__file__).resolve().parent          # experiments/
REPO_ROOT = CURRENT_DIR.parent                         # repo_root/
SUBMODULE_ROOT = REPO_ROOT / "micro_ast"               # repo_root/micro_ast

if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

# Import MicroAST components
from micro_ast.net_microAST import Encoder as MicroASTEncoder, Decoder as MicroASTDecoder

# Import AdaIN components
from experiments.adaIN.model import vgg as adain_vgg, decoder as adain_decoder

# --- Shared Utilities ---
def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates channel-wise mean and standard deviation for AdaIN."""
    ndim = feat.ndim
    if ndim == 3:
        feat = feat.unsqueeze(0)
        
    N, C, _, _ = feat.size()
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    
    if ndim == 3:
        return feat_mean.squeeze(0), feat_std.squeeze(0)
    return feat_mean, feat_std


# --- Transforms ---
class NSTRectangularTransform(nn.Module):
    """
    Unified Batchwise AdaIN Style Transfer Module for rectangular/quadratic images.
    """
    def __init__(
        self,
        style_feats: torch.Tensor,
        vgg: nn.Module,
        decoder: nn.Module,
        *,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        probability: float = 0.5,
        patch_size: int = 224,
        overlap: int = 32,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.vgg = vgg.eval()
        self.decoder = decoder.eval()
        
        for p in self.vgg.parameters(): p.requires_grad_(False)
        for p in self.decoder.parameters(): p.requires_grad_(False)

        self.register_buffer("style_features", torch.as_tensor(style_feats, dtype=torch.float32), persistent=False)

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.probability = float(probability)
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)

        if mean is not None and std is not None:
            self.register_buffer("mean_buf", torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.register_buffer("std_buf", torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.use_normalization = True
        else:
            self.mean_buf = None
            self.std_buf = None
            self.use_normalization = False

    @classmethod
    def from_files(
        cls,
        *,
        style_feats_path: Union[str, Path],
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        probability: float = 0.5,
        patch_size: int = 224,
        overlap: int = 32,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> "NSTRectangularTransform":
        vgg = adain_vgg
        decoder = adain_decoder

        vgg.load_state_dict(torch.load(Path(encoder_path), map_location="cpu", weights_only=True))
        decoder.load_state_dict(torch.load(Path(decoder_path), map_location="cpu", weights_only=True))

        vgg = nn.Sequential(*list(vgg.children())[:31])
        style_feats_np = np.load(Path(style_feats_path))
        style_feats = torch.from_numpy(style_feats_np).to(dtype=torch.float32)

        return cls(
            style_feats, vgg, decoder,
            alpha_min=alpha_min, alpha_max=alpha_max, probability=probability,
            patch_size=patch_size, overlap=overlap, mean=mean, std=std
        )

    def adaptive_instance_normalization(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    @torch.no_grad()
    def style_transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        alpha = torch.empty((), device=content.device, dtype=content.dtype).uniform_(self.alpha_min, self.alpha_max)
        content_f = self.vgg(content)
        feat = self.adaptive_instance_normalization(content_f, style)
        feat = feat * alpha + content_f * (1.0 - alpha)
        return self.decoder(feat)

    @staticmethod
    def _blend_mask(i: int, j: int, num_h: int, num_w: int, patch_size: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if num_h == 1 and num_w == 1:
            return torch.ones((3, patch_size, patch_size), device=device, dtype=dtype)
            
        mask_h = torch.ones((patch_size, patch_size), device=device, dtype=dtype)
        mask_w = torch.ones((patch_size, patch_size), device=device, dtype=dtype)
        
        if i > 0: mask_h[:overlap, :] = torch.linspace(0, 1, overlap, device=device, dtype=dtype).unsqueeze(1)
        if i < num_h - 1: mask_h[-overlap:, :] = torch.linspace(1, 0, overlap, device=device, dtype=dtype).unsqueeze(1)
        if j > 0: mask_w[:, :overlap] = torch.linspace(0, 1, overlap, device=device, dtype=dtype).unsqueeze(0)
        if j < num_w - 1: mask_w[:, -overlap:] = torch.linspace(1, 0, overlap, device=device, dtype=dtype).unsqueeze(0)
            
        return (mask_h * mask_w).unsqueeze(0).repeat(3, 1, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single_image = x.ndim == 3
        if single_image: x = x.unsqueeze(0)

        if self.style_features.device != x.device:
            self.to(x.device)

        x = x.clone()
        device, dtype = x.device, x.dtype
        batch_size, channels, H, W = x.shape

        if self.use_normalization: x = x * self.std_buf + self.mean_buf

        ratio = int(math.floor(batch_size * self.probability + torch.rand((), device=device).item()))
        if ratio <= 0:
            if self.use_normalization: x = (x - self.mean_buf) / self.std_buf
            return x.squeeze(0) if single_image else x

        img_idx = torch.randperm(batch_size, device=device)[:ratio]
        style_idx = torch.randint(0, self.style_features.shape[0], (ratio,), device=device)

        selected = x[img_idx]
        was_grayscale = selected.shape[1] == 1
        if was_grayscale: selected = selected.repeat(1, 3, 1, 1)

        patch_size = self.patch_size
        stride = patch_size - self.overlap

        metas, patches, patches_per_image = [], [], []

        for r in range(selected.shape[0]):
            img = selected[r : r + 1]
            _, _, orig_H, orig_W = img.shape
            scale = float(patch_size) / float(min(orig_H, orig_W))
            new_H, new_W = int(round(orig_H * scale)), int(round(orig_W * scale))

            img_resized = F.resize(img, size=[new_H, new_W], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
            num_h = max(1, math.ceil((new_H - self.overlap) / stride))
            num_w = max(1, math.ceil((new_W - self.overlap) / stride))
            metas.append((int(img_idx[r]), int(orig_H), int(orig_W), new_H, new_W, num_h, num_w))

            cnt = 0
            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))
                    patches.append(img_resized[:, :, top : top + patch_size, left : left + patch_size].squeeze(0))
                    cnt += 1
            patches_per_image.append(cnt)

        patches_tensor = torch.stack(patches, dim=0)
        style_batch = torch.cat([
            self.style_features[sid].unsqueeze(0).repeat(c, 1, 1, 1) 
            for sid, c in zip(style_idx.tolist(), patches_per_image)
        ], dim=0)

        stylized_patches = self.style_transfer(patches_tensor, style_batch)

        proc_ptr = 0
        for meta in metas:
            img_idx_i, orig_H, orig_W, new_H, new_W, num_h, num_w = meta
            recon = torch.zeros((3, new_H, new_W), device=device, dtype=dtype)
            weight = torch.zeros_like(recon)

            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))
                    patch = stylized_patches[proc_ptr]
                    proc_ptr += 1

                    mask = self._blend_mask(i, j, num_h, num_w, patch_size, self.overlap, device, dtype)
                    recon[:, top : top + patch_size, left : left + patch_size] += patch * mask
                    weight[:, top : top + patch_size, left : left + patch_size] += mask

            recon = F.resize(recon / torch.clamp(weight, min=1e-5), size=[orig_H, orig_W], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
            if was_grayscale: recon = F.rgb_to_grayscale(recon, num_output_channels=1)
            x[img_idx_i] = recon

        if self.use_normalization: x = (x - self.mean_buf) / self.std_buf

        return x.squeeze(0) if single_image else x


class MicroASTAugmentation(nn.Module):
    """
    MicroAST Augmentation using a multivariate normal style distribution.
    """
    def __init__(
        self,
        style_feats_path: Union[str, Path],
        content_encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        device: torch.device,
        probability: float = 0.1,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        min_spatial_size: int = 224 
    ):
        super().__init__()
        self.device = device
        self.probability = probability
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_spatial_size = min_spatial_size

        self.content_encoder = MicroASTEncoder().to(self.device)
        self.decoder = MicroASTDecoder().to(self.device)

        self.content_encoder.load_state_dict(torch.load(Path(content_encoder_path), map_location=self.device, weights_only=True))
        
        dec_ckpt = torch.load(Path(decoder_path), map_location=self.device, weights_only=False)
        dec_state = dec_ckpt.get("state_dict", dec_ckpt)
        cleaned_dec_state = {k.replace("decoder.", ""): v for k, v in dec_state.items()}
        try:
            self.decoder.load_state_dict(cleaned_dec_state)
        except Exception:
            self.decoder.load_state_dict(dec_state)

        self.content_encoder.eval()
        self.decoder.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        if mean is not None and std is not None:
            self.register_buffer("img_mean", torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.register_buffer("img_std", torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1), persistent=False)
            self.use_normalization = True
        else:
            self.img_mean = None
            self.img_std = None
            self.use_normalization = False

        archive = np.load(Path(style_feats_path))
        mean_tensor = torch.from_numpy(archive["mean"]).to(dtype=torch.float32, device=self.device)
        cov_tensor = torch.from_numpy(archive["covariance"]).to(dtype=torch.float32, device=self.device)
        self.register_buffer("style_mean", mean_tensor, persistent=False)
        self.register_buffer("style_covariance", cov_tensor, persistent=False)
        
        self.distribution = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0:
            return x

        if self.style_mean.device != x.device or self.distribution is None:
            self.to(x.device)
            self.device = x.device  
            self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=self.style_mean, covariance_matrix=self.style_covariance
            )

        batch_size = x.size(0)
        num_to_stylize = int(batch_size * self.probability + random.random())
        if num_to_stylize > batch_size: num_to_stylize = batch_size
        if num_to_stylize <= 0: return x

        out = x.clone()
        indices = torch.randperm(batch_size, device=self.device)[:num_to_stylize]
        content_selected = x[indices]

        if self.use_normalization:
            content_selected = content_selected * self.img_std + self.img_mean

        orig_h, orig_w = content_selected.shape[2], content_selected.shape[3]
        min_side = min(orig_h, orig_w)
        
        requires_upscale = min_side < self.min_spatial_size
        if requires_upscale:
            scale_factor = self.min_spatial_size / min_side
            new_h, new_w = int(round(orig_h * scale_factor)), int(round(orig_w * scale_factor))
            content_selected = torch.nn.functional.interpolate(
                content_selected, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        samples = self.distribution.sample((num_to_stylize,))
        s0_mean, s0_std = samples[:, 0:64].view(num_to_stylize, 64, 1, 1), samples[:, 64:128].view(num_to_stylize, 64, 1, 1)
        s1_mean, s1_std = samples[:, 128:192].view(num_to_stylize, 64, 1, 1), samples[:, 192:256].view(num_to_stylize, 64, 1, 1)
        w0, w1 = samples[:, 256:320].view(num_to_stylize, 64, 1, 1), samples[:, 320:384].view(num_to_stylize, 64, 1, 1)
        b0, b1 = samples[:, 384:448].view(num_to_stylize, 64, 1, 1), samples[:, 448:512].view(num_to_stylize, 64, 1, 1)

        s0_col = torch.cat([s0_mean - s0_std, s0_mean + s0_std], dim=2)
        s0_dummy = torch.cat([s0_col, s0_col], dim=3)

        s1_col = torch.cat([s1_mean - s1_std, s1_mean + s1_std], dim=2)
        s1_dummy = torch.cat([s1_col, s1_col], dim=3)

        content_feats = self.content_encoder(content_selected)
        stylized_subset = self.decoder(
            content_feats, 
            [s0_dummy, s1_dummy], 
            [w0, w1], 
            [b0, b1], 
            alpha=random.uniform(self.alpha_min, self.alpha_max)
        )
        stylized_subset = torch.clamp(stylized_subset, 0.0, 1.0)

        if requires_upscale:
            stylized_subset = torch.nn.functional.interpolate(
                stylized_subset, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )

        if self.use_normalization:
            out[indices] = (stylized_subset - self.img_mean) / self.img_std
        else:
            out[indices] = stylized_subset
            
        return out