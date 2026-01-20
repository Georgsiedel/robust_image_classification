import os
import sys
current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

from run_0 import device

########################################################################################################
### Noise2Net
########################################################################################################
# based on DeepAugment method here: https://github.com/hendrycks/imagenet-r/blob/master/DeepAugment/train_noise2net.py
# Paper Hendrycks et al: "The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization"

import sys
import os
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math

class N2N_DeepAugment(nn.Module):
    def __init__(self, orig_batch_size, image_size, channels, noisenet_max_eps=0.75, ratio=0.5, overlap=32):
        super(N2N_DeepAugment, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.ratio = ratio
        self.overlap = overlap
        self.noisenet_max_eps = noisenet_max_eps

        # initial guess; can be updated dynamically
        self.noise2net_batch_size = max(1, int(orig_batch_size * ratio))
        self.noise2net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=self.noise2net_batch_size).train()
        # note: network will be moved to device on-first-use via .to(device)

        # patching constants (match style-transfer)
        self.patch_size = 224
        self.stride = self.patch_size - self.overlap

    def _use_net_for_chunk(self, chunk: torch.Tensor, device: torch.device):
        """
        chunk: [M, C, 224,224] already on device
        Return: [M, C, 224,224] on device (augmented)
        Uses the configured self.noise2net if M == self.noise2net_batch_size, otherwise
        creates a temporary Res2Net for M.
        NOTE: chunk should be detached by caller (no grad).
        """
        M = chunk.shape[0]
        C = chunk.shape[1]
        assert C == self.channels, f"Channel mismatch {C} vs {self.channels}"

        if M == 0:
            return chunk

        # Ensure the net(s) are on the right device
        if M == self.noise2net_batch_size:
            net = self.noise2net
            if next(net.parameters()).device != device:
                net = net.to(device)
            net.reload_parameters()
            net.set_epsilon(random.uniform(self.noisenet_max_eps / 2.0, self.noisenet_max_eps))
            # chunk is expected to be detached already
            inp = chunk.reshape(1, M * C, self.patch_size, self.patch_size)
            out = net(inp)
            out = out.reshape(M, C, self.patch_size, self.patch_size)
            return out
        else:
            temp_net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=M).train().to(device)
            temp_net.reload_parameters()
            temp_net.set_epsilon(random.uniform(self.noisenet_max_eps / 2.0, self.noisenet_max_eps))
            inp = chunk.reshape(1, M * C, self.patch_size, self.patch_size)
            out = temp_net(inp)
            out = out.reshape(M, C, self.patch_size, self.patch_size)
            try:
                del temp_net
            except Exception:
                pass
            return out

    def forward(self, bx):
        """
        bx: [B, C, H, W]
        All augmentation (including writes into bx) performed under torch.no_grad()
        to avoid creating autograd history and to prevent inplace-on-view errors.
        """

        batchsize = bx.shape[0]
        device = bx.device
        C = bx.shape[1]

        if C != self.channels:
            raise ValueError(f"Input channels {C} != configured channels {self.channels}")

        target_count = int(batchsize * self.ratio)
        if target_count <= 0:
            return bx

        with torch.no_grad():
            # indices to augment
            indices = torch.randperm(batchsize, device=device)[:target_count].tolist()

            # assume all images have identical shape
            _, H0, W0 = bx[indices[0]].shape
            all_same_shape = True
            for idx in indices:
                _, H, W = bx[idx].shape
                if H != H0 or W != W0:
                    all_same_shape = False
                    break

            # ==============================================================
            # FAST + SEMI-FAST PATHS (square images, single resize assumption)
            # ==============================================================
            if all_same_shape and H0 == W0:
                selected_x = bx[indices].to(device)  # [N, C, H0, H0]
                N = selected_x.shape[0]

                # --------------------------
                # Fast: already patch_size
                # --------------------------
                if H0 == self.patch_size:
                    out_chunks = []
                    ptr = 0
                    while ptr < N:
                        chunk_size = min(self.noise2net_batch_size, N - ptr)
                        chunk = selected_x[ptr:ptr + chunk_size].detach()
                        out_chunk = self._use_net_for_chunk(chunk, device)
                        out_chunks.append(out_chunk)
                        ptr += chunk_size

                    out = torch.cat(out_chunks, dim=0)
                    bx[indices] = out.to(bx.dtype)
                    return bx

                # --------------------------
                # Semi-fast: square resize
                # --------------------------
                selected_resized = F.interpolate(
                    selected_x,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )

                out_chunks = []
                ptr = 0
                while ptr < N:
                    chunk_size = min(self.noise2net_batch_size, N - ptr)
                    chunk = selected_resized[ptr:ptr + chunk_size].detach()
                    out_chunk = self._use_net_for_chunk(chunk, device)
                    out_chunks.append(out_chunk)
                    ptr += chunk_size

                out = torch.cat(out_chunks, dim=0)

                out_back = F.interpolate(
                    out,
                    size=(H0, W0),
                    mode="bilinear",
                    align_corners=False,
                )

                bx[indices] = out_back.to(bx.dtype)
                return bx

            # ==============================================================
            # SLOW PATH: non-square images (original DeepAugment logic)
            # ==============================================================
            selected_x = bx[indices].cpu()  # [R, C, H, W] on CPU
            metas_selected = []
            patches_to_process = []
            selected_image_order = indices[:]

            for r, img in enumerate(selected_x):
                _, H, W = img.shape
                orig_H, orig_W = int(H), int(W)

                scale = 224.0 / float(min(H, W))
                new_H = int(round(H * scale))
                new_W = int(round(W * scale))

                img_resized = TF.resize(
                    img.unsqueeze(0),
                    size=[new_H, new_W],
                    interpolation=TF.InterpolationMode.BILINEAR,
                ).squeeze(0)

                num_h = max(1, math.ceil((new_H - self.overlap) / self.stride))
                num_w = max(1, math.ceil((new_W - self.overlap) / self.stride))
                num_patches = int(num_h * num_w)

                metas_selected.append(
                    (selected_image_order[r], orig_H, orig_W,
                    new_H, new_W, int(num_h), int(num_w), num_patches)
                )

                for i in range(num_h):
                    for j in range(num_w):
                        top = min(int(i * self.stride), max(0, new_H - self.patch_size))
                        left = min(int(j * self.stride), max(0, new_W - self.patch_size))
                        patch = img_resized[:, top:top + self.patch_size,
                                            left:left + self.patch_size]
                        patches_to_process.append(patch)

            if len(patches_to_process) == 0:
                return bx

            patches_tensor_cpu = torch.stack(patches_to_process, dim=0)
            N_proc = patches_tensor_cpu.shape[0]

            processed_patches = []
            ptr = 0
            while ptr < N_proc:
                chunk_size = min(self.noise2net_batch_size, N_proc - ptr)
                chunk = patches_tensor_cpu[ptr:ptr + chunk_size].to(device).detach()
                out_chunk = self._use_net_for_chunk(chunk, device)
                processed_patches.append(out_chunk.cpu())
                ptr += chunk_size

            stylized_patches = torch.cat(processed_patches, dim=0)

            # --------------------------------------------------------------
            # Reconstruction
            # --------------------------------------------------------------
            proc_ptr = 0
            for meta in metas_selected:
                img_idx, orig_H, orig_W, new_H, new_W, num_h, num_w, _ = meta

                recon = torch.zeros((self.channels, new_H, new_W), dtype=torch.float32)
                weight = torch.zeros_like(recon)

                for i in range(num_h):
                    for j in range(num_w):
                        top = min(int(i * self.stride), max(0, new_H - self.patch_size))
                        left = min(int(j * self.stride), max(0, new_W - self.patch_size))

                        patch = stylized_patches[proc_ptr]
                        proc_ptr += 1

                        mask_y = torch.linspace(0, 1, self.patch_size).unsqueeze(1).repeat(1, self.patch_size)
                        mask_x = torch.linspace(0, 1, self.patch_size).unsqueeze(0).repeat(self.patch_size, 1)

                        mask_h = torch.ones_like(mask_y)
                        mask_w = torch.ones_like(mask_x)

                        if i > 0:
                            mask_h[:self.overlap, :] = torch.linspace(0, 1, self.overlap).unsqueeze(1)
                        if i < num_h - 1:
                            mask_h[-self.overlap:, :] = torch.linspace(1, 0, self.overlap).unsqueeze(1)

                        if j > 0:
                            mask_w[:, :self.overlap] = torch.linspace(0, 1, self.overlap).unsqueeze(0)
                        if j < num_w - 1:
                            mask_w[:, -self.overlap:] = torch.linspace(1, 0, self.overlap).unsqueeze(0)

                        mask = (mask_h * mask_w).unsqueeze(0).repeat(self.channels, 1, 1)

                        recon[:, top:top + self.patch_size,
                            left:left + self.patch_size] += patch * mask
                        weight[:, top:top + self.patch_size,
                            left:left + self.patch_size] += mask

                recon = recon / torch.clamp(weight, min=1e-5)

                recon_back = TF.resize(
                    recon.unsqueeze(0),
                    size=[orig_H, orig_W],
                    interpolation=TF.InterpolationMode.BILINEAR,
                ).squeeze(0)

                bx[img_idx] = recon_back.to(device).to(bx.dtype)

            if proc_ptr != stylized_patches.shape[0]:
                raise RuntimeError("Not all processed patches were consumed.")

        return bx


class N2N_DeepAugment_w_o_rectangular(nn.Module):
    def __init__(self, orig_batch_size, image_size, channels, noisenet_max_eps=0.75, ratio=0.5):
        super(N2N_DeepAugment_w_o_rectangular, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.ratio = ratio
        self.noise2net_batch_size = int(orig_batch_size * ratio)
        self.noise2net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=self.noise2net_batch_size).train().to(device)
        self.noisenet_max_eps = noisenet_max_eps

    def forward(self, bx):
        batchsize = bx.shape[0]

        if self.noise2net_batch_size != int(batchsize * self.ratio): #last batch of an epoch may have different bs
            self.noise2net_batch_size = int(batchsize * self.ratio)
            self.noise2net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=self.noise2net_batch_size).train().to(device)
        
        with torch.no_grad():
            # Setup network
            self.noise2net.reload_parameters()
            self.noise2net.set_epsilon(random.uniform(self.noisenet_max_eps / 2.0, self.noisenet_max_eps))
            
            # Apply aug on a random subset according to ratio
            indices = torch.randperm(batchsize)[:self.noise2net_batch_size]

            bx_auged = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)(bx[indices])
            bx_auged = bx_auged.reshape((1, self.noise2net_batch_size * self.channels, 224, 224))
            bx_auged = self.noise2net(bx_auged)
            bx_auged = bx_auged.reshape((self.noise2net_batch_size, self.channels, 224, 224))
            bx_auged = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)(bx_auged)
            bx[indices] = bx_auged

        return bx

class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, hidden_planes=9, scale = 4, batch_size=5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (multiply by batch_size)
            planes: output channel dimensionality (multiply by batch_size)
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes * batch_size
        self.conv1 = nn.Conv2d(inplanes * batch_size, width*scale, kernel_size=1, bias=False, groups=batch_size)
        self.bn1 = nn.BatchNorm2d(width*scale)

        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3])
            D = random.choice([1, 2, 3])

            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True, groups=batch_size))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * batch_size, kernel_size=1, bias=False, groups=batch_size)
        self.bn3 = nn.BatchNorm2d(planes * batch_size)

        self.act = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.hidden_planes = hidden_planes
        self.batch_size = batch_size

    def forward(self, x):
        _, _, H, W = x.shape
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out) # [1, hidden_planes*batch_size*scale, H, W]
        
        # Hack to make different scales work with the hacky batches
        out = out.view(1, self.batch_size, self.scale, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        
        spx = torch.split(out, self.width, 1) # [ ... (1, hidden_planes*batch_size, H, W) ... ]
        
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        
        # Undo hack to make different scales work with the hacky batches
        out = out.view(1, self.scale, self.batch_size, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=16, batch_size=5):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)

    def reload_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
 
    def set_epsilon(self, new_eps):
        self.epsilon = new_eps

    def forward_original(self, x):                
        x = (self.block1(x) * self.epsilon) + x
        x = (self.block2(x) * self.epsilon) + x
        x = (self.block3(x) * self.epsilon) + x
        x = (self.block4(x) * self.epsilon) + x
        return x

    def forward(self, x):
        return self.forward_original(x)