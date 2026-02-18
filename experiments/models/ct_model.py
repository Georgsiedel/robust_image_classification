from experiments.utils import plot_images
import torch.nn as nn
import torch
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_noise, noise_up
from experiments.data import normalization_values
from experiments.deepaugment_n2n import N2N_DeepAugment


class CtModel(nn.Module):
    def __init__(self, dataset, normalized, num_classes):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.num_classes = num_classes
        self.dataset = dataset
        self.mean, self.std = normalization_values(
            batch=None,
            dataset=dataset,
            normalized=normalized,
            manifold=False,
            manifold_factor=1,
            verbose=True
        )
        if normalized:
            self.register_buffer("mu", self.mean)
            self.register_buffer("sigma", self.std)

        self.deepaugment_instance = None
        
    def forward_handle_greyscale(self, x):
        if x.shape[1] == 1:  # Grayscale → RGB
            return x.repeat(1, 3, 1, 1)
        else:
            return x

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x
    
    def noise_mixup(self, out, targets, robust_samples, corruptions, mixup_alpha, mixup_p, cutmix_alpha, cutmix_p, noise_minibatchsize,
                            concurrent_combinations, noise_sparsity, noise_patch_lower_scale, noise_patch_upper_scale,
                            generated_ratio, n2n_deepaugment):

        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -1
        else: k = 0

        if k == 0:  # Do input mixup if k is 0
                    
            if n2n_deepaugment: #apply deepaugment if True
                if self.deepaugment_instance is None:
                    self.deepaugment_instance = N2N_DeepAugment(orig_batch_size=out.shape[0], 
                                                                image_size=out.shape[2], 
                                                                channels=out.shape[1],
                                                                noisenet_max_eps=0.75, 
                                                                ratio=0.5)
                out = self.deepaugment_instance(out)
            mixed_out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, generated_ratio, manifold=False, inplace=True)
            noisy_out = apply_noise(mixed_out, noise_minibatchsize, corruptions, concurrent_combinations,
                                                            self.normalized, self.dataset,
                                                            manifold=False, manifold_factor=1, noise_sparsity=noise_sparsity,
                                                            noise_patch_lower_scale=noise_patch_lower_scale,
                                                            noise_patch_upper_scale=noise_patch_upper_scale)
            out = noisy_out
            # plot_images(4, self.mean, self.std, noisy_out, noisy_out)

        return out, targets

    def forward_noise_mixup(
        self,
        out,
        targets,
        robust_samples,
        corruptions,
        mixup_alpha,
        mixup_p,
        manifold,
        manifold_noise_factor,
        cutmix_alpha,
        cutmix_p,
        noise_minibatchsize,
        concurrent_combinations,
        noise_sparsity,
        noise_patch_lower_scale,
        noise_patch_upper_scale,
        generated_ratio,
        n2n_deepaugment,
        style_feats,
        **kwargs,
    ):
        if style_norm_type := kwargs.get("norm_type", None):
            int_adain_probability = kwargs.get("style_probability", 0.0)
            moex_lambda = kwargs.get("moex_lambda", 0.9)

        # define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False:
            k = -1
        elif manifold == True:
            k = np.random.choice(range(3), 1)[0]
        else:
            k = 0

        if k == 0:  # Do deepaugemtn, input mixup and noise injection if k is 0
                    
            if n2n_deepaugment: #apply deepaugment if True
                if self.deepaugment_instance is None:
                    self.deepaugment_instance = N2N_DeepAugment(orig_batch_size=out.shape[0], 
                                                                image_size=out.shape[2], 
                                                                channels=out.shape[1],
                                                                noisenet_max_eps=0.75, 
                                                                ratio=0.5)
                out = self.deepaugment_instance(out)
            mixed_out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, generated_ratio, manifold=False, inplace=False)
            noisy_out = apply_noise(mixed_out, noise_minibatchsize, corruptions, concurrent_combinations,
                                                            self.normalized, self.dataset,
                                                            manifold=False, manifold_factor=1, noise_sparsity=noise_sparsity,
                                                            noise_patch_lower_scale=noise_patch_lower_scale,
                                                            noise_patch_upper_scale=noise_patch_upper_scale)
            out = noisy_out
            #plot_images(2, self.mean, self.std, noisy_out, noisy_out)

        out = self.blocks[0](out)

        if style_feats is not None:
            style_feats = self.blocks[0](style_feats)

        prob = torch.rand(1).item()

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            if style_norm_type == "pono_moex":
                if prob < int_adain_probability and i == 0:
                    out = self.moex(out, targets, lam=moex_lambda)

            out = ResidualBlock(out)

            if style_norm_type == "int_adain":
                if prob < int_adain_probability and i == 0:
                    # style_feats = self.blocks[0](style_feats)
                    style_feats = ResidualBlock(style_feats)
                    out = self.internal_adain(out, style_feats)
                    
            if k == (i + 1):  # Do manifold mixup if k is greater 0
                out, targets = mixup_process(
                    out,
                    targets,
                    robust_samples,
                    self.num_classes,
                    mixup_alpha,
                    mixup_p,
                    cutmix_alpha,
                    cutmix_p,
                    generated_ratio,
                    manifold=True,
                    inplace=False,
                )
                out = noise_up(
                    out,
                    robust_samples=robust_samples,
                    add_noise_level=0.5,
                    mult_noise_level=0.5,
                    sparse_level=0.65,
                    l0_level=0.0,
                )

        return out, targets

    def _calc_mean_std(self, feat, eps=1e-5):
        """
        Calculate mean and standard deviation for each channel in the feature map (i.e., per image per channel or per instance).
        """

        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def internal_adain(self, content_feat, style_feat, epsilon=1e-5):
        """
        Perform Instance Normalization by normalizing content_feat and applying the style_feat's mean and std.
        """
        # Compute mean and std for content and style features
        content_mean, content_std = self._calc_mean_std(content_feat, epsilon)  # noqa: F821
        style_mean, style_std = self._calc_mean_std(style_feat, epsilon)

        # Normalize content feature and apply style moments
        normalized_content = (content_feat - content_mean) / content_std
        output = normalized_content * style_std + style_mean

        return output

    def pono(x, epsilon=1e-5):
        """
        perform positional normalization with means and standard deviation along the channel dimension.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + epsilon).sqrt()
        normalized = (x - mean) / std
        return normalized, mean, std

    def moex(x, y, lam=0.9, epsilon=1e-5):
        """
        Moment Exchange (MoEx)

        Args:
            x: input tensor of shape (B, C, ...)
            y: labels tensor (B, num_classes) or already soft labels
            lam: mixing factor for labels
            epsilon: numerical stability for PONO

        Returns:
            x_moex: tensor after moment exchange
            y_moex: mixed labels
        """
        B = x.size(0)

        if B % 2 != 0:
            raise ValueError("Batch size must be even for simple half-split MoEx.")

        # --- split batch ---
        half = B // 2
        x1, x2 = x[:half], x[half:]
        y1, y2 = y[:half], y[half:]

        # --- normalize each half ---
        n1, m1, s1 = pono(x1, epsilon)
        n2, m2, s2 = pono(x2, epsilon)

        # --- exchange moments ---
        # x1 gets moments from x2, and vice versa
        x1_moex = n1 * s2 + m2
        x2_moex = n2 * s1 + m1

        x_moex = torch.cat([x1_moex, x2_moex], dim=0)

        # --- exchange / mix labels ---
        y1_moex = lam * y1 + (1 - lam) * y2
        y2_moex = lam * y2 + (1 - lam) * y1

        y_moex = torch.cat([y1_moex, y2_moex], dim=0)

        return x_moex, y_moex

