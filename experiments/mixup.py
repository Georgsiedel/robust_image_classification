import torch
from torchvision.transforms import functional as F
import numpy as np
import math
from torch import Tensor
from typing import List, Optional, Tuple
import random
from run_0 import device

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, generated_ratio: float = 0.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.generated_ratio = generated_ratio
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor, robust_samples: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        #if batch.ndim != 4:
        #    raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim == 1:
            if target.dtype != torch.int64:
                raise TypeError(
                    f"1D target must be torch.int64 for class indices. Got {target.dtype}"
                )
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        elif target.ndim == 2:  # target.ndim == 2
            if target.shape[1] != self.num_classes:
                raise ValueError(
                    f"2D target must have shape (B, num_classes={self.num_classes}). "
                    f"Got {target.shape}"
                )
            if not target.is_floating_point():
                target = target.to(dtype=batch.dtype)
        else:
            raise ValueError(f"Target ndim should be 1 or 2. Got {target.ndim}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if torch.rand(1).item() >= self.p:
            return batch, target

        lambda_param = np.random.beta(self.alpha, self.alpha) if self.alpha > 0.0 else 1.0

        #Here, we do mixup and cutmix separately for original and for generated images to not mix up different
        #confidence levels for original and generated images respectively
        q = int(batch.shape[0] / (robust_samples+1))
        gen = int(q * self.generated_ratio)
        orig = q - gen
        index_gen = torch.randperm(gen).to(device)
        index_orig = torch.randperm(orig).to(device)
        for i in range(robust_samples+1):
            batch[q*i:q*i+orig] = lambda_param * batch[q*i:q*i+orig] + (1 - lambda_param) * batch[q*i:q*i+orig][index_orig]
            batch[q*i+orig:q*(i+1)] = lambda_param * batch[q*i+orig:q*(i+1)] + (1 - lambda_param) * batch[q*i+orig:q*(i+1)][index_gen]
        target[:orig] = lambda_param * target[:orig] + (1 - lambda_param) * target[:orig][index_orig]
        target[orig:] = lambda_param * target[orig:] + (1 - lambda_param) * target[orig:][index_gen]

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, generated_ratio: float = 0.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.generated_ratio = generated_ratio
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor, robust_samples: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
            robust_samples (int): Integer for the number of concatenated similar images in the batch

        Returns:
            Tensor: Randomly transformed batch.
        """
        #if batch.ndim != 4:
        #    raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        
        if target.ndim == 1:
            if target.dtype != torch.int64:
                raise TypeError(
                    f"1D target must be torch.int64 for class indices. Got {target.dtype}"
                )
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        elif target.ndim == 2:  # target.ndim == 2
            if target.shape[1] != self.num_classes:
                raise ValueError(
                    f"2D target must have shape (B, num_classes={self.num_classes}). "
                    f"Got {target.shape}"
                )
            if not target.is_floating_point():
                target = target.to(dtype=batch.dtype)
        else:
            raise ValueError(f"Target ndim should be 1 or 2. Got {target.ndim}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if torch.rand(1).item() >= self.p:
            return batch, target

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = np.random.beta(self.alpha, self.alpha) if self.alpha > 0.0 else 1.0
        W, H = F.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        #Here, we do mixup and cutmix separately for original and for generated images to not mix up different confidences
        q = int(batch.shape[0] / (robust_samples+1))
        gen = int(q * self.generated_ratio)
        orig = q - gen
        index_gen = torch.randperm(gen).to(device)
        index_orig = torch.randperm(orig).to(device)
        for i in range(robust_samples+1):
            batch[q*i:q*i+orig, :, y1:y2, x1:x2] = batch[q*i:q*i+orig, :, y1:y2, x1:x2][index_orig]
            batch[q*i+orig:q*(i+1), :, y1:y2, x1:x2] = batch[q*i+orig:q*(i+1), :, y1:y2, x1:x2][index_gen]
        target[:orig] = lambda_param * target[:orig] + (1 - lambda_param) * target[:orig][index_orig]
        target[orig:] = lambda_param * target[orig:] + (1 - lambda_param) * target[orig:][index_gen]

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def mixup_process(inputs, targets, robust_samples, num_classes, mixup_alpha, mixup_p, cutmix_alpha, cutmix_p,
                  generated_ratio, manifold, inplace):

    #if manifold==True and (mixup_p > 0.0 or cutmix_p > 0.0):
    #    mixupcutmix = RandomMixup(num_classes, p=mixup_p, alpha=mixup_alpha, inplace=inplace)
    #    inputs, targets = mixupcutmix(inputs, targets, robust_samples)

    total_probability = cutmix_p + mixup_p
    if total_probability > 1:
        cutmix_p /= total_probability
        mixup_p /= total_probability

    random_number = random.uniform(0, 1)

    if random_number < cutmix_p:
        if cutmix_alpha == 0.0:
            return inputs, targets
        mixupcutmix = RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha, generated_ratio=generated_ratio, inplace=inplace)
        inputs, targets = mixupcutmix(inputs, targets, robust_samples)
    elif random_number < cutmix_p + mixup_p:
        if mixup_alpha == 0.0:
            return inputs, targets
        mixupcutmix = RandomMixup(num_classes, p=1.0, alpha=mixup_alpha, generated_ratio=generated_ratio, inplace=inplace)
        inputs, targets = mixupcutmix(inputs, targets, robust_samples)

    return inputs, targets
