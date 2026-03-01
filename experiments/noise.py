import re
import torch
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
import torch.distributions as dist
from experiments.data import normalization_values, DataLoading
from experiments.utils import plot_images
from itertools import chain
from run_0 import device

def random_erasing_style_mask(batch, noise_patch_lower_scale=0.3, noise_patch_upper_scale=1.0, ratio=[0.3, 3.3]):
        """Get image mask for Patched Noise. Rectangle fully inside the image, as in RandomErasing
        (https://github.com/gatheluck/PatchGaussian/blob/master/patch_gaussian.py)
        Args:
            batch (Tensor): batch of images to be masked.
            noise_patch_lower_scale (sequence): Lower bound for range of proportion of masked area against input image. Upper bound is 1.0
            ratio (sequence): range of aspect ratio of masked area.
        """

        if noise_patch_lower_scale == 1.0 and noise_patch_upper_scale == 1.0:
            return torch.ones(batch.size(), dtype=torch.bool, device=device)

        img_c, img_h, img_w = batch.shape[-3], batch.shape[-2], batch.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))

        patched_area = area * torch.empty(1).uniform_(noise_patch_lower_scale, noise_patch_upper_scale).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        h = int(round(math.sqrt(patched_area * aspect_ratio)))
        w = int(round(math.sqrt(patched_area / aspect_ratio)))
        if h > img_h:
            h = img_h
            w = int(round(img_w * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        if w > img_w:
            w = img_w
            h = int(round(img_h * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        mask = torch.zeros(batch.size(), dtype=torch.bool, device=device)
        mask[:,:,i:i + h, j:j + w] = True

        return mask

def patch_gaussian_style_patch_mask(im_size: int, window_size: int):
    """
    Get image mask for Patched Noise. Square with center somewhere in the image, as in Patch Gaussian.
    Torch implementation from here: https://github.com/gatheluck/PatchGaussian/blob/master/patch_gaussian.py
    Args:
    - im_size: size of image
    - window_size: size of window. if -1, return full size mask
    """
    assert im_size >= 1
    assert (1 <= window_size) or (window_size == -1)

    # if window_size == -1, return all True mask.
    if window_size == -1:
        return torch.ones(im_size, im_size, dtype=torch.bool)

    mask = torch.zeros(im_size, im_size, dtype=torch.bool)  # all elements are False

    # sample window center. if window size is odd, sample from pixel position. if even, sample from grid position.
    window_center_h = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)
    window_center_w = random.randrange(0, im_size) if window_size % 2 == 1 else random.randrange(0, im_size + 1)

    for idx_h in range(window_size):
        for idx_w in range(window_size):
            h = window_center_h - math.floor(window_size / 2) + idx_h
            w = window_center_w - math.floor(window_size / 2) + idx_w

            if (0 <= h < im_size) and (0 <= w < im_size):
                mask[h, w] = True

    return mask

def _noising(x, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0, l0_level = 0.0):
    # based on https://github.com/erichson/NoisyMix
    add_noise = 0.0
    mult_noise = 1.0
    #with torch.cuda.device(0):
    std = torch.var(x).detach() ** 0.5
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.randn(size=x.shape, dtype=torch.float16, device=device)
        #torch.clamp(add_noise, min=-(2*std), max=(2*std), out=add_noise) # clamp
        sparse = torch.rand(x.shape, dtype=torch.float16, device=device)
        add_noise[sparse<sparse_level] = 0

    rand = random.random()
    if rand < 0.5 and l0_level > 0.0:
        x = x + add_noise
        l0_batch = l0(x, np.random.beta(2, 5)*l0_level, std, sparse_level)
        return l0_batch
    elif mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.rand(x.shape, dtype=torch.float16, device=device)-1) + 1
        sparse = torch.rand(x.shape, dtype=torch.float16, device=device)
        mult_noise[sparse<sparse_level] = 1.0

    return mult_noise * x + add_noise

def noise_up(x, robust_samples=0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0, l0_level=0.0):
    # based on https://github.com/erichson/NoisyMix
    x = x.clone()

    q = int(x.shape[0] / (robust_samples+1))
    sparsity = random.random() * sparse_level
    for i in range(robust_samples+1):
        x[q*i:q*(i+1)] = _noising(x[q*i:q*(i+1)], add_noise_level=add_noise_level * (i+1), mult_noise_level=mult_noise_level,
                              sparse_level=sparsity, l0_level=l0_level)
    return x

def l0(batch, epsilon, std, sparsity):
    #samples a mask of L0 noise for the batch that contains either:
    # 1) + or - or uniform distribution in between 3*std of the batch-data (random choice of elements with a total ratio epsilon for every image in the batch)
    # 2) 0 for the rest of the elements

    num_dimensions = torch.numel(batch[0])
    num_pixels = int(num_dimensions * epsilon)
    lower_bounds = torch.arange(0, batch.size(0) * num_dimensions, num_dimensions, device=device)
    upper_bounds = torch.arange(num_dimensions, (batch.size(0) + 1) * num_dimensions, num_dimensions, device=device)
    indices = torch.cat([torch.randint(l, u, (num_pixels,), device=device) for l, u in zip(lower_bounds, upper_bounds)])
    mask = torch.full(batch.size(), False, dtype=torch.bool, device=device)
    mask.view(-1)[indices] = True
    # apply sparsity: a share of the random impulse noise pixels is left out
    sparse_matrix = torch.rand(batch.shape, dtype=torch.float16, device=device)
    mask[sparse_matrix < sparsity] = False

    random_numbers = random.sample([torch.randint(2, size=batch.shape, dtype=torch.float16, device=device),
                                    torch.rand(batch.shape, dtype=torch.float16, device=device)], 1)[0]
    random_numbers = (random_numbers * 2 - 1) * std * 3
    l0_batch = torch.where(mask, random_numbers, batch)

    return l0_batch

def do_noisy_mixup(x, y, num_classes, jsd=0, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    #https://github.com/erichson/NoisyMix
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    if jsd == 0:
        index = torch.randperm(x.size()[0]).cuda()
        x = lam * x + (1 - lam) * x[index]
        x = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
    else:
        kk = 0
        q = int(x.shape[0] / 3)
        index = torch.randperm(q).cuda()

        for i in range(1, 4):
            x[kk:kk + q] = lam * x[kk:kk + q] + (1 - lam) * x[kk:kk + q][index]
            x[kk:kk + q] = _noise(x[kk:kk + q], add_noise_level=add_noise_level * i, mult_noise_level=mult_noise_level,
                                  sparse_level=sparse_level)
            kk += q

    y = torch.nn.functional.one_hot(y, num_classes=num_classes).to(dtype=y.dtype)
    y = lam * y + (1 - lam) * y[index]  # added
    return x, y

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    #https://github.com/erichson/NoisyMix
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            var = torch.var(x) ** 0.5
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
            # torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
            sparse = torch.cuda.FloatTensor(x.shape).uniform_()
            add_noise[sparse < sparse_level] = 0
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (
                        2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1) + 1
            sparse = torch.cuda.FloatTensor(x.shape).uniform_()
            mult_noise[sparse < sparse_level] = 1.0

    return mult_noise * x + add_noise

def apply_noise(batch, minibatchsize, corruptions, concurrent_combinations, normalized, dataset, manifold=False,
                manifold_factor=1, noise_sparsity=0.0, noise_patch_lower_scale=1.0, noise_patch_upper_scale=1.0):

    if corruptions is None:
        return batch
    #Calculate the mean values for each channel across all images
    mean, std = normalization_values(batch, dataset, normalized, manifold, manifold_factor, verbose=False)

    # Throw out noise outside Gaussian, (L0) and Linf for manifold noise (since epsilon is dependent on dimensionality)
    if manifold:
        if not isinstance(corruptions, dict):
            corruptions = [c for c in corruptions if c.get('noise_type') in {'gaussian', 'uniform-linf', 'standard', 'uniform-l0-impulse'}]
            corruptions = np.array(corruptions)
            if corruptions.size == 0:
                print('Warning: noise_type of p-norm outside L0 and Linf may not be applicable for manifold noise')
        else:
            if corruptions.get('noise_type') != ('gaussian' or 'uniform-linf' or 'standard' or 'uniform-l0-impulse'):
                print('Warning: noise_type of p-norm outside L0 and Linf may not be applicable for manifold noise')

    # Split into minibatches, handling residual (if any)
    full_minibatches = batch[:batch.size(0) // minibatchsize * minibatchsize].view(-1, minibatchsize, batch.size(1), batch.size(2), batch.size(3))
    residual_batch = batch[batch.size(0) // minibatchsize * minibatchsize:] if batch.size(0) % minibatchsize > 0 else None
    minibatches = chain(full_minibatches, [residual_batch] if residual_batch is not None else [])
    
    if isinstance(corruptions, dict):
        corruptions = [corruptions]
    elif isinstance(corruptions, np.ndarray):
        corruptions = corruptions.tolist()

    for id, minibatch in enumerate(minibatches):
        corruptions_list = random.sample(corruptions, k=concurrent_combinations)

        #noisy_minibatch = torch.clone(minibatch)
        for _, (corruption) in enumerate(corruptions_list):
            if corruption['distribution'] == 'uniform':
                d = dist.Uniform(0, 1).sample()
            elif corruption['distribution'] == 'beta2-5':
                d = np.random.beta(2, 5)
            elif corruption['distribution'] == 'max':
                d = 1
            else:
                print('Unknown distribution for epsilon value of p-norm corruption. Max value chosen instead.')
                d = 1
            if d == 0: #dist sampling include lower bound but exclude upper, but we do not want eps = 0
                d = 1
            epsilon = float(d) * float(corruption['epsilon'])
            sparsity = random.random() * noise_sparsity

            if corruption['noise_type'] == 'NoisyMix':
                noisy_minibatch = noise_up(
                    minibatch,
                    robust_samples=0,
                    add_noise_level=epsilon,
                    mult_noise_level=epsilon,
                    sparse_level=0.65,
                    l0_level=0.0,
                )
            else:
                noisy_minibatch = sample_lp_corr_batch(corruption['noise_type'], epsilon, minibatch, corruption['sphere'], mean, std, manifold, sparsity)

        mask = random_erasing_style_mask(minibatch, noise_patch_lower_scale=noise_patch_lower_scale,
                                         noise_patch_upper_scale=noise_patch_upper_scale, ratio = [0.3, 3.3])
        final_minibatch = torch.where(mask, noisy_minibatch, minibatch)
        batch[id*minibatchsize:min((id+1)*minibatchsize, batch.size(0))] = final_minibatch

    return batch

def sample_lp_corr_batch(noise_type, epsilon, batch, density_distribution_max, mean, std, manifold, sparsity=0.0):
    corruption = torch.zeros(batch.size(), dtype=torch.float16)

    if noise_type == 'uniform-linf':
        if density_distribution_max == True:  # sample on the hull of the norm ball
            rand = np.random.random(batch.shape)
            sign = np.where(rand < 0.5, -1, 1)
            corruption = torch.from_numpy(sign * epsilon)
        else: #sample uniformly inside the norm ball
            corruption = (torch.rand(batch.shape, dtype=torch.float16, device=device) - 0.5) * 2 * epsilon
    elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
        corruption = torch.randn(size=batch.shape, dtype=torch.float16, device=device) * epsilon
    elif noise_type == 'uniform-l0-impulse':
        num_dimensions = torch.numel(batch[0])
        num_pixels = int(num_dimensions * epsilon)
        lower_bounds = torch.arange(0, batch.size(0) * num_dimensions, num_dimensions, device=device)
        upper_bounds = torch.arange(num_dimensions, (batch.size(0) + 1) * num_dimensions, num_dimensions, device=device)
        indices = torch.cat([torch.randint(l, u, (num_pixels,), device=device) for l, u in zip(lower_bounds, upper_bounds)])
        mask = torch.full(batch.size(), False, dtype=torch.bool, device=device)
        mask.view(-1)[indices] = True
        #apply sparsity: a share of the random impulse noise pixels is left out
        sparse_matrix = torch.rand(batch.shape, dtype=torch.float16, device=device)
        mask[sparse_matrix < sparsity] = False

        if density_distribution_max == True:
            random_numbers = torch.randint(2, size=batch.size(), dtype=torch.float16, device=device)
        else:
            random_numbers = torch.rand(batch.shape, dtype=torch.float16, device=device)
        if manifold:
            random_numbers = (random_numbers * 2) - 1

        #normalizing the impulse noise values
        random_numbers = (random_numbers - mean) / std
        batch_corr = torch.where(mask, random_numbers, batch)

        return batch_corr

    elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
        img_corr = torch.zeros(batch[0].size(), dtype=torch.float16, device=device)
        #number of dimensions
        d = img_corr.numel()
        # extract Lp-number from args.noise variable
        lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)][0]
        u = dist.Gamma(1 / lp, 1).sample(img_corr.shape).to(device)
        u = u ** (1 / lp)
        sign = torch.sign(torch.rand(img_corr.shape, device=device) - 0.5)
        norm = torch.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
        if density_distribution_max == True:
            r = 1
        else:  # uniform density distribution
            r = dist.Uniform(0, 1).sample() ** (1.0 / d)
        img_corr = epsilon * r * u * sign / norm #image-sized corruption, epsilon * random radius * random array / normed
        corruption = img_corr.expand(batch.size()).to(device)

    elif noise_type == 'standard':
        return batch
    else:
        print('Unknown type of noise')

    corruption = corruption.to(device)#.clone()
    #sparsity is applied
    sparse_matrix = torch.rand(batch.shape, dtype=torch.float16, device=device)
    #corruption[sparse_matrix < sparsity] = 0
    sparse_corruption = torch.where(sparse_matrix < sparsity, 0, corruption)

    corrupted_batch = batch + (sparse_corruption / std)
    if not manifold:
        corrupted_batch = torch.clamp(corrupted_batch, (0-mean)/std, (1-mean)/std)  #clip below lower and above upper bound
    return corrupted_batch

if __name__ == '__main__':

    dataset='PCAM'
    test_corruption = {'noise_type': 'uniform-l0.5', 'epsilon': 700000.0, 'sphere': True, 'distribution': 'max'},

    Dataloader = DataLoading(dataset=dataset, validontest=True, resize=False, 
                                run=0, number_workers=0, kaggle=False)
    Dataloader.create_transforms(train_aug_strat_orig='None', train_aug_strat_gen='None')
    Dataloader.load_base_data(test_only=True)
    
    testloader = torch.utils.data.DataLoader(Dataloader.testset, 
                                             batch_size=20, 
                                             pin_memory=True, 
                                             num_workers=0,
                                             shuffle=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            inputs_pert = inputs.clone()
            inputs_pert = apply_noise(inputs_pert, 1, test_corruption, 1, False, dataset)
            break

    plot_images(2, inputs, corrupted_images=inputs_pert, save=True)