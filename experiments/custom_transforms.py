import random
import torch
import torch.cuda.amp
import torch.utils
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms
from run_0 import device
import gc
import experiments.eval_corruption_transforms as c
import torch
from PIL import Image
import numpy as np
import kornia
from typing import Optional
from experiments.utils import plot_images
import experiments.style_transfer as style_transfer
from experiments.custom_datasets import StylizedTensorDataset

class TransformFactory:
    def __init__(self, re, soft_crop, style_path, strat_name, style, style_and_aug, dataset, minibatchsize=8):
        self.re = re
        self.TAc = CustomTA_color()
        self.TAg = CustomTA_geometric()
        self.strat_name = strat_name
        self.style = style
        self.style_and_aug = style_and_aug
        self.style_path = style_path
        self.minibatchsize = minibatchsize
        self.dataset = dataset
        self.soft_crop = soft_crop

    def _stylization(self, probability=1.0, alpha_min=0.2, alpha_max=1.0):
        vgg, decoder = style_transfer.load_models()
        style_feats = style_transfer.load_feat_files(self.style_path)
        if self.dataset in ['KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
            style_transforms = style_transfer.NSTTransform_rectangular(style_feats, 
                                                                        vgg, 
                                                                        decoder, 
                                                                        alpha_min=alpha_min, 
                                                                        alpha_max=alpha_max, 
                                                                        probability=probability,
                                                                        overlap=64)
        else:
            style_transforms = style_transfer.NSTTransform(style_feats, 
                                                           vgg, decoder, 
                                                           alpha_min=alpha_min, 
                                                           alpha_max=alpha_max, 
                                                           probability=probability)
        return style_transforms

    def get_transforms(self):
        batch_transforms = BatchStyleTransforms(stylized_ratio=self.style['probability'], 
                                           batch_size=100, 
                                           transform_style=self._stylization(probability=1.0, 
                                                                             alpha_min=self.style['alpha_min'], 
                                                                             alpha_max=self.style['alpha_max']))
        
        if self.strat_name == 'None':
            transforms_potentially_masked = None
        else:
            try:
                transforms_potentially_masked = getattr(transforms_v2, self.strat_name)()
            except AttributeError:
                print(f"[Warning] Transform '{self.strat_name}' not found in transforms_v2. Skipping transform.")
                transforms_potentially_masked = None
        
        if self.soft_crop is not None:
            transforms_never_masked = transforms.Compose([self.re, self.soft_crop])
            handle_confidence_output = True
        else:
            transforms_never_masked = transforms.Compose([self.re])
            handle_confidence_output = False

        if self.style_and_aug:
            after_transforms = MaskIteratorConfidenceTransforms(transforms_potentially_masked=transforms_potentially_masked, 
                                                      transforms_never_masked=transforms_never_masked, 
                                                      filter_mask=False, 
                                                      handle_confidence_output=handle_confidence_output,
                                                      batchsize=self.minibatchsize)
        else:
            after_transforms = MaskIteratorConfidenceTransforms(transforms_potentially_masked=transforms_potentially_masked, 
                                                      transforms_never_masked=transforms_never_masked, 
                                                      filter_mask=True, 
                                                      handle_confidence_output=handle_confidence_output,
                                                      batchsize=self.minibatchsize)
        
        return batch_transforms, after_transforms
        
    def get_transforms_style_first(self):

        batch_transforms = DatasetStyleTransforms(stylized_ratio=self.style['probability'], 
                                           batch_size=100, 
                                           transform_style=self._stylization(probability=1.0, 
                                                                             alpha_min=self.style['alpha_min'], 
                                                                             alpha_max=self.style['alpha_max']))
        if self.strat_name == 'None':
            aug_class = []
        else:
            try:
                aug_class = getattr(transforms_v2, self.strat_name)()
            except AttributeError:
                print(f"[Warning] Transform '{self.strat_name}' not found in transforms_v2. Skipping transform.")
                aug_class = []
        
        if self.style_and_aug:
            after_style = transforms.Compose([aug_class, self.re])
            after_nostyle = transforms.Compose([aug_class, self.re])
        else:
            after_style = transforms.Compose([self.re])
            after_nostyle = transforms.Compose([aug_class, self.re])

        return batch_transforms, after_style, after_nostyle

def safe_kornia_loader(path, try_kornia=False):
    if try_kornia:
        try:
            return kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32) # Kornia: float32, CHW, [0,1]
        except:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
    else:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

class PilToNumpy(object):
    def __init__(self, as_float=False, scaled_to_one=False):
        self.as_float = as_float
        self.scaled_to_one = scaled_to_one
        assert (not scaled_to_one) or (as_float and scaled_to_one),\
                "Must output a float if rescaling to one."

    def __call__(self, image):
        arr = np.array(image)

        # Add channel dimension back if grayscale, because to PIL erased it before
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        # Convert dtype as needed
        if not self.as_float:
            return arr.astype(np.uint8)
        elif not self.scaled_to_one:
            return arr.astype(np.float32)
        else:
            return arr.astype(np.float32) / 255

class NumpyToPil(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return Image.fromarray(image)

class TensorToNumpyUint8(object):
    def __call__(self, tensor):
        # tensor: torch.Tensor [C,H,W], float in [0,1]
        arr = tensor.mul(255).byte().numpy()   # -> uint8
        return np.transpose(arr, (1, 2, 0)) if arr.ndim == 3 else arr[0]  # CHW -> HWC

class NumpyUint8ToTensor(object):
    def __call__(self, arr):
        if arr.ndim == 2:  # grayscale
            arr = arr[None, ...]  # add channel dim -> (1,H,W)
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(arr.copy()).float() / 255.0
        return tensor

class ExpandGrayscaleTensorTo3Channels:
    def __call__(self, x):
        # Expect x to be a torch.Tensor of shape [C, H, W] or [B, C, H, W]
        if isinstance(x, torch.Tensor):
            if x.dim() == 3 and x.shape[0] == 1:  # Single image: [C, H, W]
                return x.repeat(3, 1, 1)
            elif x.dim() == 4 and x.shape[1] == 1:  # Batch: [B, C, H, W]
                return x.repeat(1, 3, 1, 1)
        # If input is PIL Image or others, just return as is (or convert if you want)
        return x

class ToFloat32:
    def __call__(self, x):
        return x.to(torch.float32)

class DivideBy3:
    def __call__(self, x):
        return x / 3.0
    
def build_transform_c_bar(name, severity, dataset, resize):
    assert dataset in ['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet-100', 'TinyImageNet', 'GTSRB', 'PCAM', 
                       'EuroSAT', 'WaferMap', 'Casting-Product-Quality', 'Describable-Textures', 'Flickr-Material', 
                       'TreeSAT', 'KITTI_Distance_Multiclass', 'KITTI_RoadLane', 'SynthiCAD', 'NEU-surface-defect', 'DermaMNIST'],\
            "Dataset not defined and functionality not explored for c-bar benchmark."
    
    if dataset in ['CIFAR10', 'CIFAR100', 'GTSRB', 'DermaMNIST']: 
        im_size = 32
    elif dataset in ['TinyImageNet', 'EuroSAT', 'WaferMap', 'PCAM']:
        im_size = 64
    elif dataset in ['NEU-surface-defect']:
        im_size = 128
    elif dataset in ['KITTI_Distance_Multiclass', 'KITTI_RoadLane']:
        im_size = 384
    else:
        im_size = 224

    if resize:
        im_size = 224

    transform_c_bar_list = [
    c.SingleFrequencyGreyscale,
    c.CocentricSineWaves,
    c.PlasmaNoise,
    c.CausticNoise,
    c.PerlinNoise,
    c.BlueNoise,
    c.BrownishNoise,
    c.TransverseChromaticAbberation,
    c.CircularMotionBlur,
    c.CheckerBoardCutOut,
    c.Sparkles,
    c.InverseSparkles,
    c.Lines,
    c.BlueNoiseSample,
    c.PinchAndTwirl,
    c.CausticRefraction,
    c.Ripple
    ]   

    transform_c_bar_dict = {t.name : t for t in transform_c_bar_list}
    
    return transform_c_bar_dict[name](severity=severity, im_size=im_size)

def transform_c(image, severity=1, corruption_name=None, corruption_number=-1):
    """This function returns a corrupted version of the given image.
    
    Args:
        image (numpy.ndarray):      image to corrupt; a numpy array in [0, 255], expected datatype is np.uint8
                                    expected shape is either (height x width x channels) or (height x width); 
                                    width and height must be at least 32 pixels;
                                    channels must be 1 or 3;
        severity (int):             strength with which to corrupt the image; an integer in [1, 5]
        corruption_name (str):      specifies which corruption function to call, must be one of
                                        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                                        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                                    the last four are validation corruptions
        corruption_number (int):    the position of the corruption_name in the above list; an integer in [0, 18]; 
                                        useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    Returns:
        numpy.ndarray:              the image corrupted by a corruption function at the given severity; same shape as input
    """
    corruption_tuple = (c.gaussian_noise, 
                        c.shot_noise, 
                        c.impulse_noise, 
                        c.defocus_blur,
                        c.glass_blur, 
                        c.motion_blur, 
                        c.zoom_blur, 
                        c.snow, 
                        c.frost, 
                        c.fog,
                        c.brightness, 
                        c.contrast, 
                        c.elastic_transform, 
                        c.pixelate,
                        c.jpeg_compression, 
                        c.speckle_noise, 
                        c.gaussian_blur, 
                        c.spatter,
                        c.saturate)

    corruption_dict = {corr_func.__name__: corr_func for corr_func in
                   corruption_tuple}

    if not isinstance(image, np.ndarray):
        raise AttributeError('Expecting type(image) to be numpy.ndarray')
    if not (image.dtype.type is np.uint8):
        raise AttributeError('Expecting image.dtype.type to be numpy.uint8')
        
    if not (image.ndim in [2,3]):
        raise AttributeError('Expecting image.shape to be either (height x width) or (height x width x channels)')
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    
    height, width, channels = image.shape

    if height == 32:
        scale = 'cifar'
    elif 32 < height <= 96:
        scale = 'tin'
    else: 
        scale = 'in'
    
    if (height < 32 or width < 32):
        raise AttributeError('Image width and height must be at least 32 pixels')
    
    if not (channels in [1,3]):
        raise AttributeError('Expecting image to have either 1 or 3 channels (last dimension)')
        
    if channels == 1:
        image = np.stack((np.squeeze(image),)*3, axis=-1)
    
    if not severity in [1,2,3,4,5]:
        raise AttributeError('Severity must be an integer in [1, 5]')
    
    if not (corruption_name is None):
        image_corrupted = corruption_dict[corruption_name](Image.fromarray(image),
                                                       severity, scale)
    elif corruption_number != -1:
        image_corrupted = corruption_tuple[corruption_number](Image.fromarray(image),
                                                          severity, scale)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(image_corrupted)

class RandomCommonCorruptionTransform:
    def __init__(self, set, corruption_name, dataset, csv_handler, resize):
        self.corruption_name = corruption_name
        self.set = set
        self.dataset = dataset
        self.csv_handler = csv_handler
        self.TtoPIL = transforms.ToPILImage()
        self.PILtoNP = PilToNumpy()
        self.NPtoPIL = NumpyToPil()
        self.ToTensor = transforms.ToTensor()
        self.NumpyUint8ToTensor = NumpyUint8ToTensor()
        self.TensorToNumpyUint8 = TensorToNumpyUint8()
        self.resize = resize

    def __call__(self, img):
        severity = random.randint(1, 5)

        if self.set == 'c':
            img_np = self.TensorToNumpyUint8(img)
            corrupted_img = self.NumpyUint8ToTensor(transform_c(img_np, severity=severity, corruption_name=self.corruption_name))
        elif self.set == 'c-bar':
            severity_value = self.csv_handler.get_value(self.corruption_name, severity)

            comp = transforms.Compose([
                self.TensorToNumpyUint8,              # Tensor [0,1] float -> Numpy [0,255] uint8
                build_transform_c_bar(self.corruption_name, severity_value, self.dataset, self.resize),
                self.NumpyUint8ToTensor               # Numpy [0,255] uint8 -> Tensor [0,1] float
            ])

            corrupted_img = comp(img)

        return corrupted_img

class DatasetStyleTransforms:
    def __init__(self, transform_style, batch_size, stylized_ratio):
        """
        Args:
            transform_style: Callable to apply stylization.
            batch_size: Batch size for tensors passed to transform_style.
            stylized_ratio: Fraction of images to stylize (0 to 1).
        """
        self.transform_style = transform_style
        self.batch_size = batch_size
        self.stylized_ratio = stylized_ratio

    def __call__(self, dataset):
        """
        Stylize a fraction of images in the dataset and return a new dataset.

        Args:
            dataset: PyTorch Dataset to process.

        Returns:
            stylized_dataset: A new TensorDataset with stylized images.
        """

        num_images = len(dataset)
        num_stylized = int(num_images * self.stylized_ratio)
        stylized_indices = torch.randperm(num_images)[:num_stylized]
        
        # Create a Subset with the stylized indices
        stylized_subset = Subset(dataset, stylized_indices)

        # DataLoader for processing the stylized subset
        loader = DataLoader(stylized_subset, batch_size=self.batch_size, shuffle=False)
        
        # Use zeros as placeholders for non-stylized images and labels
        sample_image, _ = dataset[0]  # Get sample shape from the dataset
        stylized_images = torch.zeros((num_stylized, *sample_image.shape), dtype=sample_image.dtype)

        # Iterate over the DataLoader and process stylized images
        for batch_indices, (images, _) in zip(loader.batch_sampler, loader):  
            # Apply the transformation to the batch
            transformed_images = self.transform_style(images)

            # Store the transformed images and labels in their original positions
            stylized_images[batch_indices] = transformed_images

        # Delete intermediary variables to save memory
        del loader, stylized_subset
        gc.collect()

        style_mask = torch.zeros(num_images, dtype=torch.bool)
        style_mask[stylized_indices] = True
        style_mask = style_mask.tolist()

        # Return the stylized dataset
        return StylizedTensorDataset(dataset, stylized_images, stylized_indices), style_mask

class BatchStyleTransforms:
    def __init__(self, transform_style, batch_size, stylized_ratio):
        """
        Args:
            transform_style: Callable to apply stylization.
            batch_size: Batch size for tensors passed to transform_style.
            stylized_ratio: Fraction of images to stylize (0 to 1).
        """
        self.transform_style = transform_style
        self.batch_size = batch_size
        self.stylized_ratio = stylized_ratio

    def __call__(self, images):
        """
        Stylize a tensor batch of images.

        Args:
            images (torch.Tensor): A tensor batch of images with shape (batch_size, *image_shape).

        Returns:
            Tuple[torch.Tensor, List[bool]]: 
                - A tensor batch of images where a fraction is stylized, with the same shape as input.
                - A boolean list indicating which images were stylized.
        """
        num_images = len(images)
        num_stylized = int(num_images * self.stylized_ratio)

        if num_stylized > 0:
            # Select indices of images to stylize
            stylized_indices = torch.randperm(num_images)[:num_stylized]
            images_to_stylize = images[stylized_indices]

            # Process the subset of images in smaller batches
            for i in range(0, len(images_to_stylize), self.batch_size):
                # Apply the style transform to the batch
                batch = images_to_stylize[i:i + self.batch_size]
                images_to_stylize[i:i + self.batch_size] = self.transform_style(batch)

            # Replace the original images with the stylized ones
            images[stylized_indices] = images_to_stylize

            # Create the style mask
            style_mask = torch.zeros(num_images, dtype=torch.bool)
            style_mask[stylized_indices] = True
        else:
            # If no images are stylized, create an all-false style mask
            style_mask = torch.zeros(num_images, dtype=torch.bool)

        # Return the modified images and style mask
        return images, style_mask


class RandomChoiceTransforms:
    def __init__(self, transforms, p):
        assert len(transforms) == len(p), "The number of transforms and probabilities must match."

        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        choice = random.choices(self.transforms, self.p)[0]
        return choice(x)

class EmptyTransforms:
    def __init__(self):
        pass  # No operations needed for empty transforms.

    def __call__(self, x):
        return x

class MaskIteratorConfidenceTransforms:
    # Gets a batch of images and a mask 
    # if masked_only=True, applies the transforms only to the unmasked images (where mask==False)
    # handles confidence values returned by iterated transforms (e.g. Soft_RandomCrop)
    # returns batch of images, no mask (inplace operation), but batch of confidence values (either from Soft_RandomCrop or 1.0)
    def __init__(self, transforms_potentially_masked, transforms_never_masked, filter_mask: bool, 
                 handle_confidence_output, batchsize=8):
        self.transforms_never_masked = transforms_never_masked
        self.filter_mask = filter_mask
        self.batchsize = batchsize
        self.handle_confidence_output = handle_confidence_output

        # Gracefully compose only non-None transforms
        transform_list = [t for t in [transforms_potentially_masked, transforms_never_masked] if t is not None]
        self.combined_transforms = transforms.Compose(transform_list) if transform_list else None

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        # create confidence tensor
        confidences = torch.full((x.shape[0],), 1.0, device=x.device, dtype=x.dtype)

        # If no transforms are defined at all, just return x and 1.0 for confidences
        if self.combined_transforms is None and self.transforms_never_masked is None:
            return x, confidences

        if self.filter_mask:
            # Apply combined transforms only to unmasked images
            if self.combined_transforms is not None:

                data_to_process = x[~mask]
                conf_to_process = confidences[~mask]
                num_items = data_to_process.shape[0]

                #inplace loop
                for start in range(0, num_items, self.batchsize):
                    end = min(start + self.batchsize, num_items)
                    minibatch = data_to_process[start:end]
                    out = self.combined_transforms(minibatch)

                    if self.handle_confidence_output:
                        out, conf = out
                    else:
                        conf = 1.0

                    minibatch[:] = out
                    conf_to_process[start:end] = torch.full((out.shape[0],), conf, device=out.device, dtype=out.dtype)
                
                #write back
                x[~mask] = data_to_process
                confidences[~mask] = conf_to_process

            # Apply never-masked transforms only to masked images
            if self.transforms_never_masked is not None:

                data_to_process = x[mask]
                conf_to_process = confidences[mask]
                num_items = data_to_process.shape[0]

                #inplace loop
                for start in range(0, num_items, self.batchsize):
                    end = min(start + self.batchsize, num_items)
                    minibatch = data_to_process[start:end]
                    out = self.transforms_never_masked(minibatch)

                    if self.handle_confidence_output:
                        out, conf = out
                    else:
                        conf = 1.0

                    minibatch[:] = out
                    conf_to_process[start:end] = torch.full((out.shape[0],), conf, device=out.device, dtype=out.dtype)
                
                #write back
                x[mask] = data_to_process
                confidences[mask] = conf_to_process
                        
        else:
            # Apply combined transforms to all images (if defined)
            if self.combined_transforms is not None:

                num_items = x.shape[0]

                #inplace loop
                for start in range(0, num_items, self.batchsize):
                    end = min(start + self.batchsize, num_items)
                    minibatch = x[start:end]
                    out = self.combined_transforms(minibatch)

                    if self.handle_confidence_output:
                        out, conf = out
                    else:
                        conf = 1.0

                    minibatch[:] = out
                    confidences[start:end] = torch.full((out.shape[0],), conf, device=out.device, dtype=out.dtype)

        return x, confidences
        
class DuringTrainingTransforms:
    def __init__(self, synthetic_ratio, robust_samples, transforms_orig_batch, transforms_gen_batch, transforms_orig_iter, transforms_gen_iter):
        self.transforms_orig_batch = transforms_orig_batch
        self.transforms_gen_batch = transforms_gen_batch
        self.transforms_orig_iter = transforms_orig_iter
        self.transforms_gen_iter = transforms_gen_iter
        self.synthetic_ratio = synthetic_ratio
        self.robust_samples = robust_samples

    def __call__(self, x):
        
        total = x.shape[0]
        synth_samples = int(total * self.synthetic_ratio)
        confidences = torch.full((total,), 1.0, device=x.device, dtype=x.dtype)

        if self.robust_samples == 2:
            subset1 = x[int(x.shape[0] / 3):int(x.shape[0] * 2 / 3)]
            subset2 = x[int(x.shape[0] * 2 / 3):]
            conf1 = confidences[int(x.shape[0] / 3):int(x.shape[0] * 2 / 3)]
            conf2 = confidences[int(x.shape[0] * 2 / 3):]

            #apply batched and iterative transforms
            if self.synthetic_ratio > 0.0:
                imgs, mask = self.transforms_gen_batch(subset1[-synth_samples:])
                subset1[-synth_samples:], conf1[-synth_samples:] = self.transforms_gen_iter(imgs, mask)

                imgs, mask = self.transforms_gen_batch(subset2[-synth_samples:])
                subset2[-synth_samples:], conf2[-synth_samples:] = self.transforms_gen_iter(imgs, mask)

            if self.synthetic_ratio < 1.0:
                # (use len - synth_samples to avoid :-0 issue)
                imgs, mask = self.transforms_orig_batch(subset1[: total - synth_samples])
                subset1[: total - synth_samples], conf1[: total - synth_samples] = self.transforms_orig_iter(imgs, mask)

                imgs, mask = self.transforms_orig_batch(subset2[: total - synth_samples])
                subset2[: total - synth_samples], conf2[: total - synth_samples] = self.transforms_orig_iter(imgs, mask)

            x[int(x.shape[0] / 3):int(x.shape[0] * 2 / 3)] = subset1
            x[int(x.shape[0] * 2 / 3):] = subset2
            confidences[int(x.shape[0] / 3):int(x.shape[0] * 2 / 3)] = conf1
            confidences[int(x.shape[0] * 2 / 3):] = conf2

        else:
            
            if self.robust_samples == 0:
                subset = x
                conf = confidences
            elif self.robust_samples == 1:
                subset = x[int(x.shape[0] / 2):]
                conf = confidences[int(x.shape[0] / 2):]

            #apply batched and iterative transforms
            if self.synthetic_ratio > 0.0:
                imgs, mask = self.transforms_gen_batch(subset[-synth_samples:])
                subset[-synth_samples:], conf[-synth_samples:] = self.transforms_gen_iter(imgs, mask)
            if self.synthetic_ratio < 1.0:
                # (use len - synth_samples to avoid :-0 issue)
                imgs, mask = self.transforms_orig_batch(subset[: total - synth_samples])
                subset[: total - synth_samples], conf[: total - synth_samples] = self.transforms_orig_iter(imgs, mask)

            if self.robust_samples == 0:
                x = subset
                confidences = conf
            elif self.robust_samples == 1:
                x[int(x.shape[0] / 2):] = subset
                confidences[int(x.shape[0] / 2):] = conf

        return x, confidences
    
class OneHotAdaptiveConfidenceLabelTransform:
    def __init__(self, label_smoothing: float, n_class: int, multilabel: bool = False):
        self.label_smoothing = label_smoothing
        self.n_class = n_class
        self.multilabel = multilabel

    def __call__(self, labels, confidences):
        """
        labels:
            single-label: LongTensor [B]
            multi-label:  float/bool Tensor or ndarray [B, C]
        confidences:
            single-label: [B]
            multi-label:  [B]

        Returns:
            smoothed labels of shape [B, C]
        """

        # Convert numpy to torch
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if isinstance(confidences, np.ndarray):
            confidences = torch.from_numpy(confidences)

        labels = labels.to(torch.float32)
        confidences = confidences.to(torch.float32)

        if self.multilabel is not True:
            # -------------------------------
            # SINGLE-LABEL CASE
            # -------------------------------
            # labels: [B], confidences: [B]

            assert confidences.shape == labels.shape, "Confidences must match label shape."
            labels = labels.long().unsqueeze(1)
            confidences = confidences.unsqueeze(1)

            # Base uniform mass
            one_hot = torch.ones((labels.shape[0], self.n_class), 
                                 dtype=torch.float32,
                                 device=confidences.device)

            one_hot = one_hot * (1 - (confidences * (1 - self.label_smoothing))) / (self.n_class - 1)

            # Fill correct class with confidence
            one_hot.scatter_(
                dim=1,
                index=labels,
                src=(confidences * (1 - self.label_smoothing))
            )
            return one_hot

        else:
            # -------------------------------
            # MULTI-LABEL CASE
            # -------------------------------
            # labels: [B, C] binary mask
            # confidences: [B] → one value per sample
            assert labels.ndim == 2, "Multilabel mode requires labels with shape [B, C]."

            # Expand confidences to match label shape: [B, 1] → [B, C]
            conf_vec = confidences.unsqueeze(1)  # [B, 1]

            pos_mask = labels                             # 0/1
            neg_mask = 1 - labels

            pos_values = pos_mask * (conf_vec * (1 - self.label_smoothing))
            neg_values = neg_mask * (1 - conf_vec * (1 - self.label_smoothing))

            smoothed = pos_values + neg_values
            return smoothed

class CustomTA_color(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Posterize": (lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(), False),
    "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
    "AutoContrast": (lambda num_bins, height, width: None, False),
    "Equalize": (lambda num_bins, height, width: None, False)
    }

class CustomTA_geometric(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
    }

class Soft_RandomCrop:
    """
    based on:
    https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Soft_Augmentation_for_Image_Classification_CVPR_2023_paper.pdf
    and adapted as in
    https://github.com/Georgsiedel/soft_label_random_augmentation
    
    A class to apply a random crop transformation to an image and compute and return a 
    confidence score for the transformed image dependent on the strength of the crop augmentation.

    Attributes:
    n_class (int): Number of classes for the classification task.
    k (int): Non-linear function parameter for confidence calculation.
    bg_crop (float): Background cropping intensity.
    sigma_crop (float): Standard deviation for drawing the offset.
    """

    def __init__(
        self,
        k: int = 2,
        bg_crop: float = 1.0,
        sigma_crop: float = 0.3,
        #conservative chance level as standard for confidence calculation, assuming no prior knowledge and binary class
        chance: float = 0.5, 
        custom: bool = False    
        ):
        self.chance = chance  
        self.k = k
        self.sigma_crop = sigma_crop
        self.bg_crop = bg_crop
        self.custom = custom

    def draw_offset(
        self,
        sigma: Optional[float] = 0.3,
        limit: Optional[int] = 32,
        n: Optional[int] = 100
    ) -> int:
        """Draws a random offset within a specified limit using a normal distribution.

        Args:
            sigma (float, optional): Standard deviation for the normal distribution. Defaults to 0.3.
            limit (int, optional): Maximum absolute value for the offset. Defaults to 24.
            n (int, optional): Number of attempts to draw a valid offset. Defaults to 100.

        Returns:
            int: The drawn offset within the limit
        """
        for _ in range(n):
            x = torch.randn((1)) * limit * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def compute_visibility(self, dim1: int, dim2: int, tx: float, ty: float) -> float:
        """Computes the visibility of the cropped uimage within the background.

        Args:
            dim1 (int): Height of the image.
            dim2 (int): Width of the image.
            tx (int): Horizontal offset.
            ty (int): Vertical offset.

        Returns:
            float: Visibility ratio of the cropped image.
        """
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Applies the random crop transformation to the given image.
        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns:
            tuple: (cropped Tensor, confidence [0,1])
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected Tensor Image but got {type(image)}")

        # Normalize input to 4D (batch mode)
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
            is_batched = False
        elif image.ndim == 4:
            is_batched = True
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.ndim}D")

        B, C, H, W = image.shape
        device, dtype = image.device, image.dtype

        # Compute offsets once
        ty = self.draw_offset(self.sigma_crop, H)
        tx = self.draw_offset(self.sigma_crop, W)

        # Define crop region
        left, right = tx + W, tx + W * 2
        top, bottom = ty + H, ty + H * 2

        # Compute visibility + confidence once
        if self.custom:
            visibility = self.compute_visibility(H, W, ty, tx)
            confidence = 1 - (1 - self.chance) * (1 - visibility) ** self.k
        else:
            confidence = 1.0

        # Create batched background (vectorized)
        bg = (
            torch.zeros((B, C, H * 3, W * 3), device=device, dtype=dtype)
            * self.bg_crop
            * torch.randn((B, C, 1, 1), device=device, dtype=dtype)
        )

        # Place all images in the center region of the background
        bg[:, :, H:H * 2, W:W * 2] = image

        # Crop all in one go (same crop region for all)
        cropped = bg[:, :, top:bottom, left:right]

        # Restore shape if input was not batched
        if not is_batched:
            cropped = cropped.squeeze(0)

        return cropped, confidence

class On_GPU_Transforms():
    def __init__(self, transforms_orig_gpu, transforms_orig_post, transforms_gen_gpu, transforms_gen_post):

        self.transforms_orig_gpu = transforms_orig_gpu
        self.transforms_orig_post = transforms_orig_post
        self.transforms_gen_gpu = transforms_gen_gpu
        self.transforms_gen_post = transforms_gen_post

    def __call__(self, x, sources, apply):
        
        if self.transforms_orig_gpu == None and self.transforms_gen_gpu == None:
            return x

        x = x.to(device)

        if x.shape[0] == 2 * sources.shape[0]:
            sources = torch.cat([sources, sources], dim=0)
        
        orig_mask = (sources) & (apply)
        if orig_mask.any():
            if apply[sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[orig_mask], 200)
                processed_chunks = [self.transforms_orig_gpu(chunk) for chunk in chunks]
                x[orig_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[orig_mask] = self.transforms_orig_gpu(x[orig_mask])
        
        gen_mask = (~sources) & (apply)
        if gen_mask.any():
            if apply[~sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[gen_mask], 200)
                processed_chunks = [self.transforms_gen_gpu(chunk) for chunk in chunks]
                x[gen_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[gen_mask] = self.transforms_gen_gpu(x[gen_mask])
        
        x = x.cpu()
        if orig_mask.any():
            x[orig_mask] = torch.stack([self.transforms_orig_post(image) for image in x[orig_mask]])
        if gen_mask.any():
            x[gen_mask] = torch.stack([self.transforms_gen_post(image) for image in x[gen_mask]])
        x = x.to(device)

        return x