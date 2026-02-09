import random
import torch
import os
import re
from PIL import Image
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
from run_0 import device
import io
import json
import h5py
import kornia, kornia_rs
from collections import defaultdict, Counter
from torchvision.io import decode_image, ImageReadMode

def extract_gtsrb_validsplit_according_to_tracks(base_trainset,
                                                testsplit,
                                                random_state=0
                                            ):
    
    samples = base_trainset._samples  # (path, class)

    # 1) Build (class, track_id) groups
    track_to_indices = defaultdict(list)

    for idx, (path, cls) in enumerate(samples):
        # filename: 00049_00020.ppm -> track_id = 00049
        track_id = os.path.basename(path).split('_')[0]

        # key includes class to prevent accidental cross-class merging
        key = (cls, track_id)
        track_to_indices[key].append(idx)

    # 2) One label per track (now guaranteed by construction)
    track_keys = list(track_to_indices.keys())
    track_labels = [cls for cls, _ in track_keys]

    # 3) Convert testsplit
    if isinstance(testsplit, float):
        test_size_tracks = testsplit
    else:
        test_size_tracks = testsplit / len(samples)

    # 4) Split tracks (stratified by class)
    train_tracks, val_tracks = train_test_split(
        track_keys,
        test_size=test_size_tracks,
        random_state=random_state,
        stratify=track_labels
    )

    # 5) Expand back to sample indices
    train_indices = [
        idx for key in train_tracks for idx in track_to_indices[key]
    ]
    val_indices = [
        idx for key in val_tracks for idx in track_to_indices[key]
    ]

    return val_indices, train_indices


def custom_collate_fn(batch, batch_transform_orig, batch_transform_gen, image_transform_orig, 
                      image_transform_gen, generated_ratio, batchsize):

    inputs, labels = zip(*batch)
    batch_inputs = torch.stack(inputs)

    # Apply the batched random choice transform
    batch_inputs[:-int(generated_ratio*batchsize)] = batch_transform_orig(batch_inputs[:-int(generated_ratio*batchsize)])
    batch_inputs[-int(generated_ratio*batchsize):] = batch_transform_gen(batch_inputs[-int(generated_ratio*batchsize):])

    for i in range(len(batch_inputs)):
        batch_inputs[i] = image_transform_orig(batch_inputs[i]) if i < (len(batch_inputs)-int(generated_ratio*batchsize)) else image_transform_gen(batch_inputs[i])

    return batch_inputs, torch.tensor(labels)

class SwaLoader():
    def __init__(self, trainloader, batchsize, robust_samples):
        self.trainloader = trainloader
        self.batchsize = batchsize
        self.robust_samples = robust_samples

    def concatenate_collate_fn(self, batch):
        concatenated_batch = []
        for images, label in batch:
            concatenated_batch.extend(images)
        return torch.stack(concatenated_batch), label

    def get_swa_dataloader(self):
        # Create a new DataLoader with the custom collate function

        swa_dataloader = DataLoader(
            dataset=self.trainloader.dataset,
            batch_size=self.batchsize,
            num_workers=0,
            collate_fn=self.concatenate_collate_fn,
            worker_init_fn=self.trainloader.worker_init_fn,
            generator=self.trainloader.generator
        )

        return swa_dataloader

class NumericFolderKorniaDataset(Dataset):
    """
    ImageFolder-like dataset where subfolder names are numeric single-label IDs
    OR multilabel strings like "0_1_0_0".
    Uses kornia.io.load_image for loading images.

    Parameters
    ----------
    root : str
        Path to dataset root (folders inside are class labels).
    multilabel : bool, default=False
        If True, parse folder names as multilabel vectors (separator by label_sep).
    transform : callable, optional
        Transform to apply to the loaded image (expects a torch.Tensor returned from kornia).
    label_sep : str or regex, default='_'
        Separator used when parsing multilabel folder names. Can be a regex pattern
        passed to re.split (for example r"[,_\-]" to accept commas, underscores or dashes).
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

    def __init__(self, root, multilabel=False, transform=None, label_sep="_", try_kornia=False):
        self.root = root
        self.transform = transform
        self.samples = []            # list of (path, label) where label is int or torch.tensor
        self.classes = []            # list of folder-name strings (keeps deterministic ordering)
        self.class_to_idx = {}       # maps folder-name string -> int (single) or tensor (multilabel)
        self.multilabel = bool(multilabel)
        self.label_sep = label_sep
        self.num_labels = None       # number of label positions in multilabel mode
        self.try_kornia = try_kornia

        # iterate through subdirectories deterministically
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            if not self.multilabel:
                # single-label: folder name must be convertible to int
                try:
                    class_id = int(folder)
                except ValueError:
                    raise ValueError(f"Folder name '{folder}' is not a valid integer class label.")
                self.classes.append(folder)  # store folder name string
                self.class_to_idx[folder] = class_id

                # gather images
                for fname in sorted(os.listdir(folder_path)):
                    if fname.lower().endswith(self.IMG_EXTS):
                        img_path = os.path.join(folder_path, fname)
                        self.samples.append((img_path, class_id))

            else:
                # multilabel mode: parse folder name like "0_1_0_0"
                parts = re.split(self.label_sep, folder)
                if len(parts) == 0:
                    raise ValueError(f"Multilabel folder name '{folder}' could not be parsed.")

                # convert parts to integers and check they are 0/1
                try:
                    label_list = [int(p) for p in parts]
                except ValueError:
                    raise ValueError(
                        f"Multilabel folder '{folder}' contains non-integer parts: {parts}"
                    )

                # ensure values are 0 or 1
                if any(x not in (0, 1) for x in label_list):
                    raise ValueError(
                        f"Multilabel folder '{folder}' must contain only 0 or 1 values: got {label_list}"
                    )

                # set/validate number of label positions
                if self.num_labels is None:
                    self.num_labels = len(label_list)
                elif self.num_labels != len(label_list):
                    raise ValueError(
                        f"Inconsistent multilabel lengths: folder '{folder}' has length {len(label_list)} "
                        f"but previous folders have length {self.num_labels}"
                    )
        
                # store as a tuple (cheap Python object), not a torch.Tensor
                label_key = tuple(label_list)

                self.classes.append(folder)
                self.class_to_idx[folder] = label_key

                for fname in sorted(os.listdir(folder_path)):
                    if fname.lower().endswith(self.IMG_EXTS):
                        img_path = os.path.join(folder_path, fname)
                        # store the tuple/list, not a tensor
                        self.samples.append((img_path, label_key))


        # For single-label case, keep classes as sorted integers as before if needed
        if not self.multilabel:
            # convert stored folder-name strings to ints and sort deterministically
            try:
                self.classes = sorted([int(x) for x in self.classes])
            except Exception:
                # fallback: keep folder-name strings
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load image with Kornia
        if self.try_kornia:
            img = kornia.io.load_image(path, kornia.io.ImageLoadType.RGB32)
        else:
            img = Image.open(path).convert("RGB")
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        # Convert label to tensor here (only when needed, inside worker)
        if isinstance(label, (tuple, list)):
            label = torch.tensor(label, dtype=torch.float32)
        elif isinstance(label, int):
            label = int(label)  # keep int for single-label

        return img, label   

class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.labels = []
        for lbl in labels:
            if isinstance(lbl, np.ndarray):
                self.labels.append(torch.from_numpy(lbl).float())
            else:
                self.labels.append(int(lbl))

    def __len__(self):
        return len(self.labels)
    
    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
    
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
class StylizedTensorDataset(Dataset):
    def __init__(self, dataset, stylized_images, stylized_indices):
        """
        A dataset class that maps indices of the original dataset to stylized data when available.

        Args:
            dataset (torchvision.dataset): original dataset
            stylized_images (torch.Tensor): Tensor of stylized images.
            stylized_labels (torch.Tensor): Tensor of stylized labels.
            stylized_indices (list[int]): List of indices in the original dataset that correspond to stylized data.
        """
        self.dataset = dataset
        self.stylized_images = stylized_images

        # Map original dataset indices to the stylized dataset ensures efficient O(1) lookup
        self.index_map = {orig_idx.item(): i for i, orig_idx in enumerate(stylized_indices)} 

    def __len__(self):
        return len(self.dataset)
        
    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        x, _ = self.dataset[idx]
        return x

    def __getitem__(self, idx):
        if idx in self.index_map:
            # Fetch data from the stylized dataset
            stylized_idx = self.index_map[idx]
            x = self.stylized_images[stylized_idx]
            _, y = self.dataset[idx]
        else:
            x, y = self.dataset[idx]
            # Fetch data from the original dataset
        return x, y

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        image, _ = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class HDF5ImageDataset(Dataset):
    """
    Minimal HDF5 dataset safe for all pickling situations.

    - path_images: HDF5 file with an 'images' dataset (or dataset name containing 'images').
    - path_labels: optional HDF5 file; if omitted, 'labels' must be in path_images file.
    - transform: any picklable transform (torchvision transforms are picklable).
    returns image as torch tensor unless pil_instead_of_tensor is True
    """

    def __init__(self, path_images, path_labels=None, transform=None,
                 pil_instead_of_tensor: bool = False):
        self.path_images = path_images
        self.path_labels = path_labels
        self.transform = transform
        self.pil_instead_of_tensor = pil_instead_of_tensor

        # These are file handles, but must not be present during pickling.
        # They are intentionally initialized to None and opened lazily in __getitem__.
        self._fh_img = None
        self._fh_lbl = None

        # Probe the image file once (open/close) to determine keys and length
        if self.path_labels is None:
            self._key_lbl = "labels"
            self._key_img = "images"
        else:
            self._key_lbl = "y"
            self._key_img = "x"
        
        with h5py.File(self.path_images, "r") as f:
            self._length = f[self._key_img].shape[0]

            if "class_to_idx" in f.attrs:
                self.class_to_idx = json.loads(f.attrs["class_to_idx"])
            elif "label_names" in f.attrs:
                self.label_names = json.loads(f.attrs["label_names"])

    def __len__(self):
        return int(self._length)

    def __getitem__(self, idx):
        # Lazy open per process/worker
        if self._fh_img is None:
            self._fh_img = h5py.File(self.path_images, "r")
        if self.path_labels is not None and self._fh_lbl is None:
            self._fh_lbl = h5py.File(self.path_labels, "r")

        entry = self._fh_img[self._key_img][idx]

        # If stored as variable-length uint8 -> .npy bytes, decode. Else copy as numpy.
        if entry.dtype == np.uint8 and entry.ndim == 1:
            img = np.load(io.BytesIO(entry.tobytes()), allow_pickle=False)
        else:
            img = np.array(entry)

        if self.path_labels is not None:
            lbl_entry = self._fh_lbl[self._key_lbl][idx]
        else:
            lbl_entry = self._fh_img[self._key_lbl][idx]
        
        # Handle label for both single-label and multi-label cases.
        if isinstance(lbl_entry, np.ndarray) and lbl_entry.size == 1:
            label = int(lbl_entry.item())
        elif isinstance(lbl_entry, np.ndarray) and lbl_entry.size >= 1:
            label = torch.from_numpy(lbl_entry.astype(np.float32))
        else: # h5py may return numpy scalar types (e.g. np.int32) for scalars
            arr = np.array(lbl_entry)
            if arr.ndim == 0:
                label = int(arr.item())
            else:
                label = torch.from_numpy(arr.astype(np.float32))

        # Optionally convert to PIL Image (ImageFolder-style)
        if self.pil_instead_of_tensor:
            # ensure uint8
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)
        else: #convert directly to torch tensor
            t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
            img = t(img)
            if img.ndim == 2:  # H,W in case of grayscale
                img = img.unsqueeze(0)  # 1,H,W

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # --- defensive pickling: remove any h5py objects before pickling ---
    def __getstate__(self):
        state = self.__dict__.copy()
        # wipe out any h5py objects to avoid pickling errors
        def is_h5py_obj(x):
            try:
                return isinstance(x, (h5py.File, h5py.Dataset, h5py.Group))
            except Exception:
                return False

        for k, v in list(state.items()):
            if is_h5py_obj(v):
                state[k] = None
        # also ensure file paths remain Path objects (they are picklable)
        return state

    def __setstate__(self, state):
        # restore, file handles will be None until first access (in worker/main)
        self.__dict__.update(state)
        self._fh_img = None
        self._fh_lbl = None

    def __del__(self):
        for attr in ("_fh_img", "_fh_lbl"):
            fh = getattr(self, attr, None)
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass

class HDF5ImageDataset_raw(Dataset):
    """
    High-Performance HDF5 Dataset for Raw JPEG Bytes.
    Safe for pickling (Multi-GPU/DataLoader friendly).
    """

    def __init__(self, path_images, path_labels=None, transform=None, 
                 pil_instead_of_tensor: bool = False):
        self.path_images = path_images
        self.path_labels = path_labels
        self.transform = transform
        self.pil_instead_of_tensor = pil_instead_of_tensor

        # File handles (lazy load)
        self._fh_img = None
        self._fh_lbl = None
        
        # Determine Keys and Length immediately
        if self.path_labels is None:
            self._key_lbl = "labels"
            self._key_img = "images"
        else:
            self._key_lbl = "y"
            self._key_img = "x"

        # Probe file once to get metadata
        with h5py.File(self.path_images, "r") as f:
            self._length = f[self._key_img].shape[0]

            # Load metadata
            if "class_to_idx" in f.attrs:
                self.class_to_idx = json.loads(f.attrs["class_to_idx"])
            elif "label_names" in f.attrs:
                self.label_names = json.loads(f.attrs["label_names"])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # 1. Lazy Open (One handle per worker process)
        if self._fh_img is None:
            self._fh_img = h5py.File(self.path_images, "r")
            self._dset_img = self._fh_img[self._key_img]
        
        if self.path_labels is not None and self._fh_lbl is None:
            self._fh_lbl = h5py.File(self.path_labels, "r")
            self._dset_lbl = self._fh_lbl[self._key_lbl]
        else:
            self._dset_lbl = self._fh_img[self._key_lbl]

        # 2. Retrieve Data
        # Get raw bytes from HDF5 (returned as numpy uint8 array)
        img_bytes_np = self._dset_img[idx]
        lbl_entry = self._dset_lbl[idx]

        # 3. Decode Image
        if self.pil_instead_of_tensor:
            # Fallback: Convert bytes -> PIL
            # Note: This is slower than decode_image, but required for some legacy transforms
            img = Image.open(io.BytesIO(img_bytes_np)).convert("RGB")
        else:
            # Fast Path: Torchvision C++ Decoder
            # Zero-copy from numpy to torch tensor
            img_tensor_buffer = torch.from_numpy(img_bytes_np)
            
            # Decode JPEG/PNG -> Tensor [3, H, W]
            # mode=ImageReadMode.RGB ensures 3 channels (handles grayscale automatically)
            img = decode_image(img_tensor_buffer, mode=ImageReadMode.RGB)

            # Convert to float32 [0, 1] range standard for PyTorch transforms
            # (decode_image returns uint8 [0, 255])
            img = img.to(dtype=torch.float32).div(255.0)

        # 4. Handle Labels (Scalar vs Multi-label)
        if isinstance(lbl_entry, np.ndarray):
            if lbl_entry.size == 1:
                # Single Scalar Label
                label = int(lbl_entry.item())
            else:
                # Multi-label / Tensor
                label = torch.from_numpy(lbl_entry.astype(np.float32))
        else:
            # Scalar types (e.g. h5py returning int32 directly)
            arr = np.array(lbl_entry)
            if arr.ndim == 0:
                label = int(arr.item())
            else:
                label = torch.from_numpy(arr.astype(np.float32))

        # 5. Apply Transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # --- Pickling Safety (Same as your original) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove h5py objects
        for k in ["_fh_img", "_fh_lbl", "_dset_img", "_dset_lbl"]:
            if k in state:
                state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh_img = None
        self._fh_lbl = None
        self._dset_img = None
        self._dset_lbl = None

    def __del__(self):
        # Close handles if they exist
        if hasattr(self, "_fh_img") and self._fh_img is not None:
            self._fh_img.close()
        if hasattr(self, "_fh_lbl") and self._fh_lbl is not None:
            self._fh_lbl.close()

class CustomDataset(Dataset):
    def __init__(self, np_images, testset, resize, preprocessing):
        # Load images
        self.np_images = np.memmap(np_images, dtype=np.float32, mode='r') if isinstance(np_images, str) else np_images
        self.resize = resize
        self.preprocessing = preprocessing
        self.set = testset

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.preprocessing(self.np_images[index])
        if self.resize == True:
            image = transforms.Resize(224, antialias=True)(image)

        _, label = self.set[index]

        return image, label


class ReproducibleBalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size, epoch):
        super(ReproducibleBalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)
        self.current_epoch = epoch

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch
        
    def generate_indices_order(self, num_original, num_generated, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices_original = list(range(num_original))
        indices_generated = list(range(num_original, num_generated + num_original))

        local_rng.shuffle(indices_original)
        local_rng.shuffle(indices_generated)

        return indices_original, indices_generated

    def __iter__(self):

        # Create a single permutation for the whole epoch which is reproducible.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm, generated_perm = self.generate_indices_order(self.num_original, self.num_generated, self.current_epoch)
        self.current_epoch += 1

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, start in enumerate(batch_starts):

            # Slicing the permutation to get batch indices, avoiding going out of bound
            original_indices = original_perm[min(i * self.num_original_batch, self.num_original) : min((i+1) * self.num_original_batch, self.num_original)]
            generated_indices = generated_perm[min(i * self.num_generated_batch, self.num_generated) : min((i+1) * self.num_generated_batch, self.num_generated)]

            # Combine
            batch_indices = original_indices + generated_indices
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

class GroupedAugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, original_dataset, generated_dataset, 
                 transforms_basic, transforms_batch_gen, transforms_batch_orig, transforms_iter_orig_after_style, transforms_iter_gen_after_style,
                 transforms_iter_orig_after_nostyle, transforms_iter_gen_after_nostyle, robust_samples=0, epoch=0):
        
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.transforms_basic = transforms_basic
        self.transforms_batch_gen = transforms_batch_gen
        self.transforms_batch_orig = transforms_batch_orig
        self.transforms_iter_orig_after_style = transforms_iter_orig_after_style
        self.transforms_iter_gen_after_style = transforms_iter_gen_after_style
        self.transforms_iter_orig_after_nostyle = transforms_iter_orig_after_nostyle
        self.transforms_iter_gen_after_nostyle = transforms_iter_gen_after_nostyle

        self.robust_samples = robust_samples

        # Compute cache sizes (i.e. block sizes) based on the batch transform parameters.
        if transforms_batch_gen:
            self.cache_size_gen = int(transforms_batch_gen.batch_size / transforms_batch_gen.stylized_ratio)
        else:
            self.cache_size_gen = 1
        if transforms_batch_orig:
            self.cache_size_orig = int(transforms_batch_orig.batch_size / transforms_batch_orig.stylized_ratio)
        else:
            self.cache_size_orig = 1

        self.num_original = len(original_dataset) if original_dataset else 0
        self.num_generated = len(generated_dataset) if generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

        # Initialize empty caches. They map the global (domain) index to (image, label, style_flag).
        self.cache_orig = {}
        self.cache_gen = {}

        # Generate reproducible permutation lists for each domain.
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        """
        At the beginning of each epoch, regenerate the random ordering for each domain and clear caches.
        """
        self.original_perm, self.generated_perm = self.generate_indices_order(self.num_original, self.num_generated, epoch)
        self.cache_orig.clear()
        self.cache_gen.clear()
        
    def generate_indices_order(self, num_original, num_generated, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices_original = list(range(num_original))
        indices_generated = list(range(num_original, num_generated + num_original))

        local_rng.shuffle(indices_original)
        local_rng.shuffle(indices_generated)

        return indices_original, indices_generated
    
    def __getitem__(self, idx):
        """
        Retrieve the (transformed) item corresponding to a global index.
        
        For original images, the global index is used as is; for generated images,
        the index is adjusted by subtracting num_original. If the requested item is not
        in the cache, the cache is cleared and filled by processing a block (of size cache_size)
        from the corresponding permutation starting at the requested index’s position.
        Then, an iterative transform (after the batch transform) is applied based on the style flag.
        """
        # Determine domain.
        if idx < self.num_original:
            dataset_specific_index = idx  # for original images
            perm = self.original_perm
            cache = self.cache_orig
            cache_size = self.cache_size_orig
            dataset = self.original_dataset
            transform_batch = self.transforms_batch_orig
            transforms_iter_after_style = self.transforms_iter_orig_after_style
            transforms_iter_after_nostyle = self.transforms_iter_orig_after_nostyle
        else:
            dataset_specific_index = idx - self.num_original  # for generated images, adjust index
            perm = self.generated_perm
            cache = self.cache_gen
            cache_size = self.cache_size_gen
            dataset = self.generated_dataset
            transform_batch = self.transforms_batch_gen
            transforms_iter_after_style = self.transforms_iter_gen_after_style
            transforms_iter_after_nostyle = self.transforms_iter_gen_after_nostyle

        if transform_batch == None:
            x, y = dataset[dataset_specific_index]
            style_flag = False
            
        else:
            # If the requested global index is cached, retrieve it.
            if idx not in cache:

                # Not in cache. Find the position of this global index in the permutation.
                try:
                    pos = perm.index(idx)
                except ValueError:
                    pos = 0
                # Get the block of indices: from the found position up to cache_size items.
                indices_block = perm[pos: pos + cache_size]

                items = [dataset[i - self.num_original] for i in indices_block]
                images, labels = zip(*items)
                images = torch.stack(images)

                images, style_mask = transform_batch(images)

                # Clear the cache and fill it with the new block.
                cache.clear()

                for i, d_idx in enumerate(indices_block):
                    cache[d_idx] = (images[i], labels[i], style_mask[i])
            
            x, y, style_flag = cache[idx]

        # Apply the iterative (per-image) transform based on whether the image was styled.
        transform_iter = (transforms_iter_after_style if style_flag else transforms_iter_after_nostyle)
        
        aug = transforms.Compose([self.transforms_basic, transform_iter])

        # Handle robust_samples if needed.
        if self.robust_samples == 0:
            return aug(x), y
        
        elif self.robust_samples == 1:
            x0, _ = dataset[dataset_specific_index]
            return (x0, aug(x)), y
        
        elif self.robust_samples == 2:
            x0, _ = dataset[dataset_specific_index]
            return (x0, aug(x), aug(x)), y

    def __len__(self):
        return self.total_size

class BalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size):
        super(BalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch

    def __iter__(self):

        # Create a single permutation for the whole epoch.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm = torch.randperm(self.num_original)
        generated_perm = torch.randperm(self.num_generated) + self.num_original

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, start in enumerate(batch_starts):

            # Slicing the permutation to get batch indices, avoiding going out of bound
            original_indices = original_perm[min(i * self.num_original_batch, self.num_original) : min((i+1) * self.num_original_batch, self.num_original)]
            generated_indices = generated_perm[min(i * self.num_generated_batch, self.num_generated) : min((i+1) * self.num_generated_batch, self.num_generated)]

            # Combine
            batch_indices = torch.cat((original_indices, generated_indices))
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices.tolist()

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, stylized_original_dataset, stylized_generated_dataset, style_mask, 
                 transforms_basic, transforms_orig_after_style, transforms_gen_after_style, 
                 transforms_orig_after_nostyle, transforms_gen_after_nostyle, robust_samples=0):
        self.stylized_original_dataset = stylized_original_dataset
        self.stylized_generated_dataset = stylized_generated_dataset
        self.style_mask = style_mask
        self.transforms_basic = transforms_basic
        self.transforms_orig_after_style = transforms_orig_after_style
        self.transforms_gen_after_style = transforms_gen_after_style
        self.transforms_orig_after_nostyle = transforms_orig_after_nostyle
        self.transforms_gen_after_nostyle = transforms_gen_after_nostyle
        self.robust_samples = robust_samples

        self.num_original = len(stylized_original_dataset) if stylized_original_dataset else 0
        self.num_generated = len(stylized_generated_dataset) if stylized_generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

        assert len(style_mask) == self.num_original + self.num_generated
    
    def handle_label(self, y):
        """
        Handle label for both single-label and multi-label cases.
        - If y is scalar-like -> return int(y)
        - Else -> return float tensor
        """
        if torch.is_tensor(y):
            if y.ndim == 0 or (y.ndim == 1 and y.numel() == 1):
                # Single scalar tensor
                return int(y.item())
            else:
                # Multi-label or continuous tensor
                return y.to(torch.float32)
        else:
            # Non-tensor case (e.g., int from dataset)
            return int(y)

    def __getitem__(self, idx):

        is_stylized = self.style_mask[idx]

        if idx < self.num_original:
            x, y = self.stylized_original_dataset[idx]
            aug = self.transforms_orig_after_style if is_stylized else self.transforms_orig_after_nostyle
        else:
            x, y = self.stylized_generated_dataset[idx - self.num_original]
            aug = self.transforms_gen_after_style if is_stylized else self.transforms_gen_after_nostyle

        augment = transforms.Compose([self.transforms_basic, aug])

        y = self.handle_label(y)

        if self.robust_samples == 0:
            return augment(x), y
    
        elif self.robust_samples >= 1:
            if idx < self.num_original:
                x0 = self.stylized_original_dataset.getclean(idx)
            else:
                x0 = self.stylized_generated_dataset.getclean(idx - self.num_original)

            if self.robust_samples == 1:
                return (self.transforms_basic(x0), augment(x)), y
            elif self.robust_samples == 2:
                return (self.transforms_basic(x0), augment(x), augment(x)), y

    def __len__(self):
        return self.total_size
    
class BasicAugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, original_dataset, generated_dataset, transforms_basic, robust_samples=0):
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.transforms_basic = transforms_basic
        self.robust_samples = robust_samples

        self.num_original = len(original_dataset) if original_dataset else 0
        self.num_generated = len(generated_dataset) if generated_dataset else 0
        self.total_size = self.num_original + self.num_generated
    
    def handle_label(self, y):
        """
        Handle label for both single-label and multi-label cases.
        - If y is scalar-like -> return int(y)
        - Else -> return float tensor
        """
        if torch.is_tensor(y):
            if y.ndim == 0 or (y.ndim == 1 and y.numel() == 1):
                # Single scalar tensor
                return int(y.item())
            else:
                # Multi-label or continuous tensor
                return y.to(torch.float32)
        else:
            # Non-tensor case (e.g., int from dataset)
            return int(y)

    def __getitem__(self, idx):

        if idx < self.num_original:
            x, y = self.original_dataset[idx]
        else:
            x, y = self.generated_dataset[idx - self.num_original]

        y = self.handle_label(y)

        if self.robust_samples == 0:
            return self.transforms_basic(x), y
        elif self.robust_samples == 1:
            return (self.transforms_basic(x), self.transforms_basic(x)), y
        elif self.robust_samples == 2:
            return (self.transforms_basic(x), self.transforms_basic(x), self.transforms_basic(x)), y

    def __len__(self):
        return self.total_size

class StyleDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, file)
            for file in os.listdir(root_dir)
            if file.endswith(".jpg")
        ]
        if dataset_type in ["CIFAR10", "CIFAR100", "GTSRB", 'DermaMNIST']:
            self.transform = transforms.Resize((32, 32), antialias=True)
        elif dataset_type in ["TinyImageNet", "EuroSAT", "Wafermap", "PCAM"]:
            self.transform = transforms.Resize((64, 64), antialias=True)
        elif dataset_type in ["NEU-surface-defect"]:
            self.transform = transforms.Resize((128, 128), antialias=True)
        elif dataset_type in ["ImageNet", 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 
                              'Describable-Textures', 'Flickr-Material', 'SynthiCAD']:
            self.transform = transforms.Resize((224, 224), antialias=True)
        else:
            raise AttributeError(f"Dataset: {dataset_type} is an unrecognized dataset")
        self.transform = transforms.Compose([self.transform, transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image