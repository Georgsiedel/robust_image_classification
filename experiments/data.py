import random
import os
import time
import json
import gc
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
import kornia
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, ConcatDataset, RandomSampler, BatchSampler, DataLoader, TensorDataset
import numpy as np
import experiments.custom_transforms as custom_transforms
from run_0 import device
from experiments.utils import plot_images, CsvHandler
from experiments.custom_datasets import SubsetWithTransform, NumpyDataset, AugmentedDataset, HDF5ImageDataset
from experiments.custom_datasets import CustomDataset, NumericFolderKorniaDataset, BasicAugmentedDataset 
from experiments.custom_datasets import BalancedRatioSampler, StyleDataset, extract_gtsrb_validsplit_according_to_tracks


def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1, verbose=False):

    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == 'CIFAR100':
            mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1).to(device)
        elif dataset in ['ImageNet', 'TinyImageNet', 'ImageNet-100']:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            if verbose:
                print('no normalization values set for this dataset, scaling to [-1,1]')
    else:
        mean = 0
        std = 1

    return mean, std

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    global fixed_worker_rng #impulse noise augmentations sk-learn function needs a separate rng for reproducibility
    fixed_worker_rng = np.random.default_rng()

def get_class_to_idx_recursive(obj):
    while obj is not None:
        if hasattr(obj, "class_to_idx"):
            return obj.class_to_idx

        # Prioritize subset → dataset traversal
        if hasattr(obj, "subset"):
            obj = obj.subset
        elif hasattr(obj, "dataset"):
            obj = obj.dataset
        elif hasattr(obj, "set"):
            obj = obj.dataset
        else:
            break

    raise AttributeError("No class_to_idx found in dataset hierarchy.")

def extract_labels(dataset):
    """
    Return a flat list of labels (one entry per sample) for any dataset.
    """
    if hasattr(dataset, 'targets'):
        # most vision datasets put labels here
        return list(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return list(dataset.labels)
    elif hasattr(dataset, 'samples'):
        # ImageFolder and friends
        return [s[1] for s in dataset.samples]
    else:
        # worst case: iterate—but still O(N), same as splitting
        return [dataset[i][1] for i in range(len(dataset))]

def extract_num_classes(dataset, labels=None):
    """
    Return the number of classes in the dataset.
    Supports scalar or multilabel (multi-hot vector) labels.
    """
    # Use dataset info if available
    if hasattr(dataset, 'classes'):
        return len(dataset.classes)
    if hasattr(dataset, 'class_to_idx'):
        return len(dataset.class_to_idx)
    if hasattr(dataset, 'label_names'):
        return len(dataset.label_names)
    
    # Otherwise get labels if not provided
    print('Manually counting number of classes...')

    if labels is None:
        labels = extract_labels(dataset)

    # If labels are multilabel vectors (list of arrays/tensors)    
    if (
        isinstance(labels, (list, tuple))
        and len(labels) > 0
        and (
            (hasattr(labels[0], 'ndim') and labels[0].ndim == 1 and len(labels[0]) > 1)
            or (isinstance(labels[0], (list, tuple)) and len(labels[0]) > 1)
        )
    ):
        return len(labels[0])

    # Otherwise treat as scalar labels, count unique
    unique_labels = set()
    for lbl in labels:
        # If tensor or numpy scalar convert to Python scalar
        if hasattr(lbl, 'item'):
            unique_labels.add(lbl.item())
        else:
            unique_labels.add(lbl)
    return len(unique_labels)

class DataLoading():
    def __init__(self, dataset, validontest=True, epochs=200, 
                 resize = False, run=0, number_workers=0, kaggle=False):
        self.dataset = dataset
        self.resize = resize
        self.run = run
        self.epochs = epochs
        self.validontest = validontest
        self.number_workers = number_workers
        self.kaggle = kaggle

        if dataset in ['CIFAR10', 'CIFAR100', 'DermMNIST', 'GTSRB','ImageNet', 'ImageNet-100', 'KITTI_RoadLane', 
                       'KITTI_Distance_Multiclass', 'TreeSAT', 'Casting-Product-Quality', 
                       'Describable-Textures', 'Flickr-Material', 'SynthiCAD']:
            self.factor = 1
        elif dataset in ['TinyImageNet', 'EuroSAT', 'Wafermap']:
            self.factor = 2
        elif dataset in ['PCAM']:
            self.factor = 2
        else:
            self.factor = 1
        
        if dataset in ['WaferMap', 'KITTI_Distance_Multiclass']:
            self.multilabel = True
        else:
            self.multilabel = False
        
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paths.json")
        with open(file_path, "r") as f:
            paths = json.load(f)

        suffix = "_kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else ""  # use env to detect Kaggle

        def resolve_path(key, suffix=""):
            # First try the suffixed path
            p = paths.get(f"{key}{suffix}")
            if not p:
                p = paths[key]
            
            # Make absolute
            if not os.path.isabs(p):
                # Relative to the repository root
                repo_root = os.path.dirname(os.path.dirname(__file__))
                p = os.path.abspath(os.path.join(repo_root, p))
            return p

        self.data_path = resolve_path("data", suffix)
        self.c_labels_path = resolve_path("c_labels", suffix)
        self.trained_models_path = resolve_path("trained_models", suffix)
        self.style_feats_path = resolve_path("style_feats", suffix)
        self.write_data_path = resolve_path("write_data", suffix)

    def create_transforms(self, train_aug_strat_orig, train_aug_strat_gen=None, 
                          style_orig={'probability': 0.0, 'alpha_min': 1.0, 'alpha_max': 1.0}, 
                          style_gen={'probability': 0.0, 'alpha_min': 1.0, 'alpha_max': 1.0}, 
                          style_and_aug_orig=True, style_and_aug_gen=True, 
                          RandomEraseProbability=0.0, 
                          aggressive_soft_crop=False,
                          stylization_first=False,
                          minibatchsize=8,
                          label_smoothing=0.0):
        self.train_aug_strat_orig = train_aug_strat_orig
        self.train_aug_strat_gen = train_aug_strat_gen
        self.style_orig = style_orig
        self.style_gen = style_gen 
        self.style_and_aug_orig = style_and_aug_orig,
        self.style_and_aug_gen = style_and_aug_gen
        self.stylization_first = stylization_first
        self.RandomEraseProbability = RandomEraseProbability
        self.minibatchsize = minibatchsize
        self.label_smoothing = label_smoothing

        # list of all data transformations used
        self.soft_crop = custom_transforms.Soft_RandomCrop(custom=True) if aggressive_soft_crop else None

        t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
        t_no_scaling = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32)])
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        p32 = transforms.Pad(2)
        c32_mnist = transforms.RandomCrop(32, padding=2)
        p64 = transforms.Pad(6)
        c64_WM = transforms.RandomCrop(64, padding=6)
        c96 = transforms.RandomCrop(96, padding=12)
        c224 = transforms.RandomCrop(224, padding=28)
        flip = transforms.RandomHorizontalFlip()
        flip_v = transforms.RandomVerticalFlip()
        r32 = transforms.Resize((32,32), antialias=True)
        r64 = transforms.Resize((64,64), antialias=True)
        r224 = transforms.Resize(224, antialias=True)
        r232 = transforms.Resize(232, antialias=True)
        r320 = transforms.Resize((320,1056), antialias=True)
        r384 = transforms.Resize((384,1280), antialias=True)
        cc224 = transforms.CenterCrop(224)
        rrc176 = transforms.RandomResizedCrop(176, antialias=True)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        self.re = transforms.RandomErasing(p=self.RandomEraseProbability, scale=(0.02, 0.4))
        

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 
                       'Describable-Textures', 'Flickr-Material']:
            #see https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/ for FixRes recipe
            #no tensor because HDF5 dataset class or Imagefolder with kornia loader (tensor output)
            self.transforms_preprocess_train = transforms.Compose([rrc176]) #rrc224 rrc176
            self.transforms_preprocess_test = transforms.Compose([r232, cc224])
        elif self.dataset in ['NEU-surface-defect']:
            self.transforms_preprocess_train = transforms.Compose([transforms.RandomResizedCrop(112, antialias=True)])
            self.transforms_preprocess_test = transforms.Compose([transforms.Resize(128, antialias=True)])
        elif self.dataset in ['SynthiCAD']:
            #no tensor because HDF5 dataset class or Imagefolder with kornia loader (tensor output)
            self.transforms_preprocess_train = transforms.Compose([rrc176, 
                                                        custom_transforms.ExpandGrayscaleTensorTo3Channels()]) #rrc224 rrc176
            self.transforms_preprocess_test = transforms.Compose([r232, cc224, 
                                                        custom_transforms.ExpandGrayscaleTensorTo3Channels()])
        elif self.dataset in ['KITTI_RoadLane', 'KITTI_Distance_Multiclass']: #no tensor because HDF5 dataset class
            self.transforms_preprocess_train = transforms.Compose([r320])
            self.transforms_preprocess_test = transforms.Compose([r384])
        elif self.dataset == 'GTSRB':
            self.transforms_preprocess_train = transforms.Compose([t, r32])
            self.transforms_preprocess_test = transforms.Compose([t, r32])
        elif self.dataset in ['DermaMNIST']:
            self.transforms_preprocess_train = transforms.Compose([t, p32])
            self.transforms_preprocess_test = transforms.Compose([t, p32])
        elif self.dataset == 'WaferMap':
            #https://github.com/Junliangwangdhu/WaferMap/tree/master
            self.transforms_preprocess_train = transforms.Compose([
                t_no_scaling,
                custom_transforms.DivideBy3(),
                custom_transforms.ExpandGrayscaleTensorTo3Channels(), #directly converts to 3 channels
                p64
            ])
            self.transforms_preprocess_test = transforms.Compose([
                t_no_scaling,
                custom_transforms.DivideBy3(),
                custom_transforms.ExpandGrayscaleTensorTo3Channels(), #directly converts to 3 channels
                p64
            ])
        elif self.dataset == 'PCAM':
            self.transforms_preprocess_train =  transforms.RandomCrop(64)
            self.transforms_preprocess_test = transforms.CenterCrop(64)
            
        else:
            self.transforms_preprocess_train = transforms.Compose([t])
            self.transforms_preprocess_test = transforms.Compose([t])
        
        if self.resize == True and self.dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 
                       'Describable-Textures', 'Flickr-Material', 'SynthiCAD']:
            self.transforms_preprocess_train = transforms.Compose([rrc224]) 
            self.transforms_preprocess_test = transforms.Compose([r232, cc224])
        elif self.resize == True and self.dataset in ['KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
            self.transforms_preprocess_train = transforms.Compose([r224]) 
            self.transforms_preprocess_test = transforms.Compose([r224])
        elif self.resize == True:
            self.transforms_preprocess_train = transforms.Compose([self.transforms_preprocess_train, r224]) 
            self.transforms_preprocess_test = transforms.Compose([self.transforms_preprocess_test, r224])

        # basic crop and flip augmentations used for training only
        basic_list = []
        if self.dataset not in ['GTSRB']:
            basic_list.append(flip)
        if self.dataset in ['TreeSAT', 'Casting-Product-Quality', 'SynthiCAD','EuroSAT','PCAM','WaferMap','Describable-Textures','NEU-surface-defect', 'DermaMNIST']:
            basic_list.append(flip_v)
        if self.dataset in ['CIFAR10', 'CIFAR100', 'GTSRB']:
            basic_list.append(c32)
        elif self.dataset in ['EuroSAT','PCAM','TinyImageNet']:
            basic_list.append(c64)
        elif self.dataset == 'WaferMap':
            basic_list.append(c64_WM)
        elif self.dataset in ['DermaMNIST']:
            basic_list.append(c32_mnist)

        if (self.resize or aggressive_soft_crop):
            # remove crop/resize logic if resizing or soft crop are applied
            # do not remove for WaferMap and DermaMNIST as padding is necessary to get to 32x32 or 64x64 (generated image size)
            basic_list = [tr for tr in basic_list if tr not in {c32, c64}]
        
        if basic_list:
            self.transforms_basic = transforms.Compose(basic_list)
        else:
            self.transforms_basic = transforms.Identity()     

        transform_manager_orig = custom_transforms.TransformFactory(self.re, self.soft_crop, self.style_feats_path, self.train_aug_strat_orig, 
                                                                    self.style_orig, self.style_and_aug_orig, self.dataset, self.minibatchsize)
        transform_manager_gen = custom_transforms.TransformFactory(self.re, self.soft_crop, self.style_feats_path, self.train_aug_strat_gen, 
                                                                    self.style_gen, self.style_and_aug_gen, self.dataset, self.minibatchsize)
        if self.stylization_first:
            self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = transform_manager_orig.get_transforms_style_first()
            self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = transform_manager_gen.get_transforms_style_first()
        else:
            self.stylization_orig, self.transforms_orig_after_style = transform_manager_orig.get_transforms()
            self.stylization_gen, self.transforms_gen_after_style = transform_manager_gen.get_transforms()
        
        
    def update_transforms(self, stylize_prob_orig=None, stylize_prob_syn=None, alpha_min_orig=None, alpha_min_syn=None, 
                          style_and_aug_orig=None, style_and_aug_syn=None, RandomEraseProbability=None, aggressive_soft_crop=None):
        
        if RandomEraseProbability is not None:
            self.re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4))

        if aggressive_soft_crop is not None:
            max_value = 0.5 if self.multilabel else 0.0 #set to 0.5 for multilabel where every class is binary
            # standard value for chance in class is the conservative 0.5, when no num_classes known
            kwargs = {"chance": max(1 / self.num_classes, max_value)} if hasattr(self, "num_classes") else {}
            self.soft_crop = custom_transforms.Soft_RandomCrop(custom=True, **kwargs) if aggressive_soft_crop else None

        if stylize_prob_orig is not None:
            self.style_orig['probability'] = stylize_prob_orig
        if stylize_prob_syn is not None:
            self.style_gen['probability'] = stylize_prob_syn
        if alpha_min_orig is not None:
            self.style_gen['alpha_min'] = alpha_min_orig
        if alpha_min_syn is not None:
            self.style_orig['alpha_min'] = alpha_min_syn
        if style_and_aug_orig is not None:
            self.style_and_aug_orig = style_and_aug_orig
        if style_and_aug_syn is not None:
            self.style_and_aug_gen = style_and_aug_syn

        transform_manager_orig = custom_transforms.TransformFactory(self.re, self.soft_crop, self.style_feats_path, self.train_aug_strat_orig, 
                                                                    self.style_orig, self.style_and_aug_orig, self.dataset, self.minibatchsize)
        transform_manager_gen = custom_transforms.TransformFactory(self.re, self.soft_crop, self.style_feats_path, self.train_aug_strat_gen, 
                                                                    self.style_gen, self.style_and_aug_gen, self.dataset, self.minibatchsize)
        if self.stylization_first:
            self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = transform_manager_orig.get_transforms_style_first()
            self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = transform_manager_gen.get_transforms_style_first()
        else:
            self.stylization_orig, self.transforms_orig_after_style = transform_manager_orig.get_transforms()
            self.stylization_gen, self.transforms_gen_after_style = transform_manager_gen.get_transforms()
    
    def load_base_data(self, test_only=False):

        if self.validontest:

            if self.dataset in ['ImageNet']:
                self.testset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/val'),
                                                                transform=self.transforms_preprocess_test,
                                                                loader=custom_transforms.safe_kornia_loader
                                                                )
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'),
                                                                loader=custom_transforms.safe_kornia_loader)

            elif self.dataset in ['TinyImageNet', 'ImageNet-100', 'NEU-surface-defect']:
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_val.h5',
                                                transform=transforms.Compose([self.transforms_preprocess_test]))
                if test_only:
                    self.base_trainset = None
                else:
                    #self.base_trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_train.h5')
                    self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'),
                                                                loader=custom_transforms.safe_kornia_loader)

            elif self.dataset in ['CIFAR10', 'CIFAR100']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=False, download=True,
                                        transform=self.transforms_preprocess_test)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
            
            elif self.dataset in ['GTSRB']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='test', download=True,
                                        transform=self.transforms_preprocess_test)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='train', download=True)

            elif self.dataset in ['DermaMNIST']:
                data=np.load(os.path.join(f'{self.data_path}/dermamnist.npz'))

                if test_only:
                    self.base_trainset = None
                else:
                    x_train = data["train_images"]
                    x_val = data["val_images"]
                    y_train = data["train_labels"].squeeze()
                    y_val = data["val_labels"].squeeze()
                    base_trainset = NumpyDataset(x_train, y_train)
                    base_valset = NumpyDataset(x_val, y_val)
                    self.base_trainset = ConcatDataset([base_trainset, base_valset])
                
                x_test = data["test_images"]
                y_test  = data["test_labels"].squeeze()
                self.testset = NumpyDataset(x_test, y_test, self.transforms_preprocess_test) #transforms already done

            elif self.dataset in ['PCAM']:               
                
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_test_x.h5',
                                                f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_test_y.h5',
                                                transform=self.transforms_preprocess_test)
                if test_only:
                    self.base_trainset = None
                else:
                    valset = HDF5ImageDataset(f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_valid_x.h5',
                                                f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_valid_y.h5')
                    trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_train_x.h5',
                                                f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_train_y.h5')
                    self.base_trainset = ConcatDataset([trainset, valset])
            
            elif self.dataset == 'EuroSAT':
                print('EuroSAT has no predefined test split. Using a custom seeded random split.')
                load_helper = getattr(torchvision.datasets, self.dataset)
                full_set = load_helper(root=os.path.abspath(f'{self.data_path}'), download=True)

                all_labels = extract_labels(full_set)
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=0.2,
                random_state=0) #always with 0 seed - testset split should always be the same

                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = Subset(full_set, train_indices)
                self.testset = SubsetWithTransform(Subset(full_set, val_indices), self.transforms_preprocess_test)
                
                self.num_classes = extract_num_classes(self.testset, labels=all_labels)
                return    
                        
            elif self.dataset == 'WaferMap':
                print('WaferMap has no predefined test split. Using a custom seeded random split.')
                data=np.load(os.path.join(f'{self.data_path}/MixedWM38.npz'))
                x = data["arr_0"]
                y = data["arr_1"]
                
                x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.2,
                random_state=0)

                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = NumpyDataset(x_train, y_train)
                self.testset = NumpyDataset(x_test, y_test, self.transforms_preprocess_test) #transforms already done
            
            elif self.dataset in ['TreeSAT', 'Casting-Product-Quality', 'KITTI_RoadLane']:
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_test.h5',
                                                transform=self.transforms_preprocess_test)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_train.h5')
            
            elif self.dataset in ['SynthiCAD']:
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_test.h5',
                                                transform=self.transforms_preprocess_test)
                if test_only:
                    self.base_trainset = None
                else:
                    valset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_valid.h5')
                    trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_train.h5')
                    self.base_trainset = ConcatDataset([trainset, valset])
            
            elif self.dataset in ['KITTI_Distance_Multiclass', 'Describable-Textures', 'Flickr-Material']:
                print(f'{self.dataset} has no predefined test split. Using a custom seeded random split.')
                full_set = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}.h5')
                all_labels = extract_labels(full_set)

                testsize = 0.5 if self.dataset in ['Flickr-Material'] else 0.2
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=testsize,
                random_state=0) #always with 0 seed - testset split should always be the same

                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = Subset(full_set, train_indices)
                self.testset = SubsetWithTransform(Subset(full_set, val_indices), self.transforms_preprocess_test)
                
                self.num_classes = extract_num_classes(self.testset, labels=all_labels)
                return 

            else:
                print('Dataset not loadable')

            self.num_classes = extract_num_classes(self.testset)

        else:
            if self.dataset in ['ImageNet']:
                base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'),
                                                                loader=custom_transforms.safe_kornia_loader)
            elif self.dataset in ['TinyImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 'KITTI_RoadLane', 'NEU-surface-defect']:
                base_trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_train.h5')
            elif self.dataset in ['CIFAR10', 'CIFAR100']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
            elif self.dataset in ['GTSRB']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='train', download=True)
                
                #extract custom validation split according to tracks
                testsplit = 0.2 if len(base_trainset) <= 50000 else 10000
                val_indices, train_indices = extract_gtsrb_validsplit_according_to_tracks(base_trainset, testsplit, random_state=self.run)

                if test_only == False:
                    self.base_trainset = Subset(base_trainset, train_indices)

                self.testset = SubsetWithTransform(Subset(base_trainset, val_indices), self.transforms_preprocess_test)
                
                self.num_classes = extract_num_classes(self.testset)
                return

            elif self.dataset in ['PCAM']:
                if test_only == False:
                    self.base_trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_train_x.h5',
                                                    f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_train_y.h5')
                
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_valid_x.h5',
                                                f'{self.data_path}/{self.dataset.lower()}/camelyonpatch_level_2_split_valid_y.h5', 
                                                transform=self.transforms_preprocess_test)
                    
                self.num_classes = 2
                return  #already features train/val split, so we can return
            
            elif self.dataset in ['DermaMNIST']:

                data=np.load(os.path.join(f'{self.data_path}/dermamnist.npz'))

                if test_only:
                    self.base_trainset = None
                else:
                    x_train = data["train_images"]
                    y_train = data["train_labels"].squeeze()
                    self.base_trainset = NumpyDataset(x_train, y_train)
                
                x_val = data["val_images"]
                y_val = data["val_labels"].squeeze()
                self.testset = NumpyDataset(x_val, y_val, self.transforms_preprocess_test) #transforms already done
                self.num_classes = 7
                return #already features train/val split, so we can return

            elif self.dataset in ['SynthiCAD']:
                self.testset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_valid.h5',
                                                transform=self.transforms_preprocess_test)
                self.base_trainset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}_train.h5')
                self.num_classes = extract_num_classes(self.testset)
                return #already features train/val split, so we can return
            elif self.dataset in ['EuroSAT']:
                print('EuroSAT has no predefined test split. Using a custom seeded random split.')
                load_helper = getattr(torchvision.datasets, self.dataset)
                full_set = load_helper(root=os.path.abspath(f'{self.data_path}'), download=True)
                
                all_labels = extract_labels(full_set)
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=0.2,
                random_state=0) #always with 0 seed - testset split should always be the same

                base_trainset = Subset(full_set, train_indices)

            elif self.dataset == 'WaferMap':
                print('WaferMap has no predefined test split. Using a custom seeded random split.')
                data=np.load(os.path.join(f'{self.data_path}/MixedWM38.npz'))
                x = data["arr_0"]
                y = data["arr_1"]
                
                x_train, _, y_train, _ = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.2,
                random_state=0)

                base_trainset = NumpyDataset(x_train, y_train)
            
            elif self.dataset in ['KITTI_Distance_Multiclass', 'Describable-Textures', 'Flickr-Material']:
                print(f'{self.dataset} has no predefined test split. Using a custom seeded random split.')
                full_set = HDF5ImageDataset(f'{self.data_path}/{self.dataset}/{self.dataset}.h5')
                
                all_labels = extract_labels(full_set)

                testsize = 0.5 if self.dataset in ['Flickr-Material'] else 0.2
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=testsize,
                random_state=0) #always with 0 seed - testset split should always be the same

                base_trainset = Subset(full_set, train_indices)

            else:
                print('Dataset not loadable')  

            all_labels = extract_labels(base_trainset)

            #use 20% of training set as validation set, but at most 10000 images
            testsplit = 0.2 if len(base_trainset) <= 50000 else 10000

            train_indices, val_indices, _, _ = train_test_split(
                range(len(base_trainset)),
                all_labels,
                stratify=all_labels,
                test_size=testsplit,
                random_state=self.run)  # same validation split for same runs, but new validation on multiple runs
            
            if test_only == False:
                self.base_trainset = Subset(base_trainset, train_indices)

            self.testset = SubsetWithTransform(Subset(base_trainset, val_indices), self.transforms_preprocess_test)
            
            self.num_classes = extract_num_classes(self.testset, labels=all_labels)
    
    def load_style_dataloader(self, style_dir, batch_size):
        style_dataset = StyleDataset(style_dir, dataset_type=self.dataset)
        style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=False)
        return style_loader
        
    def load_augmented_traindata(self, target_size, generated_ratio, epoch=0, robust_samples=0, stylization_first=False):
        self.robust_samples = robust_samples
        self.target_size = target_size
        if generated_ratio > 0.0:
            if not hasattr(self, "generated_dataset"):
                try:
                    if self.dataset == "ImageNet-100":
                        self.generated_dataset = HDF5ImageDataset(f'{self.data_path}/{self.dataset}-add-1m-dm.h5')
                        #self.generated_dataset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-synthetic'),
                        #                                                loader=custom_transforms.safe_kornia_loader)
                        self.generated_dataset_length = len(self.generated_dataset)
                    elif self.dataset in ["CIFAR100", "CIFAR10", 'TinyImageNet']:
                        self.generated_dataset = np.load(os.path.abspath(f'{self.data_path}/{self.dataset}-add-1m-dm.npz'),
                                                mmap_mode='r') 
                        self.generated_dataset_length = len(self.generated_dataset['label'])
                    else: 
                        # Base GAN paths
                        gan_balanced_root = Path(self.data_path) / f"{self.dataset}_GAN_balanced"
                        gan_root = Path(self.data_path) / f"{self.dataset}_GAN"
                        valid_balanced_root = Path(self.data_path) / f"{self.dataset}_GAN_valid_balanced"
                        valid_root = Path(self.data_path) / f"{self.dataset}_GAN_valid"

                        # Try to find GAN data that left out the valid split for training to not overfit here
                        if self.validontest == False and valid_balanced_root.exists():
                            # prefer balanced GAN if it exists, else fallback to normal GAN
                            root = valid_balanced_root
                        elif self.validontest == False and valid_root.exists():
                            root = valid_root   
                        # Prefer balanced GAN if it exists, else fallback to normal GAN
                        elif gan_balanced_root.exists():
                            root = gan_balanced_root
                        else:
                            root = gan_root
 
                        self.generated_dataset = NumericFolderKorniaDataset(
                            root=root.resolve(),
                            multilabel=self.multilabel,
                        )
                        self.generated_dataset_length = len(self.generated_dataset)
                    self.generated_ratio = generated_ratio
                except:
                    print(f'No synthetic data found for this dataset in {self.data_path}')
                    self.generated_ratio = 0.0
                    self.generated_dataset = None
            else:
                self.generated_ratio = generated_ratio
        else:
            self.generated_ratio = 0.0
            self.generated_dataset = None

        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        torch.cuda.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        self.num_generated = int(target_size * generated_ratio)
        self.num_original = target_size - self.num_generated

        if stylization_first == True:

            if self.num_original > 0:
                original_indices = torch.randperm(self.target_size)[:self.num_original]
                original_subset = SubsetWithTransform(Subset(self.base_trainset, original_indices), self.transforms_preprocess_train)

                if self.stylization_orig is not None:
                    stylized_original_subset, style_mask_orig = self.stylization_orig(original_subset)
                else: 
                    stylized_original_subset, style_mask_orig = original_subset, [False] * len(original_subset)
            else:
                stylized_original_subset, style_mask_orig = None, []
            
            if self.num_generated > 0 and self.generated_dataset is not None:
                generated_indices = np.random.choice(self.generated_dataset_length, size=self.num_generated, replace=False)

                if self.dataset in ["CIFAR100", "CIFAR10", 'TinyImageNet']:
                    generated_subset = NumpyDataset(
                        self.generated_dataset['image'][generated_indices],
                        self.generated_dataset['label'][generated_indices],
                        transform=self.transforms_preprocess_train
                    )
                else:
                    generated_subset = SubsetWithTransform(Subset(self.generated_dataset, generated_indices), 
                                                           self.transforms_preprocess_train)

                if self.stylization_gen is not None:
                    stylized_generated_subset, style_mask_gen = self.stylization_gen(generated_subset)
                else:
                    stylized_generated_subset, style_mask_gen = generated_subset, [False] * len(generated_subset)
            else:
                stylized_generated_subset, style_mask_gen = None, []
            
            style_mask = style_mask_orig + style_mask_gen
            
            self.trainset = AugmentedDataset(stylized_original_subset, stylized_generated_subset, style_mask,
                                            self.transforms_basic, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                            self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples)

        else:
            if self.num_original > 0:
                original_indices = torch.randperm(len(self.base_trainset))[:self.num_original]
                original_subset = SubsetWithTransform(Subset(self.base_trainset, original_indices), self.transforms_preprocess_train)

            else:
                original_subset = None
            
            if self.num_generated > 0 and self.generated_dataset is not None:
                generated_indices = np.random.choice(self.generated_dataset_length, size=self.num_generated, replace=False)

                if self.dataset in ["CIFAR100", "CIFAR10", 'TinyImageNet']:
                    generated_subset = NumpyDataset(
                        self.generated_dataset['image'][generated_indices],
                        self.generated_dataset['label'][generated_indices],
                        transform=self.transforms_preprocess_train
                    )
                else:
                    generated_subset = SubsetWithTransform(Subset(self.generated_dataset, generated_indices), self.transforms_preprocess_train)

            else:
                generated_subset = None

            self.during_train_transform = custom_transforms.DuringTrainingTransforms(generated_ratio, robust_samples, 
                                                                                     self.stylization_orig, 
                                                                                     self.stylization_gen, 
                                                                                     self.transforms_orig_after_style,
                                                                                     self.transforms_gen_after_style)
            self.during_train_label_transform = custom_transforms.OneHotAdaptiveConfidenceLabelTransform(self.label_smoothing, 
                                                                                                         self.num_classes,
                                                                                                         self.multilabel)
            
            self.trainset = BasicAugmentedDataset(original_subset, generated_subset, self.transforms_basic, self.robust_samples)
    
    def precompute_and_append_c_data(self, set, c_datasets, corruption, csv_handler, subset, subsetsize,
                valid_run, load_root='../data'):

        random_corrupted_testset = None
        loaded_from_repo = False

        # --- Try to load from disk if load_root was explicitly provided and the testset is used---
        if load_root is not None:
            if self.validontest == True:
                base_paths = [
                    os.path.join(load_root, f"{self.dataset}-c", corruption),
                    os.path.join(load_root, f"{self.dataset}-c-bar", corruption)
                ]
            else:
                base_paths = [
                    os.path.join(load_root, f"{self.dataset}-c-valid_{self.run}", corruption),
                    os.path.join(load_root, f"{self.dataset}-c-bar-valid_{self.run}", corruption)
                ]

            for repo_path in base_paths:
                if os.path.isdir(repo_path) and any(os.scandir(repo_path)):
                    random_corrupted_testset = NumericFolderKorniaDataset(
                        root=repo_path,
                        multilabel=self.multilabel,
                        #transform=self.transforms_preprocess_test,
                    )
                    loaded_from_repo = True
                    break

            if not loaded_from_repo:
                print(
                    f"Tried using precomputed corruptions, but none found at either:\n"
                    f"  {base_paths[0]}\n  {base_paths[1]}\n"
                    "Falling back to on-the-fly corruption."
                )

        # --- Fallback: use on-the-fly corruption ---
        if random_corrupted_testset is None:
            random_corrupted_testset = SubsetWithTransform(
                self.testset,
                transform=custom_transforms.RandomCommonCorruptionTransform(
                    set, corruption, self.dataset, csv_handler, self.resize
                )
            )

        # --- Apply subset if requested ---
        if subset == True and subsetsize < len(random_corrupted_testset):
            selected_indices = np.random.choice(
                len(random_corrupted_testset), subsetsize, replace=False
            )
            random_corrupted_testset = Subset(random_corrupted_testset, selected_indices)

        # --- Precompute only if NOT loaded from repo ---
        if valid_run and not loaded_from_repo:
            
            
            batch_size = min(100, subsetsize)

            workers = (
                self.number_workers
                if corruption in ['caustic_refraction', 'perlin_noise',
                                'plasma_noise', 'sparkles']
                and self.dataset not in ['CIFAR10', 'CIFAR100', 'TinyImageNet',
                                        'EuroSAT', 'WaferMap', 'GTSRB', 'PCAM', 'DermaMNIST']
                else 0
            )

            r = torch.Generator()
            r.manual_seed(0)

            precompute_loader = DataLoader(
                random_corrupted_testset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=workers,
                worker_init_fn=seed_worker,
                generator=r,
                drop_last=False
            )

            all_samples, all_labels = [], []

            for batch_samples, batch_labels in precompute_loader:
                all_samples.append(batch_samples)
                all_labels.append(batch_labels)

            all_samples_tensor = torch.cat(all_samples, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)

            random_corrupted_testset = TensorDataset(
                all_samples_tensor, all_labels_tensor
            )

        c_datasets.append(random_corrupted_testset)
        return c_datasets
    
    def load_data_a(self):
        dataset_a = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-a/'),
                                                                                transform=self.transforms_preprocess_test,
                                                                        loader=custom_transforms.safe_kornia_loader)
        original_class_to_idx = get_class_to_idx_recursive(self.testset)
        dataset_a.class_to_idx = original_class_to_idx
        # ImageFolder stores samples as (path, class_index). 
        # We must re-map the indices based on the folder names.
        new_samples = []
        for path, _ in dataset_a.samples:
            folder_name = path.split('/')[-2] # Extract folder name from path
            correct_idx = original_class_to_idx[folder_name]
            new_samples.append((path, correct_idx))

        dataset_a.samples = new_samples
        dataset_a.targets = [s[1] for s in new_samples]

        return dataset_a

    def load_data_r(self):
        dataset_r = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-r/'),
                                                                                transform=self.transforms_preprocess_test,
                                                                        loader=custom_transforms.safe_kornia_loader)
        original_class_to_idx = get_class_to_idx_recursive(self.testset)
        dataset_r.class_to_idx = original_class_to_idx
        # ImageFolder stores samples as (path, class_index). 
        # We must re-map the indices based on the folder names.
        new_samples = []
        for path, _ in dataset_r.samples:
            folder_name = path.split('/')[-2] # Extract folder name from path
            correct_idx = original_class_to_idx[folder_name]
            new_samples.append((path, correct_idx))

        dataset_r.samples = new_samples
        dataset_r.targets = [s[1] for s in new_samples]

        return dataset_r

    def load_data_es(self):
        dataset_es_auto = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-es/test/auto_exposure/'),
                                                                                transform=self.transforms_preprocess_test,
                                                                        loader=custom_transforms.safe_kornia_loader)
        dataset_es_param = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-es/test/param_control/'),
                                                                                transform=self.transforms_preprocess_test,
                                                                        loader=custom_transforms.safe_kornia_loader)
        
        return dataset_es_auto, dataset_es_param
    
    def load_data_sketch(self):
        dataset_sketch = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-sketch/'),
                                                                                transform=self.transforms_preprocess_test,
                                                                        loader=custom_transforms.safe_kornia_loader)
        return dataset_sketch
    
    def load_data_v2(self):

        dataset_v2 = torchvision.datasets.ImageFolder(
            root=os.path.abspath(f'{self.data_path}/{self.dataset}-v2-matched-frequency/'),
            transform=self.transforms_preprocess_test,
            loader=custom_transforms.safe_kornia_loader
        )
        return dataset_v2
                    
    def load_data_c(self, subset, subsetsize, valid_run, load_root='../data'):

        c_datasets = []
        #c-corruption benchmark: https://github.com/hendrycks/robustness
        corruptions_c = np.asarray(np.loadtxt(os.path.join(self.c_labels_path, "c-labels.txt"), dtype=list))
        
        np.random.seed(0) # to make subsamples reproducible
        torch.manual_seed(0)
        random.seed(0)
        global fixed_worker_rng #impulse noise augmentations sk-learn function needs a separate rng for reproducibility
        fixed_worker_rng = np.random.default_rng()

        if not self.validontest:
            print('Using custom c- and c-bar corruptions on validation data.')
        if load_root is not None:
            print(f'Trying to load precomputed corruptions from {load_root} if available.')

        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            
            csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/cifar_c_bar.csv'))
            corruptions_bar = csv_handler.read_corruptions()

            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:

                if self.validontest:
                    subtestset = self.testset
                    np_data_c = np.load(os.path.abspath(f'{self.data_path}/{self.dataset}-{set}/{corruption}.npy'), mmap_mode='r')
                    np_data_c = np.array(np.array_split(np_data_c, 5))

                    if subset == True and subsetsize < len(self.testset):
                        selected_indices = np.random.choice(len(self.testset), subsetsize, replace=False)
                        subtestset = Subset(self.testset, selected_indices)
                        np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]
                    concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, self.resize, self.transforms_preprocess_test) for intensity_data_c in np_data_c])
                    c_datasets.append(concat_intensities)

                else:
                    c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run, load_root)
                    
        elif self.dataset in ['ImageNet', 'TinyImageNet', 'ImageNet-100']:

            csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/imagenet_c_bar.csv'))
            corruptions_bar = np.asarray(csv_handler.read_corruptions())
            
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:
                
                if self.validontest:
                    intensity_datasets = [torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-{set}/' + corruption + '/' + str(intensity)),
                                                                        transform=self.transforms_preprocess_test,
                                                                loader=custom_transforms.safe_kornia_loader) for intensity in range(1, 6)]
                    if subset == True and subsetsize < len(self.testset):
                        selected_indices = np.random.choice(len(intensity_datasets[0]), subsetsize, replace=False)
                        intensity_datasets = [Subset(intensity_dataset, selected_indices) for intensity_dataset in intensity_datasets]
                    
                    concat_intensities = ConcatDataset(intensity_datasets)
                    c_datasets.append(concat_intensities)

                else:
                    c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run, load_root)

        else:
            if self.validontest and load_root == None:
                print(f'No c- and c-bar-benchmark available for {self.dataset}. Using custom corruptions.')

            if self.dataset in ['GTSRB', 'Wafermap']:
                csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/cifar_c_bar.csv'))
                corruptions_bar = csv_handler.read_corruptions()
            elif self.dataset in ['KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
                corruptions_bar = []
                csv_handler = None
            else:
                csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/imagenet_c_bar.csv'))
                corruptions_bar = csv_handler.read_corruptions()
            
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:
                c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, 
                                                               subset, subsetsize, valid_run, load_root)

        if valid_run == True:
            c_datasets = ConcatDataset(c_datasets)
            self.c_datasets_dict = {'combined': c_datasets}
        else:
            self.c_datasets_dict = {label: dataset for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)}

        return self.c_datasets_dict

    def get_loader(self, batchsize):

        self.batchsize = batchsize

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        if self.generated_ratio > 0.0:
            self.CustomSampler = BalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                batch_size=batchsize)
        else:
            self.CustomSampler = BatchSampler(RandomSampler(self.trainset), batch_size=batchsize, drop_last=False)            

        self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                    num_workers=self.number_workers, worker_init_fn=seed_worker, 
                                    generator=g, persistent_workers=True)
        
        val_workers = self.number_workers if self.dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 
                       'Describable-Textures', 'SynthiCAD', 'KITTI_RoadLane', 'KITTI_Distance_Multiclass'] else 0
        self.testloader = DataLoader(self.testset, batch_size=batchsize, pin_memory=True, num_workers=val_workers)

        return self.trainloader, self.testloader
    

    def update_set(self, epoch, start_epoch, stylization_first=False, config=None):
        
        if config:
            self.update_transforms(stylize_prob_orig=config.get("stylize_prob_real", None), 
                            stylize_prob_syn=config.get("stylize_prob_synth", None), 
                            alpha_min_orig=config.get("alpha_min_real", None), 
                            alpha_min_syn=config.get("alpha_min_synth", None),
                            style_and_aug_orig=config.get("style_and_aug_orig", None), 
                            style_and_aug_syn=config.get("style_and_aug_synth", None), 
                            RandomEraseProbability=config.get('RandomEraseProbability', None))
        
            self.generated_ratio = config.get(["synth_ratio"], self.generated_ratio)

        if ((self.generated_ratio != 0.0 or self.stylization_gen is not None or self.stylization_orig is not None) and epoch != 0 and epoch != start_epoch) or config is not None:
            # This should be updated when config gives new transforms parameters, when there is generated data or when there is stylization
            del self.trainset

            self.load_augmented_traindata(self.target_size, generated_ratio=self.generated_ratio, epoch=epoch, robust_samples=self.robust_samples, stylization_first=stylization_first)

        del self.trainloader
        gc.collect()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True, 
                                      num_workers=self.number_workers, worker_init_fn=seed_worker,
                                      generator=g, persistent_workers=True)
        
        return self.trainloader