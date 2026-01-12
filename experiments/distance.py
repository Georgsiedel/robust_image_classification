"""
This script is based on the founding paper repo here: 
https://github.com/yangarbiter/robust-local-lipschitz
and the adaptation of the method here:
https://github.com/Georgsiedel/minimal-separation-corruption-robustness
"""
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
module_path = os.path.abspath(os.path.dirname(__file__))

if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import random
from pathlib import Path

from data import DataLoading
from custom_datasets import SubsetWithTransform
import custom_transforms

def plot_distances(traintrain, traintest, testtest, traintrain_ret, traintest_ret, testtest_ret, n_bins):

    y1 = np.arange(len(traintrain_ret)) / len(traintrain_ret)
    y2 = np.arange(len(traintest_ret)) / len(traintest_ret)
    y3 = np.arange(len(testtest_ret)) / len(testtest_ret)

    fig = plt.figure(figsize=[5,3.5], dpi=300)
    fig, axs = plt.subplots(3, 2, sharex='col', tight_layout=True)
    axs[0, 0].hist(traintrain_ret, bins=n_bins)
    axs[1, 0].hist(traintest_ret, bins=n_bins)
    axs[2, 0].hist(testtest_ret, bins=n_bins)
    axs[0, 1].plot(traintrain, y1)
    axs[1, 1].plot(traintest, y2)
    axs[2, 1].plot(testtest, y3)
    plt.xlim(0, 1)
    axs[0,0].set_title("Train-Train Separation Distribution")
    axs[1,0].set_title("Train-Test Separation Distribution")
    axs[2,0].set_title("Test-Test Separation Distribution")
    axs[0,1].set_title("Train-Train Separation CDF")
    axs[1,1].set_title("Train-Test Separation CDF")
    axs[2,1].set_title("Test-Test Separation CDF")
    #fig.savefig(r"results/r-distance-distribution.svg", dpi=300)

    fig2 = plt.figure(figsize=[5,3.5], dpi=300)
    fig2, axs2 = plt.subplots(1, 1, sharex='col', tight_layout=True)
    axs2.hist(traintrain_ret, bins=n_bins)
    #axs2[1].plot(traintrain, y1)
    plt.xlim(0.2, 0.7)
    plt.xlabel("Distance (Linf)")
    plt.ylabel("Frequency of points")
    axs2.set_title("Train-Train Separation Distribution")
    #axs2[1].set_title("Train-Train Separation CDF")
    #fig2.savefig(r"results/r-distance-distribution2.svg", dpi=300)

def int_or_inf(x):
    if x.lower() in ("inf", "np.inf", "infinity"):
        return np.inf
    return int(x)

def _compute_min_dists(query_loader, db_loader, device, norm):
    """
    Helper function to compute minimum opposite-class distances.
    
    Iterates through query_loader. For each query_batch, it iterates
    through the entire db_loader, computes distances, masks same-class
    neighbors, and finds the minimum.
    """
    all_min_dists = []
    
    print(f"Starting distance calculation...")
    # Outer loop: one batch of query points at a time
    for query_x, query_y in tqdm(query_loader):
        
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        # Ensure data is flattened
        if len(query_x.shape) > 2:
            query_x = query_x.reshape(query_x.shape[0], -1)
            
        # Store min distances for this query batch
        batch_min_dists = torch.full((query_x.shape[0],), float('inf'), device=device)

        # Inner loop: compare against all batches in the database
        for db_x, db_y in db_loader:
            
            db_x = db_x.to(device)
            db_y = db_y.to(device)
            
            # Ensure data is flattened
            if len(db_x.shape) > 2:
                db_x = db_x.reshape(db_x.shape[0], -1)
            
            # Calculate pairwise distances between query and db batch
            # Shape: (query_batch_size, db_batch_size)

            dists = torch.cdist(query_x, db_x, p=norm)
            
            # Create a mask to find *different* classes
            # Shape: (query_batch_size, db_batch_size) for single-label, 
            # Shape: (query_batch_size, db_batch_size, n_labels) for multi-label
            broadcasted_inequality = (query_y.unsqueeze(1) != db_y.unsqueeze(0))
            
            if query_y.ndim == 1:
                # single-label case. broadcasted_inequality is already [N, M]
                mask = broadcasted_inequality
            else:
                # multi-label case. We want [N, M], where True means *any* of the C elements were different.
                mask = broadcasted_inequality.any(dim=-1)
            
            # Invalidate distances to same-class points by setting them to infinity
            dists[~mask] = float('inf')
            
            # Find the minimum distance in this db_batch for each query point
            min_dists_in_batch, _ = torch.min(dists, dim=1)
            
            # Update the overall minimum for the query batch
            batch_min_dists = torch.minimum(batch_min_dists, min_dists_in_batch)
        
        # We have the final minimum distances for this query batch
        all_min_dists.append(batch_min_dists.cpu())

    # Concatenate results from all query batches
    return torch.cat(all_min_dists).numpy()


# calculate r-separation distance of dataset
def get_nearest_oppo_dist(norm, dataset, batch_size):

    # Handle 'inf' from argparse
    if norm == np.inf:
        p_norm = float('inf')
    else:
        p_norm = norm
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_class = DataLoading(dataset=dataset)
    data_class.create_transforms(train_aug_strat_orig='None', train_aug_strat_gen='None')
    
    #Adjust training image preprocessing. 
    #Normally uses lower-size training images (FixRes recipe) and random resized crop that are different to test transforms.
    #This could induce bias, hence we now use the test images preprocessing on train image as well.
    #Distance is measured with those transforms for later evaluation on test images.
    if dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 
                       'Describable-Textures', 'Flickr-Material', 'SynthiCAD']:
        data_class.transforms_preprocess_train = transforms.Compose([transforms.Resize(232, antialias=True), 
                                                                     transforms.CenterCrop(224)])
    elif dataset in ['KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
        data_class.transforms_preprocess_train = transforms.Resize((384,1280), antialias=True)
    elif dataset in ['NEU-surface-defect']:
        data_class.transforms_preprocess_train = transforms.Resize(128, antialias=True)
    elif dataset in ['WaferMap']:
        #getting rid of any random padding
        data_class.transforms_preprocess_train = transforms.Compose([
                    transforms.ToImage(), 
                    transforms.ToDtype(torch.float32), #no scaling
                    custom_transforms.DivideBy3(),
                    custom_transforms.ExpandGrayscaleTensorTo3Channels(),
                    transforms.Pad(6) #our test time transform just pads 6px. 
                ])
    elif dataset in ['PCAM']:
        data_class.transforms_preprocess_train = transforms.CenterCrop(64)

    data_class.load_base_data()
    trainset = SubsetWithTransform(data_class.base_trainset, data_class.transforms_preprocess_train)
    testset = data_class.testset

    print(f"Loading {dataset}.")
    # Load data in batches, not all at once
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    trainloader_inner = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=16)

    # --- NEW CALCULATION ---
    print(f"Starting Train-Train evaluation.")
    start_time = time.perf_counter()
    # Query: train, Database: train
    traintrain_ret = _compute_min_dists(trainloader, trainloader_inner, device, p_norm)
    print(f"Train-Train done in {time.perf_counter() - start_time:.2f}s")

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Starting Train-Test evaluation.")
    start_time = time.perf_counter()
    # Query: test, Database: train
    traintest_ret = _compute_min_dists(testloader, trainloader_inner, device, p_norm)
    print(f"Train-Test done in {time.perf_counter() - start_time:.2f}s")

    testloader_inner = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    
    print(f"Starting Test-Test evaluation.")
    start_time = time.perf_counter()
    # Query: test, Database: test
    testtest_ret = _compute_min_dists(testloader, testloader_inner, device, p_norm)
    print(f"Test-Test done in {time.perf_counter() - start_time:.2f}s")

    return traintrain_ret, traintest_ret, testtest_ret

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_distance", type=int_or_inf, default='inf', help="Integer or 'inf' for the norm to be measured")
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to calculate distance for')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size for GPU computation')

    args = parser.parse_args()

    for dataset in ['NEU-surface-defect', 'PCAM']:
        
        for norm_distance in [np.inf, 1, 2]:
            #random seeding for reproducibility (this is only important should random preprocessing be used - 
            #here we use only deterministic test image preprocessing all around, so seed should not make a difference.)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            print(f"Calculating {norm_distance}-norm class-separation for dataset {dataset} with batch size {args.batch_size}.")

            traintrain_ret, traintest_ret, testtest_ret = get_nearest_oppo_dist(norm_distance, dataset, args.batch_size)

            traintrain = np.sort(traintrain_ret[traintrain_ret != 0])
            traintest = np.sort(traintest_ret[traintest_ret != 0])
            testtest = np.sort(testtest_ret[testtest_ret != 0])
            first_distances_to_save = 200
                
            summary = np.array([
                [len(traintrain), len(traintest), len(testtest)],
                [traintrain.mean(), traintest.mean(), testtest.mean()],
                [traintrain.min(), traintest.min(), testtest.min()]
            ])

            # Stack indexed rows from 0–99
            # shape -> (100, 3)
            indexed = np.stack([
                traintrain[1:first_distances_to_save],
                traintest[1:first_distances_to_save],
                testtest[1:first_distances_to_save]
            ], axis=1)

            # Concatenate summary rows on top
            ret = np.vstack([summary, indexed])

            index_labels = ['non-zero images', 'Mean Distance', 'Minimal Distance'] + [
                f'{i+1}th' for i in range(1, first_distances_to_save)
            ]
            
            # build DataFrame
            df_ret = pd.DataFrame(
                ret,
                columns=['Train-Train', 'Train-Test', 'Test-Test'],
                index=index_labels
            )

            epsilon_min = ret[2, :].min()/2
            print("Epsilon: ", epsilon_min)
            Path(f"./results/{dataset}/").mkdir(parents=True, exist_ok=True)

            df_ret.to_csv(f"./results/{dataset}/class_separation_distance_{norm_distance}_norm.csv", index=True, header=True, sep=';', float_format='%1.6f', decimal=',')