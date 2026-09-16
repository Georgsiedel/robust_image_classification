# Style Transfer Data Augmentation for Robust Image Classification

This repository accompanies the paper **“TBA”**. It allows to validate the positive effect of two style transfer methods for image classification training on various model architectures and general or domain-specific benchmark datasets.

---

## 🧭 Overview

- Training of corruption-robust image classifiers
- Integration of benchmark and domain-specific data
- Support for **stylization-based augmentation** with
  - The [original AdaIN](https://arxiv.org/abs/1703.06868) as implemented [here](https://arxiv.org/abs/2512.15675), extended with a blending method for arbitrary-resolution images
  - The lightweight, arbitrary-resolution [MicroAST method](https://arxiv.org/abs/2211.15313)
- Flexible experiment configuration with diverse augmentations and model architectures

---

## 📂 Repository Structure

- `run_exp.py` – main experiment launcher  
- `experiments/`
  - `train.py` – training script  
  - `eval.py` – evaluation script  
  - `configs/config_{ID}.py` – experiment configuration files  
- `experiments/models/` – model definitions  
- `paths.json` – configuration for dataset and checkpoint paths  
- `data/` – contains information for c and c-bar datasets

---

## ▶️ Running Experiments

`run_exp.py` runs one or multiple experiment IDs.

Each experiment setup must be defined in
`experiments/configs/config_{ID}.py`


Internally, the launcher calls:

- `experiments/train.py`
- `experiments/eval.py`

---

## 🛠 Path Configuration

Use `paths.json` to specify directories for:

- datasets
- pretrained or trained models
- external storage layouts (e.g., Kaggle, custom structures)

Default expectation:

project_root/

├── repository/

├── data/

└── trained_models/


> The `data/` folder inside this repository only contains information for c and c-bar datasets; full datasets should be placed in the external `data/` directory referenced in `paths.json`.

---

## 📚 Datasets

### Automatically downloaded
- CIFAR-10  
- CIFAR-100
- GTSRB
- EuroSAT
are placed automatically into `data/`.

### Must be added manually
- ImageNet  
- TinyImageNet
- Other domain-specific sets as can be found in `experiments/data.py`
- Corrupted variants (though they can be generated on the fly from given test data in `experiments/eval.py`):
  - `-c`
  - `-c-bar`

---

## 🧪 Synthetic Data Usage

To enable generated data, set `generate_ratio > 0.0` and 

### for CIFAR and TinyImageNet
place `.npz` files in `data/` with the naming pattern:
`{dataset}-add-1m-dm.npz`

that can be obtained from:
- https://github.com/wzekai99/DM-Improves-AT

or generated via:
- https://github.com/NVlabs/edm

### for ImageNet-100
generate latent diffusion images from the respective subrepository and place `.npz` files in `data/` with the naming pattern:
{self.dataset}-add-1m-dm.h5

### for other datasets
generate StyleGAN-3 images from the respective subrepository and place ImageFolder styly files in `data/` with the naming pattern:
"{self.dataset}_GAN"

---

## 🎨 Stylization Features

Stylization with AdaIN ('nst') requires encoded image features from **Painter-by-Numbers**.

Required file in `data/`:
`style_feats_adain_1000.npy`


For exact reproduction, download the 1000 features used here:

- https://zenodo.org/records/16279015

Stylization with MicroAST ('microast') draws style statistics randomly from a precomputed distribution, hence requiring a file "style_distribution.npz".
Use the script precompute_style_distribution.py to obtain the file, here using the train-1 split from [Painter-by-Numbers](https://www.kaggle.com/c/painter-by-numbers).

---

## 🧭 Model Architectures

Models are located in:
`experiments/models/`


Key characteristics:

- include parameter `factor`, which injects a stride factor in the first conv layer in order to adapt CIFAR (32×32) architectures for TinyImageNet (64×64)
- Large resolution models include pretrained weights and training from scratch
- all models inherit forward pass from `ct_model.py`, enabling:
  - normalization  
  - noise injection  
  - mixup
  - DeepAugment
  - deeper-layer augmentations  

---

## 📚 Evaluation

Multi-dimensional robustness evaluation options can be selected in the experiments config, including

- real-world c-corruptions (precomputed test benchmarks for CIFAR, TinyImageNet and ImageNet, computed on the fly for other datasets or validation splits)
- diverse c-bar corruptions that are dissimilar in their frequency spectrum
- diverse precomputed ImageNet benchmarks (A, R, ES, Sketch, v2)
- Accuracy on diverse p-norm noise, including class-separated and imperceptible noise as parametrized in `experiments/distance.py` and `experiments/noise.py`
- AutoAttack adversarial accuracy
- Adversarial Distance
- CLEVER score (estimated lower bound adversarial distance)
- confidence calibration error on clean and corrupted data

## ✅ Capabilities Summary

- corruption-robust training  
- integration of synthetic and stylized augmentation
- integration of various benchmark datasets  
- configurable experiment setups for diverse data augmentation

---


