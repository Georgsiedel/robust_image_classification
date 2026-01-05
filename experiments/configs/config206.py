import numpy as np

train_corruptions = np.array([
{'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'beta2-5'},
#{'noise_type': 'uniform-linf', 'epsilon': 0.6, 'sphere': False, 'distribution': 'uniform'},
#{'noise_type': 'uniform-l0.5', 'epsilon': 24000000.0, 'sphere': False, 'distribution': 'uniform'},
#{'noise_type': 'uniform-l1', 'epsilon': 3000.0, 'sphere': False, 'distribution': 'uniform'},
#{'noise_type': 'uniform-l2', 'epsilon': 40.0, 'sphere': False, 'distribution': 'uniform'},
#{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.6, 'sphere': False, 'distribution': 'uniform'},
])
noise_sparsity = 1.0
noise_patch_scale = {'lower': 0.2, 'upper': 0.7}
combine_train_corruptions = True #augment the train dataset with all corruptions
concurrent_combinations = 1 #only has an effect if combine_train_corruption is True

batchsize = 128
minibatchsize = 8
#ImageNet ImageNet-100 TreeSAT Describable-Textures Flickr-Material KITTI_RoadLane KITTI_Distance_Multiclass 
# CIFAR100 CIFAR10 TinyImageNet PCAM GTSRB WaferMap EuroSAT SynthiCAD
dataset = 'WaferMap' 
generated_ratio = 0.1
normalize = True
validontest = False
validonc = True
validonadv = False
lrschedule = 'CosineAnnealingLR'
learningrate = 0.1
epochs = 195
lrparams = {'T_max': 195}
warmupepochs = 5
earlystop = False
earlystopPatience = 15
optimizer = 'SGD'
optimizerparams = {'momentum': 0.9, 'weight_decay': 1e-4, 'nesterov': False}
number_workers = 2
modeltype = 'WideResNet_28_4'
modelparams = {'dropout_rate': 0.2}
resize = False
aggressive_soft_crop = False
style_orig = {'probability': 0.2, 'alpha_min': 1.0, 'alpha_max': 1.0}
style_gen = {'probability': 0.2, 'alpha_min': 0.1, 'alpha_max': 1.0}
train_aug_strat_orig = 'TrivialAugmentWide' #TrivialAugmentWide, RandAugment, AutoAugment, AugMix
train_aug_strat_gen = 'TrivialAugmentWide' #TrivialAugmentWide, RandAugment, AutoAugment, AugMix
style_and_aug_orig = True
style_and_aug_gen = True
loss = 'BCEWithLogitsLoss'
lossparams = {'label_smoothing': 0.1}
trades_loss = False
trades_lossparams = {'step_size': 0.003, 'epsilon': 0.031, 'perturb_steps': 10, 'beta': 5.0, 'distance': 'l_inf'}
robust_loss = False
robust_lossparams = {'num_splits': 3, 'alpha': 12} #jsd if 3 splits, KL divergence if 2 splits
mixup = {'alpha': 0.2, 'p': 0.0} #default alpha 0.2 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
cutmix = {'alpha': 1.0, 'p': 0.0} # default alpha 1.0 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
manifold = {'apply': False, 'noise_factor': 3}
n2n_deepaugment = False
RandomEraseProbability = 0.0
swa = {'apply': True, 'start_factor': 0.8, 'lr_factor': 0.2}

#define train and test corruptions:
#define noise type (first column): 'gaussian', 'uniform-l0-impulse', 'uniform-l0-salt-pepper', 'uniform-linf'. also: all positive numbers p>0 for uniform Lp possible: 'uniform-l1', 'uniform-l2', ...
#define intensity (second column): max.-distance of random perturbations for model training and evaluation (gaussian: std-dev; l0: proportion of pixels corrupted; lp: epsilon)
#define whether density_distribution=max (third column) is True (sample only maximum intensity values) or False (uniformly distributed up to maximum intensity)
test_corruptions = np.array([
{'noise_type': 'uniform-l1', 'epsilon': 130.9, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 2.611, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.1303, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 700000.0, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 125.0, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 2.0, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.06, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.04, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.01, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.01, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.02, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.05, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.1, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.2, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.5, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 500000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 1000000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 2000000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 5000000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 10000000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 20000000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 100.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 200.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 500.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 1000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 2000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 5000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 1.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 2.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 5.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 10.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 20.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 50.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.5, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 1.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 2.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.5, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 1.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.01, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.5, 'sphere': False, 'distribution': 'max'},
])

test_on_c = True
combine_test_corruptions = False #augment the test dataset with all corruptions
calculate_adv_distance = False
adv_distance_params = {'setsize': 500, 'iters_pgd': 500, 'eps_iter': [0.0003,0.005,0.2], 'iters_second_attack': 40, 'norm': ['inf', 2, 1],
                       "clever": True, "clever_batches": [5,10,50,500], "clever_samples": [5,20,100,1024]}
calculate_autoattack_robustness = False
autoattack_params = {'setsize': 1000, 'epsilon': 8/255, 'norm': 'Linf'}

if dataset == 'CIFAR10':
    num_classes = 10
    pixel_factor = 1
elif dataset == 'CIFAR100':
    num_classes = 100
    pixel_factor = 1
elif dataset == 'ImageNet':
    num_classes = 1000
elif dataset == 'TinyImageNet':
    num_classes = 200
    pixel_factor = 2