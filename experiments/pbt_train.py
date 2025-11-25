import sys
import os
import re
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
module_path = os.path.abspath(os.path.dirname(__file__))

warnings.filterwarnings( # this supresses soft errors due to interaction of ray and multiprocessing
    "ignore",
    message="resource_tracker: process died unexpectedly",
    category=UserWarning,
    module="multiprocessing.resource_tracker",
)
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0" #supresses warning of excessive checkpointing
os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "120" #one global update every 120s
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

if module_path not in sys.path:
    sys.path.append(module_path)
import shutil
from tqdm import tqdm
import argparse
import importlib
import torch.multiprocessing as mp
import json
import numpy as np
import torch.amp
import tempfile
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import time
import random

import data
import custom_datasets
import utils
import losses
import models
from eval_corruptions import compute_c_corruptions
from eval_adversarial import fast_gradient_validation

import torch.backends.cudnn as cudnn
from run_pbt import device

if torch.cuda.is_available():
    cudnn.benchmark = False #this slightly speeds up 32bit precision training (5%). False helps achieve reproducibility
    cudnn.deterministic = True

# Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, PopulationBasedTrainingReplay

import logging
import time

# Configure logging levels to minimize output
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.tune.execution.trial_runner").setLevel(logging.ERROR)
logging.getLogger("ray.tune.execution.trial_executor").setLevel(logging.ERROR)
logging.getLogger("ray.tune.trainable").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='PyTorch Training with perturbations')
parser.add_argument('--resume', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='resuming from saved checkpoint in fixed-path repo defined below')
parser.add_argument('--mode', default='both', type=str, help='pass "tune", "replay" or "both"')
parser.add_argument('--pbt_params', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for population based tuning')
parser.add_argument('--pbt_hyperparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='hyperparameters to tune with population based tuning')
parser.add_argument('--train_corruptions', default={'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'max'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='dictionary for type of noise, epsilon value, '
                    'whether it is always the maximum noise value and a distribution from which various epsilon are sampled')
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=128, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=utils.str2bool, nargs='?', const=True, default=True, help='True: Use full '
                    'training data and test data. False: 80:20 train:valiation split, validation also used for testing.')
parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
parser.add_argument('--learningrate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrschedule', default='MultiStepLR', type=str, help='Learning rate scheduler from pytorch.')
parser.add_argument('--lrparams', default={'milestones': [85, 95], 'gamma': 0.2}, type=str, action=utils.str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the learning rate scheduler')
parser.add_argument('--earlystop', type=utils.str2bool, nargs='?', const=False, default=False, help='Use earlystopping after '
                    'some epochs (patience) of no increase in performance')
parser.add_argument('--earlystopPatience', default=15, type=int,
                    help='Number of epochs to wait for a better performance if earlystop is True')
parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer from torch.optim')
parser.add_argument('--optimizerparams', default={'momentum': 0.9, 'weight_decay': 5e-4}, type=str,
                    action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the optimizer')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--aggressive_soft_crop', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Use aggressive random crops combined with adaptive label smoothing (soft augmentation, see ' \
                    'https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Soft_Augmentation_for_Image_Classification_CVPR_2023_paper.pdf).')
parser.add_argument('--train_aug_strat_orig', default='TrivialAugmentWide', type=str, help='augmentation scheme')
parser.add_argument('--train_aug_strat_gen', default='TrivialAugmentWide', type=str, help='augmentation scheme')
parser.add_argument('--style_orig', default={'probability': 0.0, 'alpha_min': 1.0, 'alpha_max': 1.0}, type=str,
                    action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for stylization of generated '
                    'images. 0.0 probability means no stylization.')
parser.add_argument('--style_gen', default={'probability': 0.0, 'alpha_min': 1.0, 'alpha_max': 1.0}, type=str,
                    action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for stylization of generated '
                    'images. 0.0 probability means no stylization.')
parser.add_argument('--style_and_aug_orig', type=utils.str2bool, nargs='?', const=False, default=True,
                    help='Apply Stylization and main augmentation strategy sequentially to original images. If False, ' \
                    'main augmentation only on images that are not stylized due to stylize probability.')
parser.add_argument('--style_and_aug_gen', type=utils.str2bool, nargs='?', const=False, default=True,
                    help='Apply Stylization and main augmentation strategy sequentially to generated images. If False, ' \
                    'main augmentation only on images that are not stylized due to stylize probability.')
parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='loss function to use, chosen from torch.nn loss functions')
parser.add_argument('--lossparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the standard loss function')
parser.add_argument('--trades_loss', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='whether or not to use trades loss for training')
parser.add_argument('--trades_lossparams',
                    default={'step_size': 0.003, 'epsilon': 0.031, 'perturb_steps': 10, 'beta': 1.0, 'distance': 'l_inf'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--robust_loss', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='whether or not to use robust (JSD/stability) loss for training')
parser.add_argument('--robust_lossparams', default={'num_splits': 3, 'alpha': 12}, type=str, action=utils.str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the robust loss function. If 3, JSD will be used.')
parser.add_argument('--mixup', default={'alpha': 0.2, 'p': 0.0}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Mixup parameters, Pytorch suggests 0.2 for alpha. Mixup, Cutmix are randomly '
                    'chosen without overlapping based on their probability, even if the sum of the probabilities is >1')
parser.add_argument('--cutmix', default={'alpha': 1.0, 'p': 0.0}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Cutmix parameters, Pytorch suggests 1.0 for alpha. Mixup, Cutmix are randomly '
                    'chosen without overlapping based on their probability, even if the sum of the probabilities is >1')
parser.add_argument('--manifold', default={'apply': False, 'noise_factor': 4}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='Choose whether to apply noisy mixup in manifold layers')
parser.add_argument('--concurrent_combinations', default=1, type=int, help='How many of the training noise values should '
                    'be applied at once on one image. USe only if you defined multiple training noise values.')
parser.add_argument('--number_workers', default=2, type=int, help='How many workers are launched to parallelize data '
                    'loading. Experimental. 4 for ImageNet, 1 for Cifar. More demand GPU memory, but maximize GPU usage.')
parser.add_argument('--RandomEraseProbability', default=0.0, type=float,
                    help='probability of applying random erasing to an image')
parser.add_argument('--warmupepochs', default=5, type=int,
                    help='Number of Warmupepochs for stable training early on. Start with factor 10 lower learning rate')
parser.add_argument('--normalize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--minibatchsize', default=8, type=int, help='batchsize, for which a new corruption type is sampled. '
                    'batchsize must be a multiple of minibatchsize. in case of p-norm corruptions with 0<p<inf, the same '
                    'corruption is applied for all images in the minibatch')
parser.add_argument('--validonc', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to do a validation on a subset of c-data every epoch')
parser.add_argument('--validonadv', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to do a validation with an FGSM adversarial attack every epoch')
parser.add_argument('--swa', default={'apply': True, 'start_factor': 0.85, 'lr_factor': 0.2}, type=str,
                    action=utils.str2dictAction, metavar='KEY=VALUE', help='start_factor defines when to start weight '
                    'averaging compared to overall epochs. lr_factor defines which learning rate to use in the averaged area.')
parser.add_argument('--noise_sparsity', default=0.0, type=float,
                    help='probability of not applying a calculated noise value to a dimension of an image')
parser.add_argument('--noise_patch_scale', default={'lower': 0.3, 'upper': 1.0}, type=str, action=utils.str2dictAction,
                    metavar='KEY=VALUE', help='bounds of the scale to choose the area ratio of the image from, which '
                    'gets perturbed by random noise')
parser.add_argument('--generated_ratio', default=0.0, type=float, help='ratio of synthetically generated images mixed '
                    'into every training batch')
parser.add_argument('--n2n_deepaugment', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to apply DeepAugment according to https://github.com/hendrycks/imagenet-r')
parser.add_argument('--stylization_first', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='True: Stylization and caching of the next epoch of images upon trainset loading. ' \
                    'False: Stylization of all images to be stylized batchwise in the training loop. False is faster,' \
                    'as long as train_aug_strat is not too slow (which is heavily influenced by minibatchsize).' \
                    'True is infeasible for large datasets as stylized subset needs to be fit into VRAM')
parser.add_argument(
    "--int_adain_params",
    default={},
    type=str,
    action=utils.str2dictAction,
    metavar="KEY=VALUE",
    help="parameters for the chosen model",
)
parser.add_argument(
    "--kaggle",
    type=utils.str2bool,
    nargs="?",
    const=False,
    default=False,
    help="Whether to run on Kaggle or locally.",
)

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
train_corruptions = config.train_corruptions


def train_epoch(model, trainloader, optimizer, criterion, Scaler, Dataloader, style_dataloader=None, pbar=None,
                noise_config=None, manifold_noise_config=None):
    
    if noise_config == False:
        noise = None
    else:
        noise = train_corruptions

    if manifold_noise_config == False:
        manifold_noise = False
    elif manifold_noise_config == True:
        manifold_noise = True
    else: 
        manifold_noise = args.manifold['apply']

    model.train()
    correct, total, train_loss, avg_train_loss = 0, 0, 0, 0

    if style_dataloader:
        style_iter = iter(style_dataloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        optimizer.zero_grad()
        if criterion.robust_samples >= 1:
            inputs = torch.cat(inputs, 0)
        
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
        if style_dataloader:
            try:
                style_feats = next(style_iter)
                style_feats = style_feats.to(device, dtype=torch.float32)
            except StopIteration:
                style_iter = iter(style_dataloader)
                style_feats = next(style_iter)
                style_feats = style_feats.to(device, dtype=torch.float32)

            content_batch_size = inputs.size(0)
            style_batch_size = style_feats.size(0)

            if content_batch_size != style_batch_size:
                style_feats = style_feats[:content_batch_size]

        else:
            style_feats = None
        with torch.amp.autocast(device_type=device):
            outputs, mixed_targets = model(inputs, targets, criterion.robust_samples, noise, args.mixup['alpha'],
                                           args.mixup['p'], manifold_noise, args.manifold['noise_factor'],
                                           args.cutmix['alpha'], args.cutmix['p'], args.minibatchsize,
                                           args.concurrent_combinations, args.noise_sparsity, args.noise_patch_scale['lower'],
                                           args.noise_patch_scale['upper'], Dataloader.generated_ratio, args.n2n_deepaugment, 
                                           style_feats=style_feats, **args.int_adain_params)
            if args.trades_loss:
                with torch.amp.autocast(device_type=device, enabled=False): # recommended for numerical stability
                    loss = losses.trades_loss(model,
                    inputs,
                    targets,
                    optimizer,
                    **args.trades_lossparams)
            else:    
                criterion.update(model, optimizer)
                loss = criterion(outputs, mixed_targets, inputs, targets)

        Scaler.scale(loss).backward()
        Scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        Scaler.step(optimizer)
        Scaler.update()
        if device == 'cuda':
            torch.cuda.synchronize()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        
        if args.dataset in ['WaferMap', 'KITTI_Distance_Multiclass']:
            predicted = (outputs > 0.5).float()    
            mixed_targets = (mixed_targets > 0.5).float()   
        elif np.ndim(mixed_targets) == 2:    
            _, mixed_targets = mixed_targets.max(1)

        if criterion.robust_samples >= 1:
            mixed_targets = torch.cat([mixed_targets] * (criterion.robust_samples+1), 0)

        total += mixed_targets.size(0)
        if args.dataset in ['WaferMap', 'KITTI_Distance_Multiclass']:
                matches = predicted.eq(targets)  # shape: [batch_size, num_labels]
                exact_match = matches.all(dim=1)  # shape: [batch_size], bool tensor
                correct += exact_match.sum().item()
        else:
            correct += predicted.eq(mixed_targets).sum().item()
        avg_train_loss = train_loss / (batch_idx + 1)
        if pbar:
            pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(
                avg_train_loss, 100. * correct / total, correct, total))
            pbar.update(1)

    train_acc = 100. * correct / total
    return train_acc, avg_train_loss, model, optimizer, Scaler, Dataloader, style_dataloader

def valid_epoch(net, validationloader, Traintracker, Dataloader, criterion, testsets_c=None, pbar=None):
    net.eval()
    with torch.no_grad():
        test_loss, correct, total, avg_test_loss, adv_acc, acc_c, adv_correct = 0, 0, 0, 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(validationloader):

            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)

            with torch.amp.autocast(device_type=device):

                if args.validonadv == True:
                    adv_inputs, outputs = fast_gradient_validation(model_fn=net, eps=8/255, x=inputs, y=None,
                                                                   norm=np.inf, criterion=criterion)
                    _, adv_predicted = net(adv_inputs).max(1)
                    adv_correct += adv_predicted.eq(targets).sum().item()
                else:
                    outputs = net(inputs)

                loss = criterion.test(outputs, targets)

            test_loss += loss.item()

            if args.dataset in ['WaferMap', 'KITTI_Distance_Multiclass']:
                predicted = (outputs > 0.5).float()    
            else:
                _, predicted = outputs.max(1)

            total += targets.size(0)
            if args.dataset in ['WaferMap', 'KITTI_Distance_Multiclass']:
                matches = predicted.eq(targets)  # shape: [batch_size, num_labels]
                exact_match = matches.all(dim=1)  # shape: [batch_size], bool tensor
                correct += exact_match.sum().item()
            else:
                correct += predicted.eq(targets).sum().item()
            avg_test_loss = test_loss / (batch_idx + 1)
            if pbar:
                pbar.set_description(
                '[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{}) | Adversarial Acc: {:.3f}'.format(avg_test_loss, 100. * correct / total,
                                                                    correct, total, 100. * adv_correct / total))
                pbar.update(1)

        if args.validonc == True:
            if pbar:
                pbar.set_description(
                '[Valid] Robust Accuracy Calculation. Last Robust Accuracy: {:.3f}'.format(Traintracker.valid_accs_robust[-1] if Traintracker.valid_accs_robust else 0))
                pbar.update(1)
            accs_c, avg_loss_c = compute_c_corruptions(args.dataset, testsets_c, net, batchsize=args.batchsize, num_classes=Dataloader.num_classes, 
                                                       criterion=criterion, valid_run = True, workers = 0)
            acc_c = accs_c[0]

    acc = 100. * correct / total
    adv_acc = 100. * adv_correct / total
    #print('[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{}) | Adv Acc: {:.3f} ({}/{})'.format(avg_test_loss, acc, correct, total, adv_acc, adv_correct, total))
    #print('[Valid] Robust Accuracy Calculation. Last Robust Accuracy: {:.3f}'.format(Traintracker.valid_accs_robust[-1] if Traintracker.valid_accs_robust else 0))
    return acc, avg_test_loss, acc_c, adv_acc, avg_loss_c

def manual_replay(config, start_epoch, end_epoch, resume, final=False):
    # Load and transform data
    #print('Preparing data..')

    mp.set_start_method('spawn', force=True)

    validontest = True #pbt tuning on valid data and replay on test data

    #this ensures reproducibility for model initialization on same runs. reproducibility for data augmentation and resumed training is ensured in data
    torch.manual_seed(args.run)
    torch.cuda.manual_seed(args.run)
    np.random.seed(args.run)
    random.seed(args.run)

    label_smoothing = args.lossparams.pop('label_smoothing', 0.0)
    lossparams = args.trades_lossparams | args.robust_lossparams | args.lossparams
    criterion = losses.Criterion(args.loss, trades_loss=args.trades_loss, robust_loss=args.robust_loss, **lossparams)

    Dataloader = data.DataLoading(args.dataset, validontest, args.epochs, args.resize, args.run, args.number_workers, kaggle=args.kaggle)
    Dataloader.create_transforms(args.train_aug_strat_orig, args.train_aug_strat_gen, args.style_orig, args.style_gen, 
                                 args.style_and_aug_orig, args.style_and_aug_gen, args.RandomEraseProbability, 
                                 args.aggressive_soft_crop, args.stylization_first, args.minibatchsize, label_smoothing)    
    Dataloader.load_base_data(test_only=False)

    testsets_c = Dataloader.load_data_c(subset=True, subsetsize=1000, valid_run=True) if args.validonc else None
    # Construct model
    #print(f'\nBuilding {args.modeltype} model with {args.modelparams} | Loss Function: {args.loss}, Stability Loss: {args.robust_loss}, Trades Loss: {args.trades_loss}')
    
    model_class = getattr(models, args.modeltype)
    model = model_class(dataset=args.dataset, normalized =args.normalize, num_classes=Dataloader.num_classes,
                        factor=Dataloader.factor, **args.modelparams)
    model = model.to(device) #torch.nn.DataParallel(model).to(device)

    # Define Optimizer, Learningrate Scheduler, Scaler, and Early Stopping
    opti = getattr(optim, args.optimizer)
    optimizer = opti(model.parameters(), lr=args.learningrate, **args.optimizerparams)
    schedule = getattr(optim.lr_scheduler, args.lrschedule)
    scheduler = schedule(optimizer, **args.lrparams)
    if args.warmupepochs > 0:
        warmupscheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmupepochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmupscheduler, scheduler], milestones=[args.warmupepochs])

    if args.swa["apply"] == True:
        swa_model = AveragedModel(model)
        swa_start = args.epochs * args.swa['start_factor']
        swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=args.learningrate * args.swa['lr_factor'])
    else:
        swa_model, swa_scheduler = None, None
    Scaler = torch.amp.GradScaler(device=device)
    
    Checkpointer = utils.Checkpoint(
        args.dataset, args.modeltype, args.experiment,
        train_corruptions, args.run,
        earlystopping=args.earlystop,
        patience=args.earlystopPatience,
        verbose=False,
        checkpoint_dir=Dataloader.trained_models_path,
        pbt=2
    )

    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                args.validonc, args.validonadv, args.swa, pbt=True)

    history = []
    
    # Resume from checkpoint
    if resume == True:
        _, model, swa_model, optimizer, scheduler, swa_scheduler, history = Checkpointer.load_model(model, swa_model,
                                                                    optimizer, scheduler, swa_scheduler, 'standard')
        Traintracker.load_learning_curves()
        print('\nResuming from checkpoint at epoch', start_epoch)

    total_steps, start_steps = utils.calculate_steps(Dataloader.base_trainset, Dataloader.testset, args.batchsize, args.epochs, 
                                    start_epoch, args.warmupepochs, args.validonc, args.swa['apply'], args.swa['start_factor'])
    
    with tqdm(total=total_steps, initial=start_steps) as pbar:    
        
        training_start_time = time.time()

        if style_dir := args.int_adain_params.get("style_dir", None):
            style_dataloader = Dataloader.load_style_dataloader(
                style_dir=style_dir, batch_size=args.batchsize
            )
        else: 
            style_dataloader = None
        
        # Training loop
        for epoch in range(start_epoch, end_epoch):

            #get new generated data sample in the trainset and reset the augmentation seed for corrupted data validation
            # load augmented trainset and Dataloader
            Dataloader.update_transforms(stylize_prob_orig=config.get("stylize_prob_orig", None), 
                                    stylize_prob_syn=config.get("stylize_prob_synth", None), 
                                    alpha_min_orig=config.get("alpha_min_orig", None), 
                                    alpha_min_syn=config.get("alpha_min_synth", None), 
                                    RandomEraseProbability=config.get('random_erase_prob', None))
            Dataloader.load_augmented_traindata(target_size=len(Dataloader.base_trainset),
                                                generated_ratio=config.get("synth_ratio", args.generated_ratio),
                                                epoch=epoch,
                                                robust_samples=criterion.robust_samples,
                                                grouped_stylization=args.grouped_stylization)
            trainloader, validationloader = Dataloader.get_loader(args.batchsize, 
                                                                    args.grouped_stylization)
            
            train_acc, train_loss, model, optimizer, Scaler, Dataloader, style_dataloader = train_epoch(model, trainloader, 
                                                                                                        optimizer, criterion, 
                                                                                                        Scaler, Dataloader, 
                                                                                                        style_dataloader, pbar,
                                                                                                        config.get("input_noise", None), 
                                                                                                        config.get("manifold_noise", None))
            valid_acc, valid_loss, valid_acc_robust, valid_acc_adv, valid_loss_c = valid_epoch(model, validationloader, Traintracker, Dataloader, criterion, testsets_c, pbar)
            
            if args.swa["apply"] == True and (epoch + 1) > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                valid_acc_swa, _, valid_acc_robust_swa, valid_acc_adv_swa, valid_loss_c_swa = valid_epoch(swa_model, validationloader, Traintracker, Dataloader, criterion, testsets_c, pbar)
            else:
                if args.lrschedule == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
                valid_acc_swa, valid_acc_robust_swa, valid_acc_adv_swa = valid_acc, valid_acc_robust, valid_acc_adv

            sum_acc_rob = valid_acc_robust + valid_acc
            sum_losses = valid_loss + valid_loss_c

            metrics = {
                        "epoch": epoch,
                        "val_acc": float(valid_acc),
                        "val_rob": float(valid_acc_robust),
                        "sum_acc_rob": float(sum_acc_rob),
                        "sum_losses": float(sum_losses),
                        "synth_ratio": float(config["synth_ratio"]) if "synth_ratio" in config else None,
                        "stylize_prob_orig": float(config["stylize_prob_orig"]) if "stylize_prob_orig" in config else None,
                        "stylize_prob_synth": float(config["stylize_prob_synth"]) if "stylize_prob_synth" in config else None,
                        "alpha_min_orig": float(config["alpha_min_orig"]) if "alpha_min_orig" in config else None,
                        "alpha_min_synth": float(config["alpha_min_synth"]) if "alpha_min_synth" in config else None,
                        "random_erase_prob": float(config["random_erase_prob"]) if "random_erase_prob" in config else None,
                        "input_noise": float(config["input_noise"]) if "input_noise" in config else None,
                        "manifold_noise": float(config["manifold_noise"]) if "manifold_noise" in config else None,
                    }

            history.append(metrics)

            # Check for best model, save model(s) and learning curve and check for earlystopping conditions
            elapsed_time = time.time() - training_start_time
            Checkpointer.earlystopping(valid_acc)
            Checkpointer.save_checkpoint(model, swa_model, optimizer, scheduler, swa_scheduler, epoch, history)    
            Traintracker.save_metrics(elapsed_time, train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                            valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss)
            Traintracker.save_learning_curves()

            if Checkpointer.early_stop:
                end_epoch = epoch
                break

    if final:
        # Save final model
        if args.swa["apply"] == True:
            if criterion.robust_samples >= 1:
                SWA_Loader = custom_datasets.SwaLoader(trainloader, args.batchsize, criterion.robust_samples)
                trainloader = SWA_Loader.get_swa_dataloader()
            torch.optim.swa_utils.update_bn(trainloader, swa_model, device)
            model = swa_model
        Checkpointer.save_final_model(model, optimizer, scheduler, end_epoch)
        Traintracker.print_results()
        Traintracker.save_config()


def trainable(config):
    # Load and transform data
    #print('Preparing data..')

    mp.set_start_method('spawn', force=True)

    validontest = False #pbt tuning on valid data - only replay on test data
    swa = dict(args.swa) 
    swa["apply"] = False # no swa during tuning

    #this ensures reproducibility for model initialization on same runs. reproducibility for data augmentation and resumed training is ensured in data
    torch.manual_seed(args.run)
    torch.cuda.manual_seed(args.run)
    np.random.seed(args.run)
    random.seed(args.run)

    lossparams = args.trades_lossparams | args.robust_lossparams | args.lossparams
    criterion = losses.Criterion(args.loss, trades_loss=args.trades_loss, robust_loss=args.robust_loss, **lossparams)

    Dataloader = data.DataLoading(args.dataset, validontest, args.epochs, args.resize, args.run, args.number_workers, kaggle=args.kaggle)
    Dataloader.create_transforms(args.train_aug_strat_orig, args.train_aug_strat_gen, args.style_orig, args.style_gen, 
                                 args.style_and_aug_orig, args.style_and_aug_gen, args.RandomEraseProbability, 
                                 args.aggressive_soft_crop, args.stylization_first, args.minibatchsize)    
    Dataloader.load_base_data(test_only=False)
    
    #passing the testsets preloaded so that is not done over and over again
    #testsets_c = Dataloader.load_data_c(subset=True, subsetsize=500, valid_run=True) if args.validonc else None
    testsets_c = ray.get(config['testsets_c'])
    # Construct model
    #print(f'\nBuilding {args.modeltype} model with {args.modelparams} | Loss Function: {args.loss}, Stability Loss: {args.robust_loss}, Trades Loss: {args.trades_loss}')
    
    model_class = getattr(models, args.modeltype)
    model = model_class(dataset=args.dataset, normalized =args.normalize, num_classes=Dataloader.num_classes,
                        factor=Dataloader.factor, **args.modelparams)
    model = model.to(device) #torch.nn.DataParallel(model).to(device)

    # Define Optimizer, Learningrate Scheduler, Scaler, and Early Stopping
    opti = getattr(optim, args.optimizer)
    optimizer = opti(model.parameters(), lr=args.learningrate, **args.optimizerparams)
    schedule = getattr(optim.lr_scheduler, args.lrschedule)
    scheduler = schedule(optimizer, **args.lrparams)
    if args.warmupepochs > 0:
        warmupscheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmupepochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmupscheduler, scheduler], milestones=[args.warmupepochs])

    if swa["apply"] == True:
        swa_model = AveragedModel(model)
        swa_start = args.epochs * swa['start_factor']
        swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=args.learningrate * swa['lr_factor'])
    else:
        swa_model, swa_scheduler = None, None
    Scaler = torch.amp.GradScaler(device=device)
    
    Checkpointer = utils.Checkpoint(
        args.dataset, args.modeltype, args.experiment,
        train_corruptions, args.run,
        earlystopping=args.earlystop,
        patience=args.earlystopPatience,
        verbose=False,
        checkpoint_dir=Dataloader.trained_models_path,
        pbt=1
    )

    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                args.validonc, args.validonadv, swa, pbt=True)

    start_epoch, end_epoch = 0, args.epochs + args.warmupepochs
    history = []
    
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

        # Load model state and iteration step from checkpoint.
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        # Load optimizer state (needed since we're using momentum),
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        history = checkpoint_dict.get('history', [])

        #increment the checkpointed step by 1 to get the current step.
        start_epoch = checkpoint_dict["epoch"] + 1
        
    training_start_time = time.time()
    #if args.resume == True:
    #    training_start_time = training_start_time - max(Traintracker.elapsed_time)

    if style_dir := args.int_adain_params.get("style_dir", None):
        style_dataloader = Dataloader.load_style_dataloader(
            style_dir=style_dir, batch_size=args.batchsize
        )
    else: 
        style_dataloader = None
    
    # Training loop
    for epoch in range(start_epoch, end_epoch):

        #get new generated data sample in the trainset and reset the augmentation seed for corrupted data validation
        
        # load augmented trainset and Dataloader
        Dataloader.update_transforms(stylize_prob_orig=config.get("stylize_prob_orig", None), 
                                stylize_prob_syn=config.get("stylize_prob_synth", None), 
                                alpha_min_orig=config.get("alpha_min_orig", None), 
                                alpha_min_syn=config.get("alpha_min_synth", None), 
                                RandomEraseProbability=config.get('random_erase_prob', None))
        Dataloader.load_augmented_traindata(target_size=len(Dataloader.base_trainset),
                                            generated_ratio=config.get("synth_ratio", args.generated_ratio),
                                            epoch=epoch,
                                            robust_samples=criterion.robust_samples,
                                            grouped_stylization=args.grouped_stylization)
        trainloader, validationloader = Dataloader.get_loader(args.batchsize, 
                                                                args.grouped_stylization)
        
        train_acc, train_loss, model, optimizer, Scaler, Dataloader, style_dataloader = train_epoch(model, trainloader, 
                                                                                                    optimizer, criterion, 
                                                                                                    Scaler, Dataloader, 
                                                                                                    style_dataloader, pbar=None,
                                                                                                    noise_config=config.get("input_noise", None), 
                                                                                                    manifold_noise_config=config.get("manifold_noise", None))
        valid_acc, valid_loss, valid_acc_robust, valid_acc_adv, valid_loss_c = valid_epoch(model, validationloader, Traintracker, Dataloader, criterion, testsets_c)

        if swa["apply"] == True and (epoch + 1) > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            valid_acc_swa, _, valid_acc_robust_swa, valid_acc_adv_swa, valid_loss_c = valid_epoch(swa_model, validationloader, Traintracker, Dataloader, criterion, testsets_c)
        else:
            if args.lrschedule == 'ReduceLROnPlateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()
            valid_acc_swa, valid_acc_robust_swa, valid_acc_adv_swa = valid_acc, valid_acc_robust, valid_acc_adv

        sum_acc_rob = valid_acc_robust + valid_acc
        sum_losses = valid_loss + valid_loss_c

        metrics = {
                    "epoch": epoch,
                    "val_acc": float(valid_acc),
                    "val_rob": float(valid_acc_robust),
                    "sum_acc_rob": float(sum_acc_rob),
                    "sum_losses": float(sum_losses),
                    "synth_ratio": float(config["synth_ratio"]) if "synth_ratio" in config else None,
                    "stylize_prob_orig": float(config["stylize_prob_orig"]) if "stylize_prob_orig" in config else None,
                    "stylize_prob_synth": float(config["stylize_prob_synth"]) if "stylize_prob_synth" in config else None,
                    "alpha_min_orig": float(config["alpha_min_orig"]) if "alpha_min_orig" in config else None,
                    "alpha_min_synth": float(config["alpha_min_synth"]) if "alpha_min_synth" in config else None,
                    "random_erase_prob": float(config["random_erase_prob"]) if "random_erase_prob" in config else None,
                    "input_noise": float(config["input_noise"]) if "input_noise" in config else None,
                    "manifold_noise": float(config["manifold_noise"]) if "manifold_noise" in config else None,
                }

        history.append(metrics)

        # Check for best model, save model(s) and learning curve and check for earlystopping conditions
        elapsed_time = time.time() - training_start_time
        Checkpointer.earlystopping(valid_acc)
            
        Traintracker.save_metrics(elapsed_time, train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                        valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss)
        #Traintracker.save_learning_curves()
        
        if epoch + 1 >= end_epoch:
            print('saving the latest model now')
            # Save final model
            Traintracker.print_results()
            Traintracker.save_config()
            Checkpointer.save_final_model(model, optimizer, scheduler, end_epoch)

        if epoch % config["checkpoint_interval"] == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "history": history,
                        }, os.path.join(tmpdir, "checkpoint.pt"),
                )
                tune.report(metrics=metrics, checkpoint=tune.Checkpoint.from_directory(tmpdir))
        else:
            tune.report(metrics=metrics)

        if Checkpointer.early_stop:
            end_epoch = epoch
            break

def pbt():
    storage_path = "/trained_models/ray_tune"
    experiment_name = f"config_{args.experiment}"

    hyperparameter_mutations = { #these will only be used if defined as keys in args.pbt_hyperparams
        "synth_ratio": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "stylize_prob_orig": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "stylize_prob_synth": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "alpha_min_orig": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "alpha_min_synth": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "random_erase_prob": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "input_noise": [True, False],
        "manifold_noise": [True, False],
        }
    
    Dataloader = data.DataLoading(args.dataset, False, args.epochs, args.resize, args.run, args.number_workers, kaggle=args.kaggle)
    Dataloader.create_transforms(args.train_aug_strat_orig, args.train_aug_strat_gen, args.style_orig, args.style_gen, 
                                 args.style_and_aug_orig, args.style_and_aug_gen, args.RandomEraseProbability, 
                                 args.aggressive_soft_crop, args.stylization_first, args.minibatchsize)    
    Dataloader.load_base_data(test_only=False)
    testsets_c = Dataloader.load_data_c(subset=True, subsetsize=1000, valid_run=True) if args.validonc else None
    testsets_c_ref = ray.put(testsets_c) 
    print('Succesfully pre-loaded corrupted validation data...')

    start_values = args.pbt_hyperparams
    filtered_hyperparameter_mutations, filtered_start_values = utils.filter_common_keys(start_values, 
                                                                                hyperparameter_mutations)
    final_start_values = dict(filtered_start_values) #make a copy
    final_start_values["checkpoint_interval"] = args.pbt_params.get('interval', 1)
    final_start_values["testsets_c"] = testsets_c_ref

    ray.init(
        log_to_driver=False,           # Don't log worker output to driver
        configure_logging=False,        # Don't configure Ray's logging
        logging_level=logging.ERROR,    # Only show errors
        ignore_reinit_error=True
    )
    logging.getLogger("ray").setLevel(logging.WARNING)


    # build PBT scheduler
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval = args.pbt_params.get('interval', 1),
        metric=args.pbt_params.get('metric', "sum_acc_rob"),
        mode=args.pbt_params.get('mode', "max"),
        resample_probability=args.pbt_params.get('resample_probability', 0.0),
        hyperparam_mutations=filtered_hyperparameter_mutations,
        burn_in_period = args.pbt_params.get('burn_in_period', 1.0),
        synch=True,   #sync exploitation across trials (we want synch behavior to extract parameter policies, not efficient optimal runs
    )
    
    exp_dir = os.path.join(storage_path, experiment_name)
    progress_callback = utils.PBTProgressCallback(print_every=8, show_all_params=False)

    if args.resume and tune.Tuner.can_restore(exp_dir):
        print(f"Resuming PBT experiment from {exp_dir}")
        tuner = tune.Tuner.restore(
                            exp_dir,
                            trainable=tune.with_resources(trainable, 
                            resources={'cpu': args.number_workers, 
                                    'gpu': 1}))
        tuner._local_tuner._tune_config.scheduler = pbt
    else:
    # Build the Tuner object
        tuner = tune.Tuner(
            # `trainable` can be a function or Trainable subclass; keep same API as before
            tune.with_resources(trainable, 
                                resources={'cpu': args.number_workers, 
                                        'gpu': 1}),
            param_space=final_start_values, #initial hyperparameters provided to trainable through config
            tune_config=tune.TuneConfig(
                scheduler=pbt,
                num_samples=args.pbt_params.get('trials', 24),
            ),
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=storage_path,         # where Ray stores experiments
                checkpoint_config=tune.CheckpointConfig(num_to_keep=2),
                stop={"training_iteration": args.epochs + args.warmupepochs},
                verbose=0,
                callbacks=[progress_callback], 
                log_to_file=("/dev/null", "/dev/null")
            ),
        )

    # Run tuning
    result = tuner.fit()   # returns an air.ResultGrid-like object (modern result object)

    # get best trial's result
    best_result = result.get_best_result(metric="sum_acc_rob", mode="max")
    print("Best sum_acc_rob:", best_result.metrics.get("sum_acc_rob"), 
          "Validation Accuracy: ", best_result.metrics.get("val_acc"), 
          'Robust Accuracy:', best_result.metrics.get("val_rob"))
    print("Best trial logdir:", best_result.path, ". Creating replay policy from it...")

        # Extract the last directory name
    last_dir = os.path.basename(best_result.path)  # e.g. "trainable_de242_00005_5_2025-08-24_11-02-47"

    # Use regex to capture the part after "trainable_" until the second underscore
    match = re.match(r"trainable_([^_]+_[^_]+)", last_dir)
    if not match:
        raise ValueError(f"Could not extract trial ID from {last_dir}")
    trial_id = match.group(1)  # just the pbt ID

    # Build new filename
    filename = f"pbt_policy_{trial_id}.txt"

    # Get parent directory (without last part)
    parent_dir = os.path.dirname(best_result.path)

    # Final path
    policy_path = os.path.join(parent_dir, filename)

    trial_prefix = trial_id.split("_", 1)[0]  # "de242" from "de242_00005"
    best_policy_path = os.path.join(os.path.dirname(policy_path), f"pbt_policy_{trial_prefix}_replay.txt")

    if os.path.exists(policy_path):
        # Case 1: Trial actually has a policy file → just copy it
        shutil.copyfile(policy_path, best_policy_path)
        print(f"Using existing policy file: {policy_path} and copying to {best_policy_path}")

    else:
        # Case 2: No policy file → synthesize a dummy replay log with only the final (and first) setup
        params_path = os.path.join(best_result.path, "params.json")
        with open(params_path, "r") as f:
            latest_config = json.load(f)
        dummy_entry = [
            None, trial_id, 0, 0,
            dict(latest_config),
            dict(latest_config)
        ]
        with open(best_policy_path, "w") as f_out:
            f_out.write(json.dumps(dummy_entry) + "\n")

        print(f"No policy file for trial {trial_id}. Created dummy replay file: {best_policy_path}")

    # Use the replay file (whether real or dummy)
    replay = PopulationBasedTrainingReplay(best_policy_path)
    policy = replay._policy
    initial_config = replay.config

    total_epochs = args.epochs + args.warmupepochs

    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                args.validonc, args.validonadv, args.swa, pbt=True)
        
    #save policy and plot
    plot_path = os.path.abspath(os.path.join(os.path.dirname(Traintracker.config_dst_path), f'pbt_config{args.experiment}_policy_plot_run_{args.run}.png'))
    policy_results_path = os.path.abspath(os.path.join(os.path.dirname(Traintracker.config_dst_path), f'pbt_config{args.experiment}_policy_run_{args.run}.txt'))
    
    filtered_start_values.pop("alpha_min_real", None)
    filtered_start_values.pop("alpha_min_synth", None)


    _ = utils.plot_policy_development(policy, initial_config, epochs=total_epochs, plot_keys=filtered_start_values, 
                                output_path=plot_path)
    shutil.copyfile(best_policy_path, policy_results_path)

    return policy_results_path

def replay_pbt(policy_results_path):

    # Use the replay file (whether real or dummy)
    replay = PopulationBasedTrainingReplay(policy_results_path)
    policy = replay._policy
    initial_config = replay.config
    total_epochs = args.epochs + args.warmupepochs
        
    print("REPLAY initial config:", replay.config)
    change_epochs = [t[0] for t in policy]
    if len(change_epochs)==1 and change_epochs[0] == 0:
        print("No policy changes through the epochs, using initial configuration all the way")
    else:
        print(len(policy), "policy changes at epochs:", ", ".join(map(str, change_epochs)))

    current_epoch = 0
    resume = False
    current_conf = dict(initial_config)

    # Iterate policy: each entry (change_at, new_conf) means "new_conf starts at epoch change_at".
    # So we train current_conf on epochs [current_epoch .. change_at-1], then switch to new_conf at change_at.
    for (change_at, new_conf) in policy:
        if int(change_at) > current_epoch: #if only policy is at 0 (initial policy only), jump over this to final tail
            print(f"Replaying epochs {current_epoch} to {int(change_at)} with {current_conf}")
            manual_replay(
                config=current_conf,
                start_epoch=current_epoch,
                end_epoch=int(change_at), #here, range() will iterate until change_at-1
                resume=resume,
                final=False,
            )
            resume = True  # after first segment, always resume from last checkpoint
        # apply the new config at change_at (next loop will use it)
        current_conf = dict(new_conf)
        current_epoch = int(change_at) #13

    # Final tail: train until total_epochs-1 if needed
    if current_epoch < total_epochs:
        print(f"Replaying epochs {current_epoch} to {int(change_at)} with {current_conf}")
        manual_replay(
            config=current_conf,
            start_epoch=current_epoch,
            end_epoch=total_epochs,
            resume=resume,
            final=True,
        )

def replay_only():
    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                args.validonc, args.validonadv, args.swa, pbt=True)

    policy_results_path = os.path.abspath(os.path.join(os.path.dirname(Traintracker.config_dst_path), f'pbt_' \
                                                       f'config{args.experiment}_policy_run_{args.run}.txt'))
    return policy_results_path

def plot_only():
    
    hyperparameter_mutations = { #these will only be used if defined as keys in args.pbt_hyperparams
        "synth_ratio": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "stylize_prob_orig": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "stylize_prob_synth": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "alpha_min_orig": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "alpha_min_synth": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "random_erase_prob": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "input_noise": [True, False],
        "manifold_noise": [True, False],
        }

    start_values = {"synth_ratio": 0.5, "stylize_prob_orig": 0.2, "stylize_prob_synth": 0.2}
    _, filtered_start_values = utils.filter_common_keys(start_values, hyperparameter_mutations)

    Traintracker = utils.TrainTracking(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run,
                                args.validonc, args.validonadv, args.swa, pbt=True)
    policy_results_path = os.path.abspath(os.path.join(os.path.dirname(Traintracker.config_dst_path), f'pbt_config{args.experiment}_policy_run_{args.run}.txt'))
    
    # Use the replay file (whether real or dummy)
    replay = PopulationBasedTrainingReplay(policy_results_path)
    policy = replay._policy
    initial_config = replay.config

    total_epochs = args.epochs + args.warmupepochs

    #save policy and plot
    plot_path = os.path.abspath(os.path.join(os.path.dirname(Traintracker.config_dst_path), f'pbt_config{args.experiment}_policy_plot_run_{args.run}.png'))
    
    filtered_start_values.pop("alpha_min_real", None)
    filtered_start_values.pop("alpha_min_synth", None)

    _ = utils.plot_policy_development(policy, initial_config, fontsize='xx-large', epochs=total_epochs, 
                                      plot_keys=filtered_start_values, output_path=plot_path)

if __name__ == '__main__':

    if args.mode == 'tune':
        _ = pbt()
    elif args.mode == 'replay':
        policy_results_path = replay_only()    
        replay_pbt(policy_results_path)
    elif args.mode == 'both':
        policy_results_path = pbt()    
        replay_pbt(policy_results_path)
    elif args.mode == 'plot':
        plot_only()
    else:
        print('Please provide a valid mode: "tune", "replay", or "both".')