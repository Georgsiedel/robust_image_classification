import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
module_path = os.path.abspath(os.path.dirname(__file__))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
import argparse
import importlib
from run_0 import device
import models
import eval_adversarial
import eval_corruptions
import data
import utils

parser = argparse.ArgumentParser(description='PyTorch Robustness Testing')
parser.add_argument('--runs', default=1, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--pbt', type=utils.str2bool, nargs='?', const=False, default=False, help='Whether the model was trained with PBT. ' \
                    'If True, evaluate both the training with valid data and PBT tuning and the replayed training on full data.')
parser.add_argument('--batchsize', default=1000, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=utils.str2bool, nargs='?', const=True, default=True, help='True: Use full '
                    'training data and test data. False: 80:20 train:valiation split, validation also used for testing.')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--combine_test_corruptions', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to combine all testing noise values by drawing from the randomly')
parser.add_argument('--number_workers', default=0, type=int, help='How many workers are launched to parallelize data loading.')
parser.add_argument('--normalize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--test_on_c', type=utils.str2bool, nargs='?', const=True, default=True,
                    help='Whether to test on corrupted benchmark sets C and C-bar')
parser.add_argument('--calculate_adv_distance', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial distance with adv_distance_params')
parser.add_argument('--adv_distance_params', default={'setsize': 500, 'nb_iters': 200, 'eps_iter': 0.0003, 'norm': 'np.inf',
                        "epsilon": 0.1, "clever": True, "clever_batches": 50, "clever_samples": 50},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--calculate_autoattack_robustness', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial accuracy with Autoattack with autoattack_params')
parser.add_argument('--autoattack_params', default={'setsize': 500, 'epsilon': 8/255, 'norm': 'Linf'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
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
test_corruptions = config.test_corruptions

def compute_clean(testloader, model, num_classes, multilabel = False):
    with torch.no_grad():
        correct = 0
        total = 0

        if multilabel:
            calibration_metric = BinaryCalibrationError(n_bins=15, norm='l1')
        else:
            calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1')
        calibration_metric = calibration_metric.to(device)    

        for inputs, targets in testloader:
            
            total += targets.size(0)
            
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)

            if multilabel:
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                matches = predicted.eq(targets)  # shape: [batch_size, num_labels]
                exact_match = matches.all(dim=1)  # shape: [batch_size], bool tensor
                correct += exact_match.sum().item()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            calibration_metric.update(outputs, targets)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric.compute().cpu())
        print("Clean Accuracy ", acc, "%, RMSCE Calibration Error: ", rmsce_clean)

        return acc, rmsce_clean

if __name__ == '__main__':
    Testtracker = utils.TestTracking(args.dataset, args.modeltype, args.experiment, args.runs,
                                args.combine_test_corruptions, args.test_on_c,
                                args.calculate_adv_distance, args.calculate_autoattack_robustness,
                                test_corruptions, args.adv_distance_params, args.kaggle, args.pbt)

    for run in range(args.runs):

        Testtracker.initialize(run)

        for i, filename in enumerate(Testtracker.filenames):

            # Load data
            Dataloader = data.DataLoading(dataset=args.dataset, validontest=args.validontest, resize=args.resize, 
                                        run=run, number_workers=args.number_workers, kaggle=args.kaggle)
            Dataloader.create_transforms(train_aug_strat_orig='None', train_aug_strat_gen='None')
            Dataloader.load_base_data(test_only=True)

            workers = args.number_workers
            if args.dataset in ['CIFAR10','CIFAR100'] and args.validontest:
                workers = 0
            
            testloader = torch.utils.data.DataLoader(Dataloader.testset, batch_size=args.batchsize, pin_memory=True, num_workers=workers)

            # Load model
            model_class = getattr(models, args.modeltype)
            model = model_class(dataset=args.dataset, normalized=args.normalize, num_classes=Dataloader.num_classes,
                                factor=Dataloader.factor, **args.modelparams)
            model = model.to(device) #torch.nn.DataParallel(

            state_dict = torch.load(filename, weights_only=True)['model_state_dict']

            # Remove "module." prefix from keys
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")  # Remove "module." prefix
                new_state_dict[new_key] = v

            # Load the modified state_dict into the original model
            model.load_state_dict(new_state_dict, strict=False)

            model.eval()

            # Clean Test Accuracy
            acc, rmsce = compute_clean(testloader, model, Dataloader.num_classes, Dataloader.multilabel)
            Testtracker.track_results([acc, rmsce], i)

            if args.test_on_c == True:  # C-dataset robust accuracy
                subset = False if args.validontest else True
                subsetsize = None if args.validontest else 1000 #subset for quick validation runs
                
                testsets_c = Dataloader.load_data_c(subset=subset, subsetsize=subsetsize, valid_run=False,
                                                    load_root="../data")
                if args.experiment in [389]:
                    root="../data"
                else:
                    root=None
                accs_c = eval_corruptions.compute_c_corruptions(args.dataset, testsets_c, model, args.batchsize, Dataloader.num_classes, 
                                                                valid_run=False, workers=workers, multilabel=Dataloader.multilabel,
                                                                save_root=root)[0]
                Testtracker.track_results(accs_c, i)

            if args.calculate_adv_distance == True:  # adversarial distance calculation
                adv_acc, dist_sorted, mean_dist = eval_adversarial.compute_adv_distance(Dataloader.testset,
                                                                0, model, args.adv_distance_params)
                Testtracker.track_results(np.concatenate(([adv_acc], mean_dist)), i)
                Testtracker.save_adv_distance(dist_sorted, args.adv_distance_params)

            if args.calculate_autoattack_robustness == True:  # adversarial accuracy calculation
                adv_acc_aa = eval_adversarial.compute_adv_acc(args.autoattack_params, Dataloader.testset,
                                                                            model, 0, args.batchsize)
                Testtracker.track_results([adv_acc_aa], i)

            # Robust Accuracy on p-norm noise - either combined or separate noise types
            accs = eval_corruptions.select_p_corruptions(testloader, model, test_corruptions, args.dataset, args.combine_test_corruptions)
            if not args.combine_test_corruptions:
                # Compute combined average robustness metrics: Class Separation Robustness, imperceptible Robustness, Average p-norm robustness
                class_separation_robustness = float(np.mean(accs[0:3]))
                imperceptible_robustness = float(np.mean(accs[3:9]))
                average_pnorm_robustness = float(np.mean(accs[9:]))
                accs.extend([class_separation_robustness, imperceptible_robustness, average_pnorm_robustness])
            Testtracker.track_results(accs, i)

            print(Testtracker.accs)
            Testtracker.accs = []

    Testtracker.create_report()