import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shutil
import argparse
import ast
import os
import datetime
import json
from ray.tune.callback import Callback
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Error: Boolean value expected for argument {v}.')

class str2dictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Parse the dictionary string into a dictionary object
        # if values == '':

        try:
            dictionary = ast.literal_eval(values)
            if not isinstance(dictionary, dict):
                raise ValueError("Invalid dictionary format")
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {values}") from e

        setattr(namespace, self.dest, dictionary)

def build_command_from_config(config_module, additional_params, base_cmd):
    """
    Build a command string from a config module and additional parameters.
    For dictionary parameters, we convert them to a string with double quotes,
    so that your custom argparse action (str2dictAction) can parse them correctly.
    """
    cmd_parts = [base_cmd]

    # Add parameters from the config module.
    for key in dir(config_module):
        if key.startswith("__"):
            continue  # skip built-in attributes
        value = getattr(config_module, key)
        if value is None:
            continue  # skip undefined values
        # If the parameter was already provided in additional_params, skip it.
        if key in additional_params:
            continue

        if isinstance(value, dict):
            # Convert dict to string with double quotes.
            value_str = str(value).replace("'", '"')
            cmd_parts.append(f'--{key}=\\"{value_str}\\"')
        elif isinstance(value, str):
            # Wrap strings in quotes if they contain spaces or quotes.
            if " " in value or any(c in value for c in ['"', "'"]):
                cmd_parts.append(f'--{key}="{value}"')
            else:
                cmd_parts.append(f'--{key}={value}')
        else:
            cmd_parts.append(f'--{key}={value}')

    # Append additional parameters.
    for key, value in additional_params.items():
        if isinstance(value, dict):
            value_str = str(value).replace("'", '"')
            cmd_parts.append(f'--{key}=\\"{value_str}\\"')
        elif isinstance(value, str):
            if " " in value or any(c in value for c in ['"', "'"]):
                cmd_parts.append(f'--{key}="{value}"')
            else:
                cmd_parts.append(f'--{key}={value}')
        else:
            cmd_parts.append(f'--{key}={value}')

    return " ".join(cmd_parts)

def plot_images(
    number,
    mean,
    std,
    images,
    labels=None,
    corrupted_images=None,
    second_corrupted_images=None
):
    # Convert images back from normalized space
    images = images * std + mean

    # Default labels if none provided
    if labels is None:
        labels = [""] * number

    # Determine number of columns
    columns = 1
    if corrupted_images is not None:
        corrupted_images = corrupted_images * std + mean
        columns = 2
        if second_corrupted_images is not None:
            second_corrupted_images = second_corrupted_images * std + mean
            columns = 3

    # Grid figure
    fig, axs = plt.subplots(
        number,
        columns,
        figsize=(2 * columns, number * 2),
        squeeze=False
    )

    # Move tensors to CPU
    images = images.cpu()
    corrupted_images = corrupted_images.cpu() if corrupted_images is not None else None
    second_corrupted_images = (
        second_corrupted_images.cpu() if second_corrupted_images is not None else None
    )

    # Plot each row
    for i in range(number):
        # original image
        img = images[i].squeeze().permute(1, 2, 0)
        axs[i, 0].imshow(img)
        #axs[i, 0].axis('off')
        axs[i, 0].set_xlabel(labels[i].item(), fontsize=10)

        # corrupted images
        if corrupted_images is not None:
            cimg = corrupted_images[i].squeeze().permute(1, 2, 0)
            axs[i, 1].imshow(cimg)
            #axs[i, 1].axis('off')
            axs[i, 1].set_xlabel(labels[i].item(), fontsize=10)

        if second_corrupted_images is not None:
            cimg2 = second_corrupted_images[i].squeeze().permute(1, 2, 0)
            axs[i, 2].imshow(cimg2)
            #axs[i, 2].axis('off')
            axs[i, 2].set_xlabel(labels[i].item(), fontsize=10)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(f"./corruption_examples.png")  # Save the figure
    plt.show()
    plt.close()

def calculate_steps(trainset, validset, batchsize, epochs, start_epoch, warmupepochs, validonc, swa, swa_start_factor):
    #+0.5 is a way of rounding up to account for the last partial batch in every epoch
    trainsteps_per_epoch = round(len(trainset) / batchsize + 0.5)
    validsteps_per_epoch = round(len(validset) / batchsize + 0.5)     

    if validonc == True:
        validsteps_per_epoch += 1

    if swa == True:
        total_validsteps = validsteps_per_epoch * int((2-swa_start_factor) * epochs) + warmupepochs
    else:
        total_validsteps = validsteps_per_epoch * (epochs + warmupepochs)
    total_trainsteps = trainsteps_per_epoch * (epochs + warmupepochs)

    if swa == True:
        started_swa_epochs = start_epoch - warmupepochs - int(swa_start_factor * epochs) if start_epoch - warmupepochs - int(swa_start_factor * epochs) > 0 else 0
        start_validsteps = validsteps_per_epoch * (start_epoch + started_swa_epochs)
    else:
        start_validsteps = validsteps_per_epoch * (start_epoch)
    start_trainsteps = trainsteps_per_epoch * start_epoch

    total_steps = int(total_trainsteps+total_validsteps)
    start_steps = int(start_trainsteps+start_validsteps)
    return total_steps, start_steps

class CsvHandler:
    def __init__(self, filename):
        self.filename = filename
        # Load the CSV with no header, so the first row is treated as data
        self.df = pd.read_csv(filename, header=None)
        # Rename the first column to 'corruption_name' for consistency
        self.df.rename(columns={self.df.columns[0]: 'corruption_name'}, inplace=True)
    
    def read_corruptions(self):
        """Reads the corruption data from CSV and returns a list of corruption names."""
        # Return the list of corruption names from the first column
        return self.df['corruption_name'].tolist()
    
    def get_value(self, corruption_name, severity):
        """Returns the float value from the row with the corruption_name and severity."""
        try:
            # Convert severity to a string because column names are likely strings
            if corruption_name in self.df['corruption_name'].values:
                # Retrieve the value from the DataFrame
                value = self.df.loc[self.df['corruption_name'] == corruption_name, severity].values[0]
                return float(value)
            else:
                return None
            
        except KeyError:
            return None


class Checkpoint:
    """Early stops the training if validation loss doesn't improve after a given patience.
    credit to https://github.com/Bjarten/early-stopping-pytorch/tree/master for early stopping functionality"""

    def __init__(self, dataset, modeltype, experiment, train_corruption, run, 
                 earlystopping=False, patience=7, verbose=False, delta=0, trace_func=print,
                 checkpoint_dir=f'../trained_models', pbt=0
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = False
        self.val_loss_min = 1000  # placeholder initial value
        self.delta = delta
        self.trace_func = trace_func
        self.early_stopping = earlystopping
        self.checkpoint_dir = checkpoint_dir
        if pbt == 0:
            first_placeholder = '' 
            second_placeholder = ''
        elif pbt == 1:
            first_placeholder = 'pbt_'
            second_placeholder = '_tune'
        elif pbt == 2: 
            first_placeholder = 'pbt_'
            second_placeholder = '_replay'
        else:
            raise ValueError("pbt must be 0, 1, or 2")
        self.checkpoint_path = os.path.abspath(f'{checkpoint_dir}/{first_placeholder}checkpoint_{experiment}_{run}.pt')
        self.final_model_path = os.path.abspath(f'{checkpoint_dir}/{dataset}/{modeltype}/{first_placeholder}config{experiment}_run_{run}{second_placeholder}.pth')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.final_model_path), exist_ok=True)

    def earlystopping(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.best_model = True
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.early_stopping == True:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = True

    def load_model(self, model, swa_model, optimizer, scheduler, swa_scheduler, type='standard', pbt=False):
        checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        if type == 'standard':
            filtered_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if "deepaugment_instance" not in k}
            model.load_state_dict(filtered_state_dict, strict=True)
            start_epoch = checkpoint['epoch'] + 1
        elif type == 'best':
            filtered_state_dict = {k: v for k, v in checkpoint["best_model_state_dict"].items() if "deepaugment_instance" not in k}
            model.load_state_dict(filtered_state_dict, strict=True)
            start_epoch = checkpoint['best_epoch'] + 1
        else:
            print('only best_checkpoint or checkpoint can be loaded')

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if pbt == True:
            history = checkpoint.get('history', [])
        else:
            history = []

        if swa_model != None:
            swa_filtered_state_dict = {k: v for k, v in checkpoint["swa_model_state_dict"].items() if "deepaugment_instance" not in k}
            swa_model.load_state_dict(swa_filtered_state_dict, strict=True)
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])

        return start_epoch, model, swa_model, optimizer, scheduler, swa_scheduler, history

    def save_checkpoint(self, model, swa_model, optimizer, scheduler, swa_scheduler, epoch, 
                        history=None):

        #filtered_state_dict = {k: v for k, v in model.state_dict().items() if "deepaugment_instance" not in k}

        swa_model = None if swa_model == None else swa_model.state_dict()
        swa_scheduler = None if swa_scheduler == None else swa_scheduler.state_dict()

        if self.best_model == True:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(), #filtered_state_dict
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'swa_model_state_dict': swa_model,
                'swa_scheduler_state_dict': swa_scheduler,
                'best_epoch': epoch,
                'best_model_state_dict': model.state_dict(),
            }, self.checkpoint_path)

        else:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            checkpoint['epoch'] = epoch
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint['swa_model_state_dict'] = swa_model
            checkpoint['swa_scheduler_state_dict'] = swa_scheduler

            if history is not None:
                checkpoint['history'] = history

            torch.save(checkpoint, self.checkpoint_path)

    def save_final_model(self, model, optimizer, scheduler, epoch):
        print('Final model saved to', self.final_model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, self.final_model_path)

class TrainTracking:
    def __init__(self, dataset, modeltype, lrschedule, experiment, run, validonc, validonadv, swa, pbt=False):
        self.dataset = dataset
        self.modeltype = modeltype
        self.lrschedule = lrschedule
        self.experiment = experiment
        self.run = run
        self.validonc = validonc
        self.validonadv = validonadv
        self.swa = swa
        self.train_accs, self.train_losses, self.valid_accs, self.valid_losses, self.valid_accs_robust = [],[],[],[],[]
        self.valid_accs_adv, self.valid_accs_swa, self.valid_accs_robust_swa, self.valid_accs_adv_swa = [],[],[],[]
        self.elapsed_time = []
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pbt_placeholder = 'pbt_' if pbt == True else '' #just to have different filenames for pbt and non-pbt runs
        self.csv_path = os.path.abspath(os.path.join(PROJECT_ROOT, f'results/{self.dataset}/{self.modeltype}/{pbt_placeholder}config{self.experiment}_'
                                           f'learning_curve_run_{self.run}.csv'))
        self.learningcurve_path = os.path.abspath(os.path.join(PROJECT_ROOT, f'results/{self.dataset}/{self.modeltype}/{pbt_placeholder}config{self.experiment}_'
                                                  f'learning_curve_run_{self.run}.svg'))
        self.config_src_path = os.path.abspath(os.path.join(PROJECT_ROOT, f'experiments/configs/{pbt_placeholder}config{self.experiment}.py'))
        self.config_dst_path = os.path.abspath(os.path.join(PROJECT_ROOT, f'results/{self.dataset}/{self.modeltype}/{pbt_placeholder}config{self.experiment}.py'))
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def load_learning_curves(self):

        learning_curve_frame = pd.read_csv(self.csv_path, sep=';', decimal=',')
        elapsed_time = learning_curve_frame.iloc[:, 0].values.tolist()
        train_accs = learning_curve_frame.iloc[:, 1].values.tolist()
        train_losses = learning_curve_frame.iloc[:, 2].values.tolist()
        valid_accs = learning_curve_frame.iloc[:, 3].values.tolist()
        valid_losses = learning_curve_frame.iloc[:, 4].values.tolist()
        columns=5

        valid_accs_robust, valid_accs_adv, valid_accs_swa, valid_accs_robust_swa, valid_accs_adv_swa = [],[],[],[],[]
        if self.validonc == True:
            valid_accs_robust = learning_curve_frame.iloc[:, columns].values.tolist()
            columns = columns + 1
        if self.validonadv == True:
            valid_accs_adv = learning_curve_frame.iloc[:, columns].values.tolist()
            columns = columns + 1
        if self.swa['apply'] == True:
            valid_accs_swa = learning_curve_frame.iloc[:, columns].values.tolist()
            if self.validonc == True:
                valid_accs_robust_swa = learning_curve_frame.iloc[:, columns+1].values.tolist()
                columns = columns + 1
            if self.validonadv == True:
                valid_accs_adv_swa = learning_curve_frame.iloc[:, columns+1].values.tolist()

        self.elapsed_time = elapsed_time
        self.train_accs = train_accs
        self.train_losses = train_losses
        self.valid_accs = valid_accs
        self.valid_losses = valid_losses
        self.valid_accs_robust = valid_accs_robust
        self.valid_accs_adv = valid_accs_adv
        self.valid_accs_swa = valid_accs_swa
        self.valid_accs_robust_swa = valid_accs_robust_swa
        self.valid_accs_adv_swa = valid_accs_adv_swa

    def save_metrics(self, elapsed_time, train_acc, valid_acc, valid_acc_robust, valid_acc_adv, valid_acc_swa,
                             valid_acc_robust_swa, valid_acc_adv_swa, train_loss, valid_loss):

        self.elapsed_time.append(elapsed_time)
        self.train_accs.append(train_acc)
        self.train_losses.append(train_loss)
        self.valid_accs.append(valid_acc)
        self.valid_losses.append(valid_loss)
        self.valid_accs_robust.append(valid_acc_robust)
        self.valid_accs_adv.append(valid_acc_adv)
        self.valid_accs_swa.append(valid_acc_swa)
        self.valid_accs_robust_swa.append(valid_acc_robust_swa)
        self.valid_accs_adv_swa.append(valid_acc_adv_swa)

    def save_learning_curves(self):

        learning_curve_frame = pd.DataFrame({'time': self.elapsed_time, "train_accuracy": self.train_accs, "train_loss": self.train_losses,
                                                 "valid_accuracy": self.valid_accs, "valid_loss": self.valid_losses})
        columns = 5
        if self.validonc == True:
            learning_curve_frame.insert(columns, "valid_accuracy_robust", self.valid_accs_robust)
            columns = columns + 1
        if self.validonadv == True:
            learning_curve_frame.insert(columns, "valid_accuracy_adversarial", self.valid_accs_adv)
            columns = columns + 1
        if self.swa['apply'] == True:
            learning_curve_frame.insert(columns, "valid_accuracy_swa", self.valid_accs_swa)
            if self.validonc == True:
                learning_curve_frame.insert(columns+1, "valid_accuracy_robust_swa", self.valid_accs_robust_swa)
                columns = columns + 1
            if self.validonadv == True:
                learning_curve_frame.insert(columns+1, "valid_accuracy_adversarial_swa", self.valid_accs_adv_swa)
        learning_curve_frame.to_csv(self.csv_path, index=False, header=True, sep=';', float_format='%1.4f', decimal=',')

        x = list(range(1, len(self.train_accs) + 1))
        plt.figure()
        plt.plot(x, self.train_accs, label='Train Accuracy')
        plt.plot(x, self.valid_accs, label='Validation Accuracy')
        if self.validonc == True:
            plt.plot(x, self.valid_accs_robust, label='Robust Validation Accuracy')
        if self.validonadv == True:
            plt.plot(x, self.valid_accs_adv, label='Adversarial Validation Accuracy')
        if self.swa['apply'] == True:
            swa_diff = [self.valid_accs_swa[i] if self.valid_accs[i] != self.valid_accs_swa[i] else None for i in
                        range(len(self.valid_accs))]
            plt.plot(x, swa_diff, label='SWA Validation Accuracy')
            if self.validonc == True:
                swa_robust_diff = [self.valid_accs_robust_swa[i] if self.valid_accs_robust[i] != self.valid_accs_robust_swa[i]
                            else None for i in range(len(self.valid_accs_robust))]
                plt.plot(x, swa_robust_diff, label='SWA Robust Validation Accuracy')
            if self.validonadv == True:
                swa_adv_diff = [self.valid_accs_adv_swa[i] if self.valid_accs_adv[i] != self.valid_accs_adv_swa[i]
                            else None for i in range(len(self.valid_accs_adv))]
                plt.plot(x, swa_adv_diff, label='SWA Adversarial Validation Accuracy')
        plt.title('Learning Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.linspace(1, len(self.train_accs), num=10, dtype=int))
        plt.legend(loc='best')
        plt.savefig(self.learningcurve_path)
        plt.close()

    def save_config(self):
        shutil.copyfile(self.config_src_path, self.config_dst_path)

    def print_results(self):
        if not self.elapsed_time: #if we train for 0 epochs (just loading pretrained model)
            return
        print('Total training time: ', str(datetime.timedelta(seconds=max(self.elapsed_time))))
        print("Maximum (non-SWA) validation accuracy of", max(self.valid_accs), "achieved after",
              np.argmax(self.valid_accs) + 1, "epochs; ")
        if self.validonc:
            print("Maximum (non-SWA) robust validation accuracy of", max(self.valid_accs_robust), "achieved after",
                  np.argmax(self.valid_accs_robust) + 1, "epochs; ")
        if self.validonadv:
            print("Maximum (non-SWA) adversarial validation accuracy of", max(self.valid_accs_adv), "achieved after",
                  np.argmax(self.valid_accs_adv) + 1, "epochs; ")

class TestTracking:
    def __init__(self, dataset, modeltype, experiment, runs, combine_test_corruptions,
                      test_on_c, calculate_adv_distance, calculate_autoattack_robustness,
                 test_corruptions, adv_distance_params, kaggle, pbt=False):
        self.dataset = dataset
        self.modeltype = modeltype
        self.experiment = experiment
        self.runs = runs
        self.combine_test_corruptions = combine_test_corruptions
        self.test_on_c = test_on_c
        self.calculate_adv_distance = calculate_adv_distance
        self.calculate_autoattack_robustness = calculate_autoattack_robustness
        self.test_corruptions = test_corruptions
        self.adv_distance_params = adv_distance_params
        self.kaggle = kaggle
        self.pbt = pbt
        self.pbt_placeholder = 'pbt_' if pbt == True else '' #just to have different filenames for pbt and non-pbt runs
        self.report_path_tune = os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/{self.pbt_placeholder}config{self.experiment}_result_metrics.csv')
        self.report_path_replay = os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/{self.pbt_placeholder}config{self.experiment}_result_metrics_replay.csv')
        os.makedirs(os.path.dirname(self.report_path_tune), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_path_replay), exist_ok=True)

        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paths.json")
        with open(file_path, "r") as f:
            self.path = json.load(f)
            suffix = '_kaggle' if self.kaggle else ''

            self.data_path = self.path.get(f"data{suffix}")
            self.c_labels_path = self.path.get(f"c_labels{suffix}")
            self.trained_models_path = self.path.get(f"trained_models{suffix}")

        self.eval_count = 1
        if self.runs > 1:
            self.eval_count = self.runs + 2

        self.test_count = 2
        if test_on_c:
            self.test_count += 34
        if combine_test_corruptions:
            self.test_count += 1
        else:
            self.test_count += test_corruptions.shape[0]
        if calculate_adv_distance:
            self.adv_count = len(self.adv_distance_params["norm"]) * (2+len(self.adv_distance_params["clever_samples"])) + 1
            self.test_count += self.adv_count
        if calculate_autoattack_robustness:
            self.test_count += 1

        self.all_test_metrics = np.empty([self.test_count, self.runs])
        if pbt == True:
            self.all_test_metrics_2 = np.empty([self.test_count, self.runs])

    def create_report(self):

        if self.pbt == True:
            metrics_list = [self.all_test_metrics, self.all_test_metrics_2]
        else:
            metrics_list = [self.all_test_metrics]

        for i, all_test_metrics in enumerate(metrics_list):

            test_metrics = np.empty([self.test_count, self.eval_count])

            for ide in range(self.test_count):
                test_metrics[ide, 0] = all_test_metrics[ide, :].mean()
                if self.eval_count > 1:
                    test_metrics[ide, 1] = all_test_metrics[ide, :].std()
                    for idr in range(self.runs):
                        test_metrics[ide, idr + 2] = all_test_metrics[ide, idr]

            column_string = np.array([f'config_{self.experiment}_avg'])
            if self.eval_count > 1:
                column_string = np.append(column_string, [f'config_{self.experiment}_std'], axis=0)
                for idr in range(self.runs):
                    column_string = np.append(column_string, [f'config_{self.experiment}_run_{idr}'], axis=0)

            test_metrics_string = np.array(['Standard_Acc', 'RMSCE'])
            if self.test_on_c == True:
                test_corruptions_label = np.loadtxt(os.path.abspath(f'{self.c_labels_path}/c-labels.txt'), dtype=list)
                if self.dataset in ['CIFAR10', 'CIFAR100', 'GTSRB', 'WaferMap']:
                    test_corruptions_bar_label = np.loadtxt(os.path.abspath(f'{self.c_labels_path}/c-bar-labels-cifar.txt'), dtype=list)
                elif self.dataset in ['ImageNet', 'ImageNet-100', 'TinyImageNet', 'EuroSAT', 'PCAM', 'KITTI_RoadLane', 'SynthiCAD',
                       'KITTI_Distance_Multiclass', 'TreeSAT', 'Casting-Product-Quality', 'Describable-Textures', 'Flickr-Material']:
                    test_corruptions_bar_label = np.loadtxt(os.path.abspath(f'{self.c_labels_path}/c-bar-labels-IN.txt'), dtype=list)
                else:
                    print('no c-bar corruption types defined for this dataset')
                test_metrics_string = np.append(test_metrics_string, test_corruptions_label, axis=0)
                test_metrics_string = np.append(test_metrics_string, test_corruptions_bar_label, axis=0)
                test_metrics_string = np.append(test_metrics_string,
                                                    ['Acc_C-all-19', 'Acc_C-original-15', 'Acc_C-bar-10', 'Acc_all-ex-pixelwise-noise-24', 'RMSCE_C'],
                                                    axis=0)

            if self.calculate_adv_distance == True:
                test_metrics_string = np.append(test_metrics_string, ['Acc_from_adv_dist_calculation'])
                for _, n in enumerate(self.adv_distance_params["norm"]):
                    test_metrics_string = np.append(test_metrics_string,
                                                        [f'{n}-norm-Mean_adv_dist_with_misclassifications_0',
                                                        f'{n}-norm-Mean_adv_dist_without_misclassifications'], axis=0)
                    for _, b in enumerate(self.adv_distance_params["clever_samples"]):
                        test_metrics_string = np.append(test_metrics_string,
                                                            [f'{n}-norm-Mean_CLEVER-{b}-samples'], axis=0)
            if self.calculate_autoattack_robustness == True:
                test_metrics_string = np.append(test_metrics_string,
                                                    ['Adversarial_accuracy_autoattack'])
            if self.combine_test_corruptions == True:
                test_metrics_string = np.append(test_metrics_string, ['Combined_Noise'])
            else:
                test_corruptions_labels = np.array([','.join(map(str, row.values())) for row in self.test_corruptions])
                test_metrics_string = np.append(test_metrics_string, test_corruptions_labels)

            report_frame = pd.DataFrame(test_metrics, index=test_metrics_string,
                                            columns=column_string)
            if i == 0:
                report_frame.to_csv(self.report_path_tune, index=True, header=True, sep=';', float_format='%1.4f', decimal=',')
            elif i == 1:
                report_frame.to_csv(self.report_path_replay, index=True, header=True, sep=';', float_format='%1.4f', decimal=',')

    def initialize(self, run):
        self.run = run
        self.accs = []
        
        print(f"Evaluating training run {run}")
        
        if self.pbt == True:
            self.filenames = [os.path.abspath(f'{self.trained_models_path}/{self.dataset}/{self.modeltype}/{self.pbt_placeholder}config{self.experiment}' \
                   f'_run_{run}_tune.pth'),
                   os.path.abspath(f'{self.trained_models_path}/{self.dataset}/{self.modeltype}/{self.pbt_placeholder}config{self.experiment}' \
                   f'_run_{run}_replay.pth')]
        else:
            self.filenames = [os.path.abspath(f'{self.trained_models_path}/{self.dataset}/{self.modeltype}/{self.pbt_placeholder}config{self.experiment}' \
                   f'_run_{run}.pth')]
    
            
    def track_results(self, new_results, i):
        
        for element in new_results:
                self.accs.append(element)
        if i == 1:
            self.all_test_metrics_2[:len(self.accs), self.run] = np.array(self.accs)
        else:
            self.all_test_metrics[:len(self.accs), self.run] = np.array(self.accs)    

    def save_adv_distance(self, dist_sorted, adv_distance_params):

        self.adv_report_path = os.path.abspath(f'./results/{self.dataset}/{self.modeltype}/config{self.experiment}_'
                                  f'run_{self.run}_adversarial_distances.csv')
        os.makedirs(os.path.dirname(self.adv_report_path), exist_ok=True)

        if adv_distance_params["clever"] == False:
            adv_distance_params["clever_batches"], adv_distance_params["clever_samples"] = [0.0], [0.0]
        columns = []
        for x in adv_distance_params["norm"]:
            columns.append(f"{x}-norm-min-adv-dist")
            columns.extend([f"{x}-norm-PGD-dist", f"{x}-norm-sec-att-dist"])
            columns.extend([f"{x}-norm-Clever-{y}-samples" for y in adv_distance_params["clever_samples"]])

        adv_distance_frame = pd.DataFrame(index=range(adv_distance_params["setsize"]), columns=columns)
        col_counter = 0

        for id, n in enumerate(adv_distance_params["norm"]):
            adv_distance_frame.iloc[:, col_counter:col_counter+3] = dist_sorted[:, id*3:(id+1)*3]
            col_counter += 3

            for j, (batches, samples) in enumerate(zip(adv_distance_params["clever_batches"], adv_distance_params["clever_samples"])):

                indices1 = np.where((dist_sorted[:,id*3+1] <= dist_sorted[:,id*3+2]) & (dist_sorted[:, id*3+1] != 0))[0]
                indices2 = np.where((dist_sorted[:,id*3+2] < dist_sorted[:,id*3+1]) & (dist_sorted[:, id*3+2] != 0))[0]
                # Find indices where column id*3+1 is 0 and column id*3+2 is not 0
                indices_zero1 = np.where((dist_sorted[:,id*3+1] == 0) & (dist_sorted[:,id*3+2] != 0))[0]
                # Find indices where column id*3+2 is 0 and column id*3+1 is not 0
                indices_zero2 = np.where((dist_sorted[:,id*3+2] == 0) & (dist_sorted[:,id*3+1] != 0))[0]
                # Find indices where both are 0 and asign them to PGD attack
                indices_doublezero = np.where((dist_sorted[:, id * 3 + 2] == 0) & (dist_sorted[:, id * 3 + 1] == 0))[0]
                # Concatenate the indices with appropriate conditions
                indices1 = np.concatenate((indices1, indices_zero2, indices_doublezero))
                indices2 = np.concatenate((indices2, indices_zero1))

                adv_fig = plt.figure(figsize=(15, 5))
                plt.scatter(indices1, dist_sorted[:,id*3+1][indices1], s=5, label="PGD Adversarial Distance")
                plt.scatter(indices2, dist_sorted[:,id*3+2][indices2], s=5, label="Second Attack Adversarial Distance")
                if adv_distance_params["clever"]:
                    plt.scatter(range(len(dist_sorted[:,len(adv_distance_params["norm"]) * 3 + id *
                                                        len(adv_distance_params["clever_batches"]) + j])),
                                dist_sorted[:,len(adv_distance_params["norm"]) * 3 + id * len(adv_distance_params["clever_batches"]) + j],
                                s=5, label=f"Clever Score: {samples} samples")
                plt.title(f"{n}-norm adversarial distance vs. CLEVER score")
                plt.xlabel("Image ID sorted by adversarial distance")
                plt.ylabel("Distance")
                plt.legend()
                plt.close()

                adv_fig.savefig(os.path.abspath(f'results/{self.dataset}/{self.modeltype}/config{self.experiment}_run'
                                f'_{self.run}_adversarial_distances_{n}-norm_{samples}-CLEVER-samples.svg'))
                adv_distance_frame.iloc[:, col_counter] = dist_sorted[:,len(adv_distance_params["norm"])*3+id*
                                                                     len(adv_distance_params["clever_batches"]) + j]
                col_counter += 1

        adv_distance_frame.to_csv(self.adv_report_path,
                                  index=False, header=True, sep=';', float_format='%1.4f', decimal=',')


# Custom Progress Callback - Recommended for your setup
class PBTProgressCallback(Callback):
    """Custom callback for PBT progress tracking with time logging"""
    
    def __init__(self, print_every=5, show_all_params=False):
        """
        Args:
            print_every: Print progress every N trial results
            show_all_params: Whether to show all hyperparameters or just key ones
        """
        self.print_every = print_every
        self.show_all_params = show_all_params
        self.result_count = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        
        # Print header
        print("\n" + "="*120)
        print(f"PBT EXPERIMENT STARTED - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120)
        if show_all_params:
            print(f"{'Trial':<12} {'Epoch':<6} {'Val_Acc':<8} {'Val_Rob':<8} {'Sum':<8} {'Time':<8} {'Synth':<6} {'Style_R':<8} {'Style_S':<8} {'Alpha_R':<8} {'Alpha_S':<8} {'RE_Prob':<8} {'In_N':<5} {'Man_N':<5}")
        else:
            print(f"{'Trial':<12} {'Epoch':<6} {'Val_Acc':<8} {'Val_Rob':<8} {'Sum':<8} {'Time':<8} {'Synth':<6} {'Style_R':<8} {'Style_S':<8}")
        print("-"*120)
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.result_count += 1
        
        if self.result_count % self.print_every == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            since_last = current_time - self.last_print_time
            self.last_print_time = current_time
            
            # Format time as HH:MM:SS
            elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
            
            # Extract key metrics
            trial_name = trial.trial_id[-8:] if trial.trial_id else "unknown"
            epoch = result.get('epoch', 0)
            val_acc = result.get('val_acc', 0.0)
            val_rob = result.get('val_rob', 0.0)
            sum_acc_rob = result.get('sum_acc_rob', 0.0)
            
            # Extract hyperparameters (handle None values)
            synth_ratio = result.get('synth_ratio', 0.0) or 0.0
            style_real = result.get('stylize_prob_orig', 0.0) or 0.0
            style_synth = result.get('stylize_prob_synth', 0.0) or 0.0
            
            if self.show_all_params:
                alpha_real = result.get('alpha_min_orig', 0.0) or 0.0
                alpha_synth = result.get('alpha_min_synth', 0.0) or 0.0
                re_prob = result.get('random_erase_prob', 0.0) or 0.0
                input_n = "T" if result.get('input_noise', False) else "F"
                manifold_n = "T" if result.get('manifold_noise', False) else "F"
                
                print(f"{trial_name:<12} {epoch:<6} {val_acc:<8.3f} {val_rob:<8.3f} {sum_acc_rob:<8.3f} {elapsed_str:<8} {synth_ratio:<6.2f} {style_real:<8.2f} {style_synth:<8.2f} {alpha_real:<8.2f} {alpha_synth:<8.2f} {re_prob:<8.2f} {input_n:<5} {manifold_n:<5}")
            else:
                print(f"{trial_name:<12} {epoch:<6} {val_acc:<8.3f} {val_rob:<8.3f} {sum_acc_rob:<8.3f} {elapsed_str:<8} {synth_ratio:<6.2f} {style_real:<8.2f} {style_synth:<8.2f}")
    
    def on_experiment_end(self, trials, **info):
        total_time = time.time() - self.start_time
        total_time_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
        
        print("-"*120)
        print(f"EXPERIMENT COMPLETED - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_time_str}")
        print(f"Total trials: {len(trials)}")
        
        # Show best trial
        if trials:
            best_trial = max(trials, key=lambda t: t.last_result.get('sum_acc_rob', 0.0) if t.last_result else 0.0)
            if best_trial.last_result:
                print(f"Best trial: {best_trial.trial_id}")
                print(f"Best sum_acc_rob: {best_trial.last_result.get('sum_acc_rob', 0.0):.3f}")
                print(f"Best val_acc: {best_trial.last_result.get('val_acc', 0.0):.3f}")
                print(f"Best val_rob: {best_trial.last_result.get('val_rob', 0.0):.3f}")
        print("="*120)

class MinimalPBTCallback(Callback):
    """Ultra-minimal progress callback"""
    
    def __init__(self, print_every=10):
        self.print_every = print_every
        self.result_count = 0
        self.start_time = time.time()
        print(f"\nPBT Started: {datetime.now().strftime('%H:%M:%S')}")
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.result_count += 1
        
        if self.result_count % self.print_every == 0:
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            
            active_trials = len([t for t in trials if t.status == "RUNNING"])
            best_score = max([t.last_result.get('sum_acc_rob', 0.0) for t in trials if t.last_result], default=0.0)
            
            print(f"[{elapsed_str}] Active: {active_trials}, Best: {best_score:.3f}, Results: {self.result_count}")



def project_dict(d: dict, index: int) -> dict:
        return {k: v[index] for k, v in d.items()}

def filter_common_keys(start_values, hyperparameter_mutations):
    """
    Select keys from start_values and their corresponding values from hyperparameter_mutations.
    Only includes keys that exist in both dictionaries.
    
    Args:
        start_values (dict): Dictionary containing the reference keys to loop over
        hyperparameter_mutations (dict): Dictionary to get values from
    
    Returns:
        dict: Dictionary with keys from start_values and values from hyperparameter_mutations
    """
    filtered_mutations = {}
    filtered_start_values = {}
    
    for key in start_values:
        if key in hyperparameter_mutations:
            filtered_mutations[key] = hyperparameter_mutations[key]
            filtered_start_values[key] = start_values[key]
        else:
            print(f"Warning: Key '{key}' from start_values not found in hyperparameter_mutations")
    
    return filtered_mutations, filtered_start_values

def plot_policy_development(policy_list, initial_config, fontsize='medium', epochs=300, plot_keys=None, output_path=None):
    """
    Simplified plotting function that stacks all selected keys vertically and optionally saves the figure.
    
    Parameters
    ----------
    policy_list : list
        List of (epoch, config_dict) tuples where epoch is 0-based and the config takes effect starting at displayed epoch epoch+1.
    initial_config : dict
        Initial config for displayed epoch 1.
    epochs : int
        Number of displayed epochs (x-axis).
    drop_last : int
        Number of keys (from ordered first-appearance list) to drop from plotting.
    output_path : str or None
        If provided, save the resulting figure to this path (e.g., '/mnt/data/policy_plot.png').
    """
    # Validate inputs
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if output_path is not None:
        outdir = os.path.dirname(output_path)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
    
    # Sort policy list by epoch
    policy_sorted = sorted(policy_list, key=lambda t: t[0])
    
    # Build segments: displayed epochs are 1..epochs
    starts = [e + 1 for e, _ in policy_sorted]
    starts_with_initial = [1] + starts
    configs = [initial_config] + [cfg for _, cfg in policy_sorted]
    segments = []
    for i, cfg in enumerate(configs):
        start = starts_with_initial[i]
        end = (starts_with_initial[i+1] - 1) if (i+1) < len(starts_with_initial) else epochs
        start = max(1, start)
        end = min(epochs, end)
        if start <= end:
            segments.append((start, end, cfg))
    
    # Ordered keys by first appearance
    ordered_keys = []
    for cfg in [initial_config] + [c for _, c in policy_sorted]:
        for k in cfg.keys():
            if k not in ordered_keys:
                ordered_keys.append(k)
    
    # Determine keys to plot and check for missing keys
    if plot_keys is None:
        keys_to_plot = ordered_keys[:]
    else:
        keys_to_plot = [k for k in ordered_keys if k in plot_keys]
        # Soft error for keys in policy but not in plot_keys
        missing_keys = [k for k in ordered_keys if k not in plot_keys]
        if missing_keys:
            print(f"Warning: The following keys are present in the policy but not in plot_keys dictionary: {missing_keys}")
    
    print("Keys to plot:", keys_to_plot)
    
    if len(keys_to_plot) == 0:
        print("No keys to plot (no matching keys in plot_keys dictionary). Exiting without saving.")
        return None
    
    # Build x axis
    x = np.arange(1, epochs + 1)
    
    # Prepare figure with stacked subplots (wide and low)
    n = len(keys_to_plot)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(16, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    
    # For each key, build y array and plot
    for ax, key in zip(axes, keys_to_plot):
        y = np.full_like(x, np.nan, dtype=float)
        for (s, e, cfg) in segments:
            if key in cfg:
                raw_val = cfg[key]
                # convert booleans to 0/1
                if isinstance(raw_val, bool):
                    val = float(int(raw_val))
                # numeric types to float
                elif isinstance(raw_val, (int, float, np.integer, np.floating)):
                    val = float(raw_val)
                else:
                    # try to coerce strings like '0'/'1' but otherwise raise informative error:
                    try:
                        val = float(raw_val)
                    except Exception:
                        raise ValueError(f"Key '{key}' has non-numeric non-boolean value '{raw_val}'. "
                                         "This simplified function only accepts numeric values or booleans for plotting. "
                                         "Consider increasing drop_last or removing this key from configs.")
                y[(s-1):e] = val
        # If y contains NaNs entirely, warn and skip plotting this key
        if np.all(np.isnan(y)):
            print(f"Warning: key '{key}' had no values across segments; skipping.")
            continue
        
        ax.step(x, y, where='post')
        ax.scatter(x, y, s=6)
        ax.set_ylabel(key, rotation=0, labelpad=70, va='center', fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        # vertical lines at segment starts (skip initial)
        for s, _, _ in segments[1:]:
            ax.axvline(s, linestyle='--', alpha=0.25)
    
    axes[-1].set_xlabel("Epoch", fontsize=fontsize)
    axes[-1].tick_params(axis='x', labelsize=fontsize)
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to: {output_path}")
    
    #plt.show()
    return fig