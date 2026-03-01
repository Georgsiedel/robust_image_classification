import os
import torch
import importlib
from experiments.utils import build_command_from_config
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    for experiment in [546,526,521,520,496,476,471,470]:#

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        stylization_first = False
        kaggle = False

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 1
        run_iters = [0]

        for run in run_iters:

            resume = True if experiment in [546] and run in [0] else False

            print("Training run #",run)
            cmd0 = f"python experiments/train.py --resume={resume} --run={run} --experiment={experiment} --epochs=" \
                    f"{config.epochs} --learningrate={config.learningrate} --dataset={config.dataset} --validontest=" \
                    f"{config.validontest} --lrschedule={config.lrschedule} --lrparams=\"{config.lrparams}\" " \
                    f"--earlystop={config.earlystop} --earlystopPatience={config.earlystopPatience} --optimizer=" \
                    f"{config.optimizer} --optimizerparams=\"{config.optimizerparams}\" --modeltype=" \
                    f"{config.modeltype} --modelparams=\"{config.modelparams}\" --resize={config.resize} " \
                    f"--aggressive_soft_crop={config.aggressive_soft_crop} " \
                    f"--style_orig=\"{config.style_orig}\" --train_aug_strat_orig={config.train_aug_strat_orig} " \
                    f"--style_gen=\"{config.style_gen}\" --train_aug_strat_gen={config.train_aug_strat_gen} " \
                    f"--style_and_aug_orig={config.style_and_aug_orig} --style_and_aug_gen={config.style_and_aug_gen} " \
                    f"--loss={config.loss} --lossparams=\"{config.lossparams}\" --trades_loss={config.trades_loss} " \
                    f"--trades_lossparams=\"{config.trades_lossparams}\" --robust_loss={config.robust_loss} " \
                    f"--robust_lossparams=\"{config.robust_lossparams}\" --mixup=\"{config.mixup}\" --cutmix=" \
                    f"\"{config.cutmix}\" --manifold=\"{config.manifold}\" --concurrent_combinations={config.concurrent_combinations}" \
                    f" --batchsize={config.batchsize} --number_workers={config.number_workers} " \
                    f"--RandomEraseProbability={config.RandomEraseProbability} --warmupepochs={config.warmupepochs}" \
                    f" --normalize={config.normalize} --minibatchsize=" \
                    f"{config.minibatchsize} --validonc={config.validonc} --validonadv={config.validonadv} --swa=" \
                    f"\"{config.swa}\" --noise_sparsity={config.noise_sparsity} --noise_patch_scale=" \
                    f"\"{config.noise_patch_scale}\" --generated_ratio={config.generated_ratio} " \
                    f"--n2n_deepaugment={config.n2n_deepaugment} --stylization_first={stylization_first} " \
                    f"--kaggle={kaggle} "
            if experiment in []:
                print('skip')
            else:
                os.system(cmd0)
            
        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption

        print('Beginning metric evaluation')
        cmdeval = f"python experiments/eval.py --experiment={experiment} --runs={runs} --batchsize={config.batchsize} " \
                f"--dataset={config.dataset} --modeltype={config.modeltype} --modelparams=\"{config.modelparams}\" " \
                f"--resize={config.resize} --combine_test_corruptions={config.combine_test_corruptions} --number_workers={config.number_workers} " \
                f"--normalize={config.normalize} --test_on_c={config.test_on_c} " \
                f"--calculate_adv_distance={config.calculate_adv_distance} --adv_distance_params=\"{config.adv_distance_params}\" " \
                f"--calculate_autoattack_robustness={config.calculate_autoattack_robustness} --autoattack_params=" \
                f"\"{config.autoattack_params}\" --validontest={config.validontest} --kaggle={kaggle} " \
                
        os.system(cmdeval)
