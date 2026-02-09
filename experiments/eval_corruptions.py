import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError
from run_0 import device
from PIL import Image
from noise import apply_noise
from utils import plot_images

def select_p_corruptions(testloader, model, test_corruptions, dataset, combine_test_corruptions):
    if combine_test_corruptions:  # combined p-norm corruption robust accuracy
        accs = [compute_p_corruptions(testloader, model, test_corruptions, dataset)]
        print(accs, "% Accuracy on combined Lp-norm Test Noise")

    else:  # separate p-norm corruption robust accuracy
        accs = []
        for _, (test_corruption) in enumerate(test_corruptions):
            acc = compute_p_corruptions(testloader, model, test_corruption, dataset)
            print(acc, "% Accuracy on random test corruptions of type:", test_corruption['noise_type'],
                  test_corruption['epsilon'])
            accs = accs + [acc]
    return accs

def compute_p_corruptions(testloader, model, test_corruptions, dataset):
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            inputs_pert = apply_noise(inputs, 8, test_corruptions, 1, False, dataset)

            with torch.amp.autocast('cuda'):
                targets_pred = model(inputs_pert)

            if dataset in ['WaferMap','KITTI_Distance_Multiclass']:
                predicted = (targets_pred > 0.5).float() 
            else:  
                _, predicted = targets_pred.max(1)

            total += targets.size(0)
            
            if dataset in ['WaferMap','KITTI_Distance_Multiclass']:
                matches = predicted.eq(targets)  # shape: [batch_size, num_labels]
                exact_match = matches.all(dim=1)  # shape: [batch_size], bool tensor
                correct += exact_match.sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        return acc

def compute_c_corruptions(dataset, testsets_c, model, batchsize, num_classes, criterion = None, valid_run = False, 
                          workers = 4, multilabel = False, save_root = None):

    from data import seed_worker

    accs_c, rmsce_c_list, losses_c = [], [], []
    if valid_run == False:
        print(f"Testing on {dataset}-c/c-bar")

    t = torch.Generator()
    t.manual_seed(0) #ensure that the same testset is always used when we are not working with the fixed benchmarks

    for corruption, corruption_testset in testsets_c.items():
        testloader_c = DataLoader(corruption_testset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=workers, 
                                  worker_init_fn=seed_worker, generator=t)

        acc, rmsce_c, avg_loss_c = compute_c(testloader_c, model, num_classes, 
                                            criterion, multilabel, 
                                            dataset_name=dataset,     
                                        corruption_name=corruption,
                                        save_root=save_root)
        accs_c.append(acc)
        losses_c.append(avg_loss_c)
        rmsce_c_list.append(rmsce_c)
        if valid_run == False:
            print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

    if valid_run == False:
        rmsce_c = np.average(np.asarray(rmsce_c_list))
        print("Robust Accuracy C-all (19 corruptions): ", sum(accs_c[0:19]) / 19, "%,"
            "Robust Accuracy C-original (15 corruptions): ", sum(accs_c[0:15]) / 15, "%, "
            "Robust Accuracy C-bar (10 corruptions): ", sum(accs_c[19:29]) / 10, "%, "
            "Robust Accuracy combined ex pixel-wise noise (24 corruptions): ", (sum(accs_c[3:15]) + sum(accs_c[16:19]) + sum(accs_c[20:29])) / 24, "%, "
            "RMSCE-C average: ", rmsce_c)
        accs_c.append(sum(accs_c[0:19]) / 19)
        accs_c.append(sum(accs_c[0:15]) / 15)
        accs_c.append(sum(accs_c[19:29]) / 10)
        accs_c.append((sum(accs_c[3:15]) + sum(accs_c[16:19]) + sum(accs_c[20:29])) / 24)
        accs_c.append(rmsce_c)
    
    avg_loss_c = sum(losses_c) / len(losses_c)

    return accs_c, avg_loss_c

def compute_c(loader_c, model, num_classes, criterion = None, multilabel = False, dataset_name=None, 
              corruption_name=None, save_root=None):
    with torch.no_grad():
        model.eval()
        correct, total, loss_c = 0, 0, 0
        if multilabel:
            calibration_metric = BinaryCalibrationError(n_bins=15, norm='l1')
        else:
            calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1')
        
        all_targets = torch.empty(0, device=device)
        all_targets_pred = torch.empty((0, num_classes), device=device)

        if save_root is not None:
            if corruption_name in ['gaussian_noise','shot_noise','impulse_noise','pixelate','jpeg_compression','defocus_blur',
                                   'glass_blur','motion_blur','zoom_blur','elastic_transform','speckle_noise','gaussian_blur',
                                   'fog','frost','snow','spatter','brightness','contrast','saturate']:
                save_path = os.path.join(save_root, f"{dataset_name}-c/{corruption_name}")#
            else:
                save_path = os.path.join(save_root, f"{dataset_name}-c-bar/{corruption_name}")#-valid_0
            os.makedirs(save_root, exist_ok=True)

        uid_counter = 0

        for batch_idx, (inputs, targets) in enumerate(loader_c):

            #plot_images(5, 
            #            torch.tensor([0.0,0.0,0.0]).view(1, 3, 1, 1), 
            #            torch.tensor([1.0,1.0,1.0]).view(1, 3, 1, 1),
            #            inputs, targets)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.amp.autocast('cuda'):
                
                targets_pred = model(inputs)
                if criterion is not None:
                    loss = criterion.test(targets_pred, targets)
                    loss_c += loss.item()
            
            avg_test_loss_c = loss_c / (batch_idx + 1)

            total += targets.size(0)

            if multilabel:
                predicted = (torch.sigmoid(targets_pred) > 0.5).float()
                matches = predicted.eq(targets)  # shape: [batch_size, num_labels]
                exact_match = matches.all(dim=1)  # shape: [batch_size], bool tensor
                correct += exact_match.sum().item()
            else:
                _, predicted = targets_pred.max(1)
                correct += predicted.eq(targets).sum().item()
            
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

            if save_root is not None:
                batch_cpu = inputs.cpu()
                for i in range(batch_cpu.size(0)):
                    img_tensor = batch_cpu[i]

                    # torchvision tensor → PIL
                    img_array = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
                    
                    if img_array.shape[2] == 1:  # grayscale image
                        img_array = img_array.squeeze(2)  # H,W
                        img_pil = Image.fromarray(img_array, mode="L")
                    else:
                        img_pil = Image.fromarray(img_array)

                    # Determine class folder
                    if multilabel:
                        # Example: multilabel class folder "0_0_1_0_1_0_0_1"
                        class_key = "_".join(str(int(x)) for x in targets[i].tolist())
                    else:
                        class_key = str(int(targets[i].item()))

                    class_dir = os.path.join(save_path, class_key)
                    os.makedirs(class_dir, exist_ok=True)

                    filename = f"{uid_counter}.png"
                    uid_counter += 1
                    img_pil.save(os.path.join(class_dir, filename), "PNG")

        if multilabel:
            rmsce_c = float(calibration_metric(all_targets_pred.view(-1), all_targets.view(-1)).cpu())
        else:
            rmsce_c = float(calibration_metric(all_targets_pred, all_targets).cpu())
        acc = 100. * correct / total

        return acc, rmsce_c, avg_test_loss_c

