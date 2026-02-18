import torch
from run_0 import device
import numpy as np
from torch.utils.data import DataLoader
from autoattack import AutoAttack
from art.metrics import clever_u
from cleverhans.torch.utils import optimize_linear
import torch.nn as nn
import foolbox as fb

# Import your constants/paths if needed
from run_0 import device

def fast_gradient_validation(model_fn, x, eps, norm, criterion, clip_min=None, clip_max=None, y=None, targeted=False,
    sanity_checks=False):

    """PyTorch implementation of the Fast Gradient Method. from Cleverhans package"""

    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)

    with torch.enable_grad():
        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            outputs = model_fn(x)
            _, y = torch.max(outputs, 1)

        # Compute loss
        loss = criterion.test(model_fn(x), y)

        # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        loss.backward()
        optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x, outputs

def compute_adv_acc(autoattack_params, testset, model, workers, batchsize=50):
    print(f"{autoattack_params['norm']}-norm Adversarial Accuracy calculation using AutoAttack "
          f"with epsilon={autoattack_params['epsilon']}")
    
    autoattack_params["setsize"] = min(autoattack_params["setsize"], len(testset))

    truncated_testset, _ = torch.utils.data.random_split(testset, [autoattack_params["setsize"],
                                len(testset)-autoattack_params["setsize"]], generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=autoattack_params["setsize"], shuffle=False,
                                       pin_memory=True, num_workers=workers)
    adversary = AutoAttack(model, norm=autoattack_params['norm'], eps=autoattack_params['epsilon'], version='standard')
    correct, total = 0, 0
    if autoattack_params["norm"] == 'Linf':
        autoattack_params["norm"] = np.inf
    else:
        autoattack_params["norm"] = autoattack_params["norm"][1:]
    for batch_id, (inputs, targets) in enumerate(truncated_testloader):
        adv_inputs, adv_predicted = adversary.run_standard_evaluation(inputs, targets, bs=batchsize, return_labels=True)

    correct += (adv_predicted == targets).sum().item()
    total += targets.size(0)
    adv_acc = correct / total * 100
    return adv_acc

def clever_score(testloader, evalmodel, clever_batches, clever_samples, epsilon, norm, setsize):
    torch.cuda.empty_cache()
    if len(clever_samples) != len(clever_batches):
        print('!!! clever_samples needs to be of same size as clever_batches!')
    clever_scores = np.empty([setsize, len(clever_samples)*len(norm)])
    mean_clever_array = np.empty([len(norm)*len(clever_batches)])

    for id, (n, eps) in enumerate(zip(norm, epsilon)):
        for j, (batches, samples) in enumerate(zip(clever_batches, clever_samples)):
            print(f'Clever calculation for {n}-norm with {samples} samples and {batches} batches')
            # Iterate through each image for CLEVER score calculation
            for batch_idx, (inputs, targets) in enumerate(testloader):
                for r, input in enumerate(inputs):
                    clever_score = clever_u(evalmodel,
                                            input.numpy(),
                                            nb_batches=batches,
                                            batch_size=samples,
                                            radius=eps,
                                            norm=n,
                                            pool_factor=10,
                                            verbose=False)

                    # Append the calculated CLEVER score to the list
                    clever_scores[batch_idx, id*len(clever_batches)+j] = clever_score
                if (batch_idx + 1) % 10 == 0:
                    print(f"Completed: {batch_idx + 1} of {setsize}, mean CLEVER score: "
                          f"{np.mean(clever_scores[:(batch_idx + 1), id*len(clever_batches)+j])}")
            mean_clever_array[id*len(clever_batches)+j] = np.mean(clever_scores[:, id*len(clever_batches)+j])
    return clever_scores, mean_clever_array

def batch_pgd_with_early_stopping(model, inputs, labels, iterations, epsilon_step, norm):
    """
    Implements PGD with early stopping using Foolbox for the gradient step.
    
    Logic:
    1. Identify samples that are not yet misclassified.
    2. Use Foolbox PGD (steps=1) to take exactly one gradient step on those samples.
    3. We pass epsilons=inf to Foolbox to effectively disable the maximum epsilon constraint,
       allowing the attack to 'walk' as far as needed (constrained only by iterations).
    """
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Strictly clamp clean inputs to [0,1]
    inputs = torch.clamp(inputs, 0.0, 1.0)
    
    # Initialize adversarial examples as clones of inputs
    adv_inputs = inputs.clone()

    # Select the correct Foolbox attack class based on norm
    # We initialize with steps=1 to perform a single update per loop iteration
    if norm == np.inf:
        attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=1, abs_stepsize=epsilon_step)
    elif norm == 2:
        attack = fb.attacks.L2ProjectedGradientDescentAttack(steps=1, abs_stepsize=epsilon_step)
    elif norm == 1:
        # Note: L1 PGD is sparse/complex; standard Foolbox L1PGD works here.
        attack = fb.attacks.L1ProjectedGradientDescentAttack(steps=1, abs_stepsize=epsilon_step)
    else:
        raise ValueError(f"Norm {norm} not supported")
    
    # Active mask: True = sample is still correctly classified (needs attack)
    active_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=device)

    for i in range(iterations):
        # 1. Check which samples are finished
        # We perform a forward pass to check current accuracy
        with torch.no_grad():
            preds = model(adv_inputs).argmax(dim=1)
            # Still correct means we must continue attacking
            current_correct = (preds == labels)
            
            # Update active mask: Stay active only if CURRENTLY correct AND PREVIOUSLY active
            active_mask = active_mask & current_correct
        
        if not active_mask.any():
            break

        # 2. Extract active samples
        # We need to detach to stop the computational graph from growing infinitely across loops
        current_active_inputs = adv_inputs[active_mask].detach()
        current_active_labels = labels[active_mask]
        
        # 3. Perform 1 Step using Foolbox
        # We pass the *current* adversarial images as inputs.
        # We pass epsilons=inf so Foolbox applies the step + clip(0,1) but DOES NOT 
        # project back to the original image's epsilon ball.
        # This matches your "total epsilon is irrelevant" requirement.
        _, step_adv, _ = attack(
            fmodel, 
            current_active_inputs, 
            fb.criteria.Misclassification(current_active_labels), 
            epsilons=epsilon_step 
        )
        
        # 4. Update the master batch
        adv_inputs[active_mask] = step_adv

    # Final check
    with torch.no_grad():
        final_preds = model(adv_inputs).argmax(dim=1)
        success = (final_preds != labels)

    return adv_inputs, success

def second_attack_batch(fmodel, inputs, labels, number_iterations, norm):
    """
    Uses Foolbox to run EAD, HopSkipJump, or CWL2 on GPU.
    """
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Foolbox Criterion: Misclassification
    criterion = fb.criteria.Misclassification(labels)

    if norm == 2:
        # Carlini Wagner L2
        attack = fb.attacks.L2CarliniWagnerAttack(steps=number_iterations)
        # CW in foolbox returns: raw_advs, clipped_advs, success
        # We generally want the clipped ones for evaluation
        # Note: CW is computationally expensive.
        raw_advs, clipped_advs, success = attack(fmodel, inputs, criterion, epsilons=None)
        return clipped_advs, success

    elif norm == 1:
        # EAD (ElasticNet)
        attack = fb.attacks.EADAttack(steps=number_iterations)
        raw_advs, clipped_advs, success = attack(fmodel, inputs, criterion, epsilons=None)
        return clipped_advs, success

    elif norm == np.inf:
        # HopSkipJump
        # HSJ in Foolbox is quite efficient
        attack = fb.attacks.HopSkipJumpAttack(steps=number_iterations)
        raw_advs, clipped_advs, success = attack(fmodel, inputs, criterion, epsilons=None)
        return clipped_advs, success
        
    else:
        print(f"Norm {norm} not supported in second attack.")
        return inputs, torch.zeros_like(labels, dtype=torch.bool)

def adv_distance(testloader, model, iterations_pgd, iterations_second_attack, eps_iter, norm, setsize):
    if len(eps_iter) != len(norm):
        print('!!! Please provide an eps_iter value for every norm')
        
    distances_array = np.zeros([setsize, len(norm)*3])
    mean_distances_array = np.zeros([len(norm)*2])

    model.eval()
    
    # Create Foolbox Model Wrapper (handles preprocessing if bounds are 0-1)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    
    total_samples = 0
    
    for id_norm, (n, eps_i) in enumerate(zip(norm, eps_iter)):
        print(f'Adversarial distance calculation for {n} norm (PGD + norm-specific second attack)')
        
        sample_idx = 0
        correct, total = 0, 0
        
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            inputs = torch.clamp(inputs, 0.0, 1.0)
            
            # 1. Clean Accuracy
            with torch.no_grad():
                clean_acc = fb.utils.accuracy(fmodel, inputs, labels)
                # Foolbox accuracy returns mean scalar. We need per-sample check.
                clean_preds = fmodel(inputs).argmax(dim=1)
                
            correct_mask = (clean_preds == labels)
            
            # Prepare storage for this batch
            batch_min_dists = torch.zeros(batch_size, device=device)
            batch_pgd_dists = torch.zeros(batch_size, device=device)
            batch_sec_dists = torch.zeros(batch_size, device=device)
            
            if correct_mask.any():
                attack_inputs = inputs[correct_mask]
                attack_labels = labels[correct_mask]
                
                # --- Attack 1: PGD (Custom GPU) ---
                adv_pgd, flipped_pgd = batch_pgd_with_early_stopping(
                    model, attack_inputs, attack_labels, 
                    iterations=iterations_pgd, epsilon_step=eps_i, norm=n
                )
                
                # --- Attack 2: Foolbox (EAD/HSJ/CW) ---
                # Foolbox handles batches natively
                adv_sec, flipped_sec = second_attack_batch(
                    fmodel, attack_inputs, attack_labels, 
                    number_iterations=iterations_second_attack, norm=n
                )
                
                # --- Calculate Distances ---
                # PGD
                diff_pgd = (attack_inputs - adv_pgd).view(attack_inputs.size(0), -1)
                dist_pgd = torch.norm(diff_pgd, p=(float('inf') if n == np.inf else n), dim=1)
                
                # Second Attack
                diff_sec = (attack_inputs - adv_sec).view(attack_inputs.size(0), -1)
                dist_sec = torch.norm(diff_sec, p=(float('inf') if n == np.inf else n), dim=1)
                
                # --- Logic to pick "Best" Distance ---
                # Initialize with large value
                final_dists = torch.full_like(dist_pgd, float('inf'))
                
                # If PGD worked, take its distance
                final_dists[flipped_pgd] = torch.min(final_dists[flipped_pgd], dist_pgd[flipped_pgd])
                
                # If Second worked, take min(current, second)
                final_dists[flipped_sec] = torch.min(final_dists[flipped_sec], dist_sec[flipped_sec])
                
                # Fallback: If neither worked, take min of the failures (closest we got)
                neither = ~(flipped_pgd | flipped_sec)
                if neither.any():
                    final_dists[neither] = torch.min(dist_pgd[neither], dist_sec[neither])

                # Accuracy update: Sample is correct if originally correct AND not flipped by any attack
                batch_correct_count = neither.sum().item()
                correct += batch_correct_count
                
                # Map back to full batch size
                batch_pgd_dists[correct_mask] = dist_pgd
                batch_sec_dists[correct_mask] = dist_sec
                batch_min_dists[correct_mask] = final_dists
            
            # --- Store in Numpy Array ---
            # Indices in the global array
            start = sample_idx
            end = sample_idx + batch_size
            
            # Column 0: Min Dist
            distances_array[start:end, id_norm*3] = batch_min_dists.cpu().numpy()
            # Column 1: PGD Dist
            distances_array[start:end, id_norm*3+1] = batch_pgd_dists.cpu().numpy()
            # Column 2: Second Dist
            distances_array[start:end, id_norm*3+2] = batch_sec_dists.cpu().numpy()
            
            total += batch_size
            sample_idx += batch_size
            
            if (i+1) % 5 == 0:
                print(f"{(i+1)*batch_size} images done | Mean Min Dist: {np.mean(distances_array[:sample_idx, id_norm*3]):.4f} | Adv Acc: {correct/total:.2%}")

        # Summary for this norm
        col_idx = id_norm * 3
        all_dists = distances_array[:setsize, col_idx]
        non_zero = all_dists[all_dists != 0]
        mean_distances_array[id_norm*2] = np.mean(all_dists)
        mean_distances_array[id_norm*2+1] = np.mean(non_zero) if len(non_zero) > 0 else 0.0

    adv_acc = correct / total
    return distances_array, mean_distances_array, adv_acc

def compute_adv_distance(testset, workers, model, adv_distance_params, num_classes):

    adv_distance_params["setsize"] = min(adv_distance_params["setsize"], len(testset))
    
    truncated_testset, _ = torch.utils.data.random_split(
        testset,
        [adv_distance_params["setsize"], len(testset)-adv_distance_params["setsize"]],
        generator=torch.Generator().manual_seed(42)
    )
    
    gpu_batch_size = adv_distance_params.get("batch_size", min(20, adv_distance_params["setsize"]))
    
    truncated_testloader = DataLoader(
        truncated_testset, 
        batch_size=gpu_batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=workers
    )

    model = model.to(device).float().eval()
    
    # Ensure norms are floats/standard types
    adv_distance_params["norm"] = [float(e) if isinstance(e, str) else e for e in adv_distance_params["norm"]]

    distances_array, mean_distances_array, adv_acc = adv_distance(
        testloader=truncated_testloader,
        model=model, 
        iterations_pgd=adv_distance_params["iters_pgd"],
        iterations_second_attack=adv_distance_params["iters_second_attack"], 
        norm=adv_distance_params["norm"],
        eps_iter=adv_distance_params["eps_iter"], 
        setsize=adv_distance_params["setsize"]
    )
    
    # CLEVER logic (kept separate as it relies on ART usually, or skipped)
    # If you need CLEVER, you must re-instantiate ART classifier here, 
    # but based on your request to optimize the attacks, we focus on the loop above.
    if adv_distance_params.get('clever', False):
        print("Calculating CLEVER (Warning: This might be slow if not optimized separately)...")
        # Re-import and setup ART only if needed
        from art.estimators.classification import PyTorchClassifier
        images_dummy, _ = next(iter(truncated_testloader))
        art_model = PyTorchClassifier(
            model=model, loss=nn.CrossEntropyLoss(),
            input_shape=images_dummy[0].shape, nb_classes=num_classes,
            clip_values=(0,1)
        )
        eps = []
        for id, n in enumerate(adv_distance_params["norm"]):
            eps.append(np.max(distances_array[:, id * 3]))
            
        clever_array, mean_clever_array = clever_score(
            testloader=truncated_testloader, 
            evalmodel=art_model, 
            clever_batches=adv_distance_params["clever_batches"], 
            clever_samples=adv_distance_params["clever_samples"],
            epsilon=eps, 
            norm=adv_distance_params["norm"], 
            setsize=adv_distance_params["setsize"]
        )
    else:
        mean_clever_array = np.zeros([len(adv_distance_params["clever_batches"]) * len(adv_distance_params["norm"])])
        clever_array = np.zeros((distances_array.shape[0], 0))

    # Sort Results (Dataframes expect sorted input)
    for id, n in enumerate(adv_distance_params["norm"]):
        sorted_indices = np.argsort(distances_array[:, id * 3])
        start = id * 3
        end = (id+1) * 3
        distances_array[:, start:end] = distances_array[:, start:end][sorted_indices]
        
        if adv_distance_params.get('clever', False):
            c_len = len(adv_distance_params["clever_batches"])
            c_start = id * c_len
            c_end = (id+1) * c_len
            clever_array[:, c_start:c_end] = clever_array[:, c_start:c_end][sorted_indices]

    distances = np.concatenate((distances_array, clever_array), axis=1)
    mean_distances = np.concatenate((mean_distances_array, mean_clever_array))

    return adv_acc*100, distances, mean_distances

def compute_adv_distance_depr(testset, workers, model, adv_distance_params, num_classes):

    print(f"Adversarial Distance upper bound calculation using lowest of PGD and a norm-specific second attack")
    truncated_testset, _ = torch.utils.data.random_split(testset,
                                                         [adv_distance_params["setsize"], len(testset)-adv_distance_params["setsize"]],
                                                         generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=1, shuffle=False,
                                       pin_memory=True, num_workers=workers)
    class Float32ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Force input to float32 before passing to the real model
            return self.model(x.float())
        
    images, _ = next(iter(truncated_testloader))
    evalmodel = PyTorchClassifier(model=Float32ModelWrapper(model),
                            loss=torch.nn.CrossEntropyLoss(),
                            optimizer=torch.optim.SGD(model.parameters(), momentum= 0.9, weight_decay= 1e-4, lr=0.01),
                            input_shape=images[0].size(),
                            nb_classes=num_classes)

    adv_distance_params["norm"] = [float(e) if isinstance(e, str) else e for e in adv_distance_params["norm"]]

    distances_array, mean_distances_array, adv_acc = adv_distance(testloader=truncated_testloader,
                                    model=model, evalmodel=evalmodel, iterations_pgd=adv_distance_params["iters_pgd"],
                                    iterations_second_attack=adv_distance_params["iters_second_attack"], norm=adv_distance_params["norm"],
                                    eps_iter=adv_distance_params["eps_iter"], setsize=adv_distance_params["setsize"])

    if adv_distance_params['clever'] == True:
        eps = []
        for id, n in enumerate(adv_distance_params["norm"]):
            eps.append(np.max(distances_array[:, id * 3]))

        print(f"Adversarial Distance (statistical) lower bound calculation using Clever Score with epsilon = largest "
              f"adversarial attack distance, batches: "
              f"{adv_distance_params['clever_batches']}, samples per batch: {adv_distance_params['clever_samples']}.")
        clever_array, mean_clever_array = clever_score(testloader=truncated_testloader, evalmodel=evalmodel, clever_batches=
                            adv_distance_params["clever_batches"], clever_samples=adv_distance_params["clever_samples"],
                            epsilon=eps, norm=adv_distance_params["norm"], setsize=adv_distance_params["setsize"])
    else:
        mean_clever_array = np.zeros([len(adv_distance_params["clever_batches"]) * len(adv_distance_params["norm"])])
        clever_array = np.empty((distances_array.shape[0], 0))
    for id, n in enumerate(adv_distance_params["norm"]):
        sorted_indices = np.argsort(distances_array[:, id * 3])
        distances_array[:,id * 3:(id+1)*3] = distances_array[:,id * 3:(id+1)*3][sorted_indices[:, np.newaxis], np.arange(distances_array[:,id * 3:(id+1)*3].shape[1])]
        if adv_distance_params['clever'] == True:
            clever_array[:,id * len(adv_distance_params["clever_batches"]):(id+1)*len(adv_distance_params["clever_batches"])] = \
                clever_array[:,id * len(adv_distance_params["clever_batches"]):(id+1)*len(adv_distance_params["clever_batches"])][sorted_indices[:, np.newaxis],
                        np.arange(clever_array[:,id * len(adv_distance_params["clever_batches"]):(id+1)*len(adv_distance_params["clever_batches"])].shape[1])]
    print(f'Mean CLEVER scores: {mean_clever_array}')

    distances = np.concatenate((distances_array, clever_array), axis=1)
    mean_distances = np.concatenate((mean_distances_array, mean_clever_array))

    return adv_acc*100, distances, mean_distances

def adv_distance_depr(testloader, model, evalmodel, iterations_pgd, iterations_second_attack, eps_iter, norm, setsize):
    if len(eps_iter) != len(norm):
        print('!!! Please provide an eps_iter value for every norm')
    distances_array = np.empty([setsize, len(norm)*3])
    mean_distances_array = np.empty([len(norm)*2])

    for id, (n, eps_i) in enumerate(zip(norm, eps_iter)):
        print(f'Adversarial distance calculation for {n} norm')

        correct, total = 0, 0

        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            if predicted == labels:
                helper_array = np.empty([3, 2], dtype=object)
                adv_inputs1, label_flipped1 = pgd_with_early_stopping(evalmodel, inputs, labels, predicted, iterations_pgd, eps_i, n)
                helper_array[:,0] = [label_flipped1, adv_inputs1.cpu().numpy(), torch.norm((inputs - adv_inputs1), p=n).cpu().numpy()]
                adv_inputs2, label_flipped2 = second_attack(evalmodel, inputs, labels, predicted, iterations_second_attack, n)
                helper_array[:,1] = [label_flipped2, adv_inputs2.cpu().numpy(), torch.norm((inputs - adv_inputs2), p=n).cpu().numpy()]

                if np.any(helper_array[0]):
                    # At least one success: pick the best (minimum distance) successful attack
                    selected_indices = np.where(helper_array[0])[0]
                    flipped = helper_array[:, selected_indices]
                    
                    # Safely get the minimum of the successful ones
                    best_idx = np.argmin(flipped[2])
                    min_dist = flipped[2][best_idx]
                    min_adv_example = flipped[1][best_idx]
                else:
                    # No attack succeeded: just take the one that got "closest" (smallest norm)
                    # even though the label didn't change.
                    best_idx = np.argmin(helper_array[2])
                    min_dist = helper_array[2][best_idx]
                    min_adv_example = helper_array[1][best_idx]

                if not np.any(helper_array[0]):
                    best_idx = np.argmin(helper_array[2]) 
                    min_dist = helper_array[2][best_idx]
                    min_adv_example = helper_array[1][best_idx]

                distances_array[i, id*3] = min_dist
                distances_array[i, id*3+1:id*3+3] = helper_array[2,0:2]

                _, adv_predicted = torch.max(model(torch.tensor(min_adv_example, device='cuda')).data, 1)

            else:
                distances_array[i, id*3:id*3+3] = np.array([0.0, 0.0, 0.0])
                adv_predicted = predicted
            correct += ((adv_predicted) == labels).sum().item()
            total += labels.size(0)
            if (i+1) % 10 == 0:
                print(f"Completed: {i+1} of {setsize}, mean_distances: {np.mean(distances_array[:(i+1), id*3])}, "
                      f"{np.mean(distances_array[:(i+1), id*3][distances_array[:(i+1), id*3] != 0.0])}, correct: "
                      f"{correct}, total: {total}, accuracy: {correct / total * 100}%")
        mean_distances_array[id*2] = np.mean(distances_array[:, id*3])
        mean_distances_array[id * 2 +1] = np.mean(distances_array[:, id*3][distances_array[:, id*3] != 0.0])

    adv_acc = correct / total
    return distances_array, mean_distances_array, adv_acc

def pgd_with_early_stopping(model, inputs, labels, clean_predicted, number_iterations, epsilon_iters, norm):

    attacker = ProjectedGradientDescentPyTorch(estimator=model, norm=norm, eps=epsilon_iters * number_iterations,
                                               eps_step=epsilon_iters, max_iter=1, verbose=False)
    inputs = np.asarray(inputs.cpu())
    labels = np.asarray(labels.cpu())

    for i in range(number_iterations):

        adv_inputs = attacker.generate(inputs, labels)
        adv_outputs = model.predict(adv_inputs)
        adv_predicted = np.array([np.argmax(adv_outputs)])

        label_flipped = bool(adv_predicted!=clean_predicted.cpu().numpy())
        if label_flipped:
            break
        inputs = adv_inputs.copy()

    return torch.Tensor(adv_inputs).to('cuda'), label_flipped

def second_attack(model, inputs, labels, clean_predicted, number_iterations, norm):
    if norm == 2:
        attacker = CarliniL2Method(model,
                               max_iter=number_iterations,
                                   verbose=False)
    elif norm == 1:
        attacker = ElasticNet(model,
                      max_iter=number_iterations,
                                   verbose=False)
    elif norm == np.inf:
        attacker = HopSkipJump(model,
                         norm=norm,
                         max_iter=number_iterations,
                                   verbose=False)
    else:
        print(f'Norm {norm} not within 1, 2, or np.inf.')
        return inputs, labels

    inputs = np.asarray(inputs.cpu())
    labels = np.asarray(labels.cpu())
    adv_inputs = attacker.generate(inputs, labels)
    adv_outputs = model.predict(adv_inputs)
    adv_inputs = torch.tensor(adv_inputs, device='cuda')
    adv_outputs = torch.tensor(adv_outputs, device='cuda')
    _, adv_predicted = torch.max(adv_outputs.data, 1)
    label_flipped = True if adv_predicted != clean_predicted else False

    return adv_inputs, label_flipped