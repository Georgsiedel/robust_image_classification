import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def get_criterion(loss_function, lossparams):
    if loss_function == 'jsd':
        criterion, robust_samples = JsdCrossEntropy(**lossparams), lossparams["num_splits"] - 1
    elif loss_function == 'trades':
        criterion, robust_samples = Trades(**lossparams), 0
    elif loss_function == 'bce':
        criterion, robust_samples = torch.nn.BCELoss(**lossparams), 0
    else:
        criterion, robust_samples = torch.nn.CrossEntropyLoss(label_smoothing=lossparams["smoothing"]), 0
    test_criterion = torch.nn.CrossEntropyLoss()
    return criterion, test_criterion, robust_samples

class Criterion(nn.Module):
    def __init__(self, standard_loss, trades_loss=False, fraction=1.0, beta=1.0, step_size=0.003, epsilon=0.031, random_eps=False,
                 perturb_steps=10, distance='l_inf', robust_loss=False, alpha=12, num_splits=3, **kwargs):
        super().__init__()
        loss = getattr(torch.nn, standard_loss)
        self.standard_criterion = loss(**kwargs)
        self.robust_samples = num_splits - 1 if robust_loss == True else 0
        self.trades_loss = trades_loss
        self.robust_loss = robust_loss
        if trades_loss == True:
            self.step_size=step_size
            self.epsilon=epsilon
            self.perturb_steps=perturb_steps
            self.beta=beta
            self.distance=distance
            self.trades_fraction=fraction
            self.random_eps = random_eps
            assert 0.0 < self.trades_fraction <= 1.0, "TRADES fraction must be over 0.0 and up to including 1.0"
        if robust_loss == True:
            self.num_splits=num_splits
            self.alpha=alpha

    def calculate_standard_and_robust_loss(self, outputs, mixed_targets):
        split_size = outputs.shape[0] // (self.robust_samples+1)
        loss = self.standard_criterion(outputs[:split_size], mixed_targets)
        
        if self.robust_loss:
            loss += jsd_loss(outputs, self.num_splits, self.alpha)

        return loss
    
    def add_trades_loss(self, 
                        loss, 
                        model, 
                        optimizer, 
                        inputs, 
                        targets, 
                        generated_ratio: float):
        
        if self.trades_loss:
            # Determine number per original and generated samples, according to ratio
            # assumes every batch layers original images first, then generated images, as done in data loader
            split_size = inputs.shape[0] // (self.robust_samples+1)
            n_generated = int(split_size * generated_ratio)
            n_original = split_size - n_generated
            n_trades_total = int(self.trades_fraction * split_size)
            n_trades_original = int(self.trades_fraction * n_original)
            start_index = n_original - n_trades_original

            trades_loss = trades(model=model, 
                                      x_natural=inputs[start_index:start_index + n_trades_total], 
                                      y=targets[start_index:start_index + n_trades_total], 
                                      optimizer=optimizer, 
                                      step_size=self.step_size,                                 
                                      epsilon=self.epsilon,                                   
                                      perturb_steps=self.perturb_steps,                                    
                                      beta=self.beta * self.trades_fraction, #scales aggressiveness according to fraction                                
                                      distance=self.distance,
                                      random_eps=self.random_eps)
            loss += trades_loss

        return loss

    def test(self, outputs, mixed_targets):
        loss = self.standard_criterion(outputs, mixed_targets)
        return loss

def jsd_loss(output, num_splits=3, alpha=12):
    """ Jensen-Shannon Divergence loss (no cross entropy)

    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781

    Hacked together by / Copyright 2020 Ross Wightman
    """
    split_size = output.shape[0] // num_splits
    assert split_size * num_splits == output.shape[0]
    logits_split = torch.split(output, split_size)

    # Cross-entropy is only computed on clean images
    probs = [F.softmax(logits, dim=1) for logits in logits_split]

    # Clamp mixture distribution to avoid exploding KL divergence
    logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
    loss = alpha * sum([F.kl_div(
        logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / num_splits
    return loss

def trades(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                random_eps=False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    device = x_natural.device

    if random_eps:
        # u in [0,1], reused for eps and step size
        u = torch.rand(batch_size, device=device)

        eps = u * epsilon
        eps_view = eps.view(batch_size, 1, 1, 1)

        step_sizes = u * step_size
        step_sizes_view = step_sizes.view(batch_size, 1, 1, 1)
    else:
        eps = torch.full((batch_size,), epsilon, device=device)
        eps_view = eps.view(batch_size, 1, 1, 1)

        # standard fixed step size
        step_sizes = torch.full((batch_size,), step_size, device=device)
        step_sizes_view = step_sizes.view(batch_size, 1, 1, 1)
    
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_sizes_view * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps_view), x_natural + eps_view)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection with per-sample eps ball
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            
            delta_flat = delta.data.view(batch_size, -1)
            delta_norms = delta_flat.norm(p=2, dim=1)

            scale = torch.clamp(eps / (delta_norms + 1e-12), max=1.0)
            delta.data = delta.data * scale.view(batch_size, 1, 1, 1)

        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    # natural loss is calculated in the training loop already, we leave it out here
    #logits = model(x_natural)[0] #changed for ct_model outputs a tuple with mixed_targets
    #loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv)[0], dim=1),
                                                    F.softmax(model(x_natural)[0], dim=1))
    loss = beta * loss_robust # + loss_natural
    return loss