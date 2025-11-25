import os
import sys
current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

from run_0 import device

########################################################################################################
### Noise2Net
########################################################################################################
# based on DeepAugment method here: https://github.com/hendrycks/imagenet-r/blob/master/DeepAugment/train_noise2net.py
# Paper Hendrycks et al: "The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization"

import sys
import os
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class N2N_DeepAugment(nn.Module):
    def __init__(self, orig_batch_size, image_size, channels, noisenet_max_eps=0.75, ratio=0.5):
        super(N2N_DeepAugment, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.ratio = ratio
        self.noise2net_batch_size = int(orig_batch_size * ratio)
        self.noise2net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=self.noise2net_batch_size).train().to(device)
        self.noisenet_max_eps = noisenet_max_eps

    def forward(self, bx):
        batchsize = bx.shape[0]

        if self.noise2net_batch_size != int(batchsize * self.ratio): #last batch of an epoch may have different bs
            self.noise2net_batch_size = int(batchsize * self.ratio)
            self.noise2net = Res2Net(epsilon=0.5, hidden_planes=16, batch_size=self.noise2net_batch_size).train().to(device)
        
        with torch.no_grad():
            # Setup network
            self.noise2net.reload_parameters()
            self.noise2net.set_epsilon(random.uniform(self.noisenet_max_eps / 2.0, self.noisenet_max_eps))
            
            # Apply aug on a random subset according to ratio
            indices = torch.randperm(batchsize)[:self.noise2net_batch_size]

            bx_auged = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)(bx[indices])
            bx_auged = bx_auged.reshape((1, self.noise2net_batch_size * self.channels, 224, 224))
            bx_auged = self.noise2net(bx_auged)
            bx_auged = bx_auged.reshape((self.noise2net_batch_size, self.channels, 224, 224))
            bx_auged = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)(bx_auged)
            bx[indices] = bx_auged

        return bx

class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, hidden_planes=9, scale = 4, batch_size=5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality (multiply by batch_size)
            planes: output channel dimensionality (multiply by batch_size)
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes * batch_size
        self.conv1 = nn.Conv2d(inplanes * batch_size, width*scale, kernel_size=1, bias=False, groups=batch_size)
        self.bn1 = nn.BatchNorm2d(width*scale)

        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3])
            D = random.choice([1, 2, 3])

            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True, groups=batch_size))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * batch_size, kernel_size=1, bias=False, groups=batch_size)
        self.bn3 = nn.BatchNorm2d(planes * batch_size)

        self.act = nn.ReLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.hidden_planes = hidden_planes
        self.batch_size = batch_size

    def forward(self, x):
        _, _, H, W = x.shape
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out) # [1, hidden_planes*batch_size*scale, H, W]
        
        # Hack to make different scales work with the hacky batches
        out = out.view(1, self.batch_size, self.scale, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)
        
        spx = torch.split(out, self.width, 1) # [ ... (1, hidden_planes*batch_size, H, W) ... ]
        
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]),1)
        
        # Undo hack to make different scales work with the hacky batches
        out = out.view(1, self.scale, self.batch_size, self.hidden_planes, H, W)
        out = torch.transpose(out, 1, 2)
        out = torch.flatten(out, start_dim=1, end_dim=3)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=16, batch_size=5):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes, batch_size=batch_size)

    def reload_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()
 
    def set_epsilon(self, new_eps):
        self.epsilon = new_eps

    def forward_original(self, x):                
        x = (self.block1(x) * self.epsilon) + x
        x = (self.block2(x) * self.epsilon) + x
        x = (self.block3(x) * self.epsilon) + x
        x = (self.block4(x) * self.epsilon) + x
        return x

    def forward(self, x):
        return self.forward_original(x)