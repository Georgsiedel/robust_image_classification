'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.models import ct_model
import experiments.models.ImageNet.resnet as in_resnet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation_function=F.relu):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation_function(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_function(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_function=F.relu):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation_function(self.bn1(self.conv1(x)))
        out = self.activation_function(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation_function(out)
        return out


class ResNet(ct_model.CtModel):
    activation_function: object

    def __init__(self, block, num_blocks, dataset, normalized, num_classes=10, factor=1, activation_function='relu'):
        super(ResNet, self).__init__(dataset=dataset, normalized=normalized, num_classes=num_classes)
        self.in_planes = 64
        self.activation_function = getattr(F, activation_function)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=factor, activation_function=self.activation_function)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation_function=self.activation_function)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation_function=self.activation_function)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation_function=self.activation_function)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.blocks = [nn.Sequential(self.conv1, self.bn1), self.layer1, self.layer2, self.layer3]

    def _make_layer(self, block, planes, num_blocks, stride, activation_function):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation_function))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, robust_samples=0, corruptions=None, mixup_alpha=0.0, mixup_p=0.0, manifold=False,
                manifold_noise_factor=1, cutmix_alpha=0.0, cutmix_p=0.0, noise_minibatchsize=1,
                concurrent_combinations=1, noise_sparsity=0.0, noise_patch_lower_scale=0.3, noise_patch_upper_scale=1.0,
                generated_ratio=0.0, n2n_deepaugment=False, style_feats=None, **kwargs):

        x = super(ResNet, self).forward_handle_greyscale(x)
        out = super(ResNet, self).forward_normalize(x)
        out, mixed_targets = super(ResNet, self).forward_noise_mixup(out, targets, robust_samples, corruptions,
                                        mixup_alpha, mixup_p, manifold, manifold_noise_factor, cutmix_alpha, cutmix_p,
                                        noise_minibatchsize, concurrent_combinations, noise_sparsity,
                                        noise_patch_lower_scale, noise_patch_upper_scale, generated_ratio, n2n_deepaugment, 
                                        style_feats, **kwargs)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training == True:
            return out, mixed_targets
        else:
            return out


def ResNet18(num_classes, dataset, normalized, factor, activation_function='relu'):
    if dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 'SynthiCAD',
                       'Describable-Textures', 'Flickr-Material', 'KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
        return in_resnet.resnet18(dataset=dataset, normalized=normalized, activation_function=activation_function, num_classes=num_classes)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dataset=dataset, normalized=normalized, factor=factor,
                  activation_function=activation_function)
    
def PreActResNet18(num_classes, dataset, normalized, factor, activation_function='relu'):
        if dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 'SynthiCAD',
                       'Describable-Textures', 'Flickr-Material', 'KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
            return in_resnet.resnet18(weights=in_resnet.ResNet18_Weights.IMAGENET1K_V1, dataset=dataset, normalized=normalized, activation_function=activation_function, num_classes=num_classes)
        else:
            print('not yet implemented')

def ResNet34(num_classes, dataset, normalized, factor, activation_function='relu'):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dataset=dataset, normalized=normalized, factor=factor,
                  activation_function=activation_function)

def ResNet50(num_classes, dataset, normalized, factor, activation_function='relu'):
    if dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 'SynthiCAD', 
                       'Describable-Textures', 'Flickr-Material', 'KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
        return in_resnet.resnet50(dataset=dataset, normalized=normalized, activation_function=activation_function, num_classes=num_classes)
    else:
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dataset=dataset, normalized=normalized, factor=factor,
                  activation_function=activation_function)
    
def PreActResNet50(num_classes, dataset, normalized, factor, activation_function='relu'):
        if dataset in ['ImageNet', 'ImageNet-100', 'TreeSAT', 'Casting-Product-Quality', 'SynthiCAD', 
                       'Describable-Textures', 'Flickr-Material', 'KITTI_RoadLane', 'KITTI_Distance_Multiclass']:
            return in_resnet.resnet50(weights=in_resnet.ResNet50_Weights.IMAGENET1K_V1, dataset=dataset, normalized=normalized, activation_function=activation_function, num_classes=num_classes)
        else:
            print('not yet implemented')

def ResNet101(num_classes, dataset, normalized, factor, activation_function='relu'):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, dataset=dataset, normalized=normalized, factor=factor,
                  activation_function=activation_function)

def ResNet152(num_classes, dataset, normalized, factor, activation_function='relu'):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, dataset=dataset, normalized=normalized, factor=factor,
                  activation_function=activation_function)