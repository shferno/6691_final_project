import torch
from torch import nn
from typing import TypeVar
from torchvision.models import densenet169
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import densenet121




# model choice
baseline_models = {
    'densenet169' : densenet169(pretrained = True),
    'resnet50': resnet50(pretrained = True),
    'resnet18': resnet18(pretrained = True),
    'densenet121': densenet121(pretrained = True),
}


class CNN_model(nn.Module):
    '''
    CNN model for pretraining CNN models in the first 3 epochs
    '''
    def __init__(self, baseline, n_step_classes = 11, n_expertise_classes = 2):
        super().__init__()
        self.cnn = baseline_models[baseline]
        if baseline == 'densenet169':
            self.feature_size = 1664
            tmp_conv_weights = self.cnn.features.conv0.weight.data.clone()
            self.cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # copy RGB weights pre-trained on ImageNet
            self.cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            # compute average weights over RGB channels and use that to initialize the optical flow channels
            # technique from Temporal Segment Network, L. Wang 2016
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights
            self.cnn.classifier = nn.Identity()
        elif baseline == 'densenet121':
            self.feature_size = 1024
            tmp_conv_weights = self.cnn.features.conv0.weight.data.clone()
            self.cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights
            self.cnn.classifier = nn.Identity()
        elif baseline == 'resnet50':
            self.feature_size = 2048
            tmp_conv_weights = self.cnn.conv1.weight.data.clone()
            self.cnn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights =  torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.conv1.weight.data[:, 3, :, :] = mean_weights
            self.cnn.fc = nn.Identity()
        else:
            self.feature_size = 512
            tmp_conv_weights = self.cnn.conv1.weight.data.clone()
            self.cnn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.conv1.weight.data[:, 3, :, :] = mean_weights
            self.cnn.fc = nn.Identity()
        self.step_fc = nn.Linear(self.feature_size, n_step_classes)
        self.expertise_fc = nn.Linear(self.feature_size, n_expertise_classes)
    def forward(self,X):
        X = self.cnn(X)
        y1 = self.step_fc(X)
        y2 = self.expertise_fc(X)
        return y1, y2

