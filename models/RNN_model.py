import torch
from torch import nn
from typing import TypeVar
from torchvision.models import densenet169
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import densenet121

baseline_models = {
    'densenet169' : densenet169(pretrained = True),
    'resnet50': resnet50(pretrained = True),
    'resnet18': resnet18(pretrained = True),
    'densenet121': densenet121(pretrained = True),
}

class RNN_model(nn.Module):
    '''
    CataNet model: CNN + LSTM
    '''
    def __init__(self, baseline, n_step_classes = 11, n_expertise_classes = 2, n_rsd = 1):
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
        self.rnn = nn.LSTM(input_size= self.feature_size,
                           hidden_size=128,
                           num_layers=2,
                           dropout=0.0,
                           batch_first=True)
        if isinstance(self.rnn.hidden_size, list):
            hidden_size = self.rnn.hidden_size[-1]
        else:
            hidden_size = self.rnn.hidden_size
        self.last_state = None
        self.fc1 = nn.Linear(hidden_size, n_step_classes)
        self.fc2 = nn.Linear(hidden_size, n_expertise_classes)
        self.fc3 = nn.Linear(hidden_size, n_rsd)


    def freeze_cnn(self, freeze = False):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze = False):
        for param in self.rnn.parameters():
            param.requires_grad = not freeze

    # def forward(self, X):
    #     X = self.cnn(X)
    #     X = self.rnn(X)
    #     step = self.fc1(X.clone())
    #     experience = self.fc2(X.clone())
    #     rsd = self.fc3(X.clone())
    #     return step, experience, rsd

    def forward(self, X, stateful = False, skip_features = False):
        if skip_features:
            features = X
        else:
            features = self.cnn(X)
            features = features.unsqueeze(0) # go from [batch size, C] to [batch size, sequence length, C]
        if stateful:
            init_state = self.last_state
        else:
            init_state = None
        y, last_state = self.rnn(features, init_state)# y: [batch_size, len sequence, out_feature_size]
        self.last_state = (last_state[0].detach(), last_state[1].detach()) # we need to break the graph here
        y = y.squeeze(0)
        step_prediction = self.fc1(y.clone())
        exp_prediction = self.fc2(y.clone())
        rsd_prediction = self.fc3(y.clone())
        return step_prediction, exp_prediction, rsd_prediction


