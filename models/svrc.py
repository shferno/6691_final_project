
import torch
from torch import nn
from torchvision.models import densenet169, resnet50, resnet18, densenet121


baseline_models = {
    'densenet169' : densenet169(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    'resnet50': resnet50(pretrained=True),
    'densenet121': densenet121(pretrained=True)
}

# svrc with rsd
class SVRC(nn.Module):
    '''
    SVRC model for training reduction hernia dataset
    '''
    def __init__(self, baseline, n_phases = 14, n_rsd = 1):
        super().__init__()
        assert baseline in baseline_models.keys(), \
            'Unknow baseline model, use on of: {}'.format(baseline_models.keys())
        self.cnn = baseline_models[baseline]
        self.pretrain = True
        self.lstm_states = None
        if baseline == 'densenet169':
            self.feature_size = 1664
            tmp_conv_weights = self.cnn.features.conv0.weight.data[:, :3, :, :].clone()
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
            tmp_conv_weights = self.cnn.features.conv0.weight.data[:, :3, :, :].clone()
            self.cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights
            self.cnn.classifier = nn.Identity()
        elif baseline == 'resnet50':
            self.feature_size = 2048
            tmp_conv_weights = self.cnn.conv1.weight.data[:, :3, :, :].clone()
            self.cnn.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights =  torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
            self.cnn.conv1.weight.data[:, 3, :, :] = mean_weights
            self.cnn.fc = nn.Identity()
        else:
            self.feature_size = 512
            tmp_conv_weights = self.cnn.conv1.weight.data[:, :3, :, :].clone()
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
        # if pretrain
        self.linear = nn.Linear(self.feature_size, hidden_size)
        # full connect layer
        self.fc1 = nn.Linear(hidden_size, n_phases)
        self.fc2 = nn.Linear(hidden_size, n_rsd)
    
    def freeze_cnn(self, freeze = False):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze = False):
        for param in self.rnn.parameters():
            param.requires_grad = not freeze
    
    def forward(self, X):
        x = self.cnn(X.float())
        # Reshape
        x = x.squeeze() # batch size x channels
        if self.pretrain: # train all
            x = self.linear(x)
        else:
            x,s = self.rnn(x.unsqueeze(0), self.lstm_states) # 1 x time steps x n_channels
            x = x.squeeze() # batch size x lstm_output_size
            # save lstm states
            self.lstm_states = (s[0].detach(), s[1].detach())
        x = x.squeeze(0)
        phase_pred = self.fc1(x.clone())
        rsd_pred = self.fc2(x.clone())
        return phase_pred, rsd_pred

'''
    def predict(self, X, y, BATCH_SIZE, transform, device):
        self.eval()
        dataset = SVRCDataset(X, y, transform)
        loader = DataLoader(
            dataset, batch_sampler=BatchSampler(
                SequentialSampler(dataset), 
                BATCH_SIZE, 
                drop_last=True
            )
        )

        test_acc = 0.0
        predicts = []
        for i, data in enumerate(loader):
            features = data['feature'].float()
            labels = data['label']
            features,labels = features.to(device), labels.to(device)
            predictions = self.forward(features)
            preds = torch.max(predictions.data, 1)[1]
            predicts.append(preds)
            if labels != None:
                test_acc += (preds == labels).sum().item()
        if labels != None:
            test_acc /= len(dataset)
            print(f'test_acc:{test_acc}')
        return predicts
'''

