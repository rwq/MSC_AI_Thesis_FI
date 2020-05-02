import torch
from torch import nn
import torchvision
from constants import FP_VGG

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.nnet = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.ad_avg_pool = nn.AdaptiveAvgPool2d(output_size=8)
        self.lin1 = nn.Linear(512, 32)
        self.lin2 = nn.Linear(32, 1)
    
    def forward(self, x):
        conv_out = self.nnet(x)
        conv_out = self.ad_avg_pool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)
        out = self.lin1(conv_out).relu()
        out = self.lin2(out)
        
        return out

class Discriminator(nn.Module):

    def __init__(self, input_size=2):
        super(Discriminator, self).__init__()
        self.vgg = torchvision.models.vgg16_bn(pretrained=False)

        state_dict = torch.load(FP_VGG)
        self.vgg.load_state_dict(state_dict)
        
        self.vgg.classifier[6] = nn.Linear(4096, 1)

        W = self.vgg.features[0].weight.data.clone()
        b = self.vgg.features[0].bias.data.clone()

        in_channels = (input_size+1) * 3
        n_repeats = input_size+1

        self.vgg.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride = 1, padding=1)
        self.vgg.features[0].bias.data = b
        self.vgg.features[0].weight.data = W.repeat(repeats=(1,n_repeats,1,1))

    def forward(self, X, y):
        X = torch.cat([X, y.unsqueeze(1)], dim=1)
        B, F, C, H, W = X.shape
        X = X.view(B, F*C, H, W)
        return self.vgg(X)