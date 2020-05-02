import sys
sys.path.append('E:\scriptieAI\code\sepconvfull')

import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sepconv
from torch.nn import functional as F
import numpy as np



def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleVertical1 = Subnet(self.kernel_size)
        self.moduleVertical2 = Subnet(self.kernel_size)
        self.moduleHorizontal1 = Subnet(self.kernel_size)
        self.moduleHorizontal2 = Subnet(self.kernel_size)

    def forward(self, rfields):
        B, F, C, H, W = rfields.shape
        
        # A
        # tensorJoin = rfields.permute(0,3,4,2,1).contiguous().view(B, H, W, F*C).permute(0,3,1,2)
        
        
        # B
        # tensorJoin = rfields.permute(0,3,4,1,2).contiguous().view(B, H, W, F*C).permute(0,3,1,2)
        
        # C
        tensorJoin = rfields.view(B, F*C, H, W)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        tensorCombine = tensorUpsample2 + tensorConv2
        
        Vertical1 = self.moduleVertical1(tensorCombine)
        Vertical2 = self.moduleVertical2(tensorCombine)
        Horizontal1 = self.moduleHorizontal1(tensorCombine)
        Horizontal2 = self.moduleHorizontal2(tensorCombine)

        return Vertical1, Horizontal1, Vertical2, Horizontal2


class SepConvNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SepConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_pad = int(math.floor(kernel_size / 2.0))

        # self.epoch = Variable(torch.tensor(0, requires_grad=False))
        self.epoch = 0
        self.get_kernel = KernelEstimation(self.kernel_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frames):

        _, f, _, h, w = frames.shape

        h_padded = False
        w_padded = False
        padded_frames = frames.clone()

        if h % 32 != 0:
            pad_h = 32 - (h % 32)
            padded_frames = F.pad(padded_frames, (0, 0, 0, pad_h))
            h_padded = True


        if w % 32 != 0:
            pad_w = 32 - (w % 32)
            padded_frames = F.pad(padded_frames, (0, pad_w, 0, 0))
            w_padded = True


        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.get_kernel(padded_frames)

        frame_before = int(0+f/4)
        frame_after  = int(1+f/4)

        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(padded_frames[:, frame_before]), Vertical1, Horizontal1)
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(padded_frames[:, frame_after]), Vertical2, Horizontal2)

        frame1 = tensorDot1 + tensorDot2

        if h_padded:
            frame1 = frame1[:, :, 0:h, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w]

        return frame1

    def train_model(self, frame0, frame2, frame1):
        self.optimizer.zero_grad()
        output = self.forward(frame0, frame2)
        loss = self.criterion(output, frame1)
        loss.backward()
        self.optimizer.step()
        return loss

    def increase_epoch(self):
        self.epoch += 1
