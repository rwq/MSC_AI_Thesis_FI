#
# KTH Royal Institute of Technology
#

import sys
sys.path.append('E:\scriptieAI\code\sepconvfull2\src')
sys.path.append('E:\scriptieAI\code\sepconvfull2')

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, gradcheck
from src.separable_convolution import SeparableConvolutionSlow
from libs.sepconv.SeparableConvolution import SeparableConvolution
import src.config as config
import src.interpolate as interpol


class Net(nn.Module):

    def __init__(self, init_weights=True):
        super(Net, self).__init__()

        conv_kernel = (3, 3)
        conv_stride = (1, 1)
        conv_padding = 1
        sep_kernel = config.OUTPUT_1D_KERNEL_SIZE

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU()

        self.conv32 = self._conv_module(6, 32, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv64 = self._conv_module(32, 64, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv128 = self._conv_module(64, 128, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv256 = self._conv_module(128, 256, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512 = self._conv_module(256, 512, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512x512 = self._conv_module(512, 512, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp512 = self._upsample_module(512, 512, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv256 = self._conv_module(512, 256, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp256 = self._upsample_module(256, 256, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv128 = self._conv_module(256, 128, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp128 = self._upsample_module(128, 128, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv64 = self._conv_module(128, 64, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp64 = self._upsample_module(64, 64, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv51_1 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv51_2 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv51_3 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv51_4 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)

        self.pad = nn.ReplicationPad2d(sep_kernel // 2)

        if torch.cuda.is_available() and not config.ALWAYS_SLOW_SEP_CONV:
            self.separable_conv = SeparableConvolution.apply
        else:
            self.separable_conv = SeparableConvolutionSlow()

        if init_weights:
            print('Initializing weights...')
            self.apply(self._weight_init)

    @staticmethod
    def from_file(file_path: str) -> nn.Module :
        """
        Initializes a new Net object from an existing model
        :param file_path: path to the model.pth file
        :return: Net object
        """
        model = Net(init_weights=False)
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)
        return model

    def to_file(self, file_path: str):
        """
        Writes the current model to disk as a .pth file
        :param file_path: path to the output model.pth file
        """
        torch.save(self.cpu().state_dict(), file_path)

    def interpolate(self, *args):
        return interpol.interpolate(self, *args)

    def interpolate_f(self, *args):
        return interpol.interpolate_f(self, *args)

    def interpolate_batch(self, *args):
        return interpol.interpolate_batch(self, *args)

    def forward(self, x):

        i1 = x[:, :3]
        i2 = x[:, 3:6]

        # ------------ Contraction ------------

        x = self.conv32(x)
        x = self.pool(x)

        x64 = self.conv64(x)
        x128 = self.pool(x64)

        x128 = self.conv128(x128)
        x256 = self.pool(x128)

        x256 = self.conv256(x256)
        x512 = self.pool(x256)

        x512 = self.conv512(x512)
        x = self.pool(x512)

        x = self.conv512x512(x)

        # ------------ Expansion ------------

        x = self.upsamp512(x)
        x += x512
        x = self.upconv256(x)

        x = self.upsamp256(x)
        x += x256
        x = self.upconv128(x)

        x = self.upsamp128(x)
        x += x128
        x = self.upconv64(x)

        x = self.upsamp64(x)
        x += x64

        # ------------ Final branches ------------

        k2h = self.upconv51_1(x)

        k2v = self.upconv51_2(x)

        k1h = self.upconv51_3(x)

        k1v = self.upconv51_4(x)

        padded_i2 = self.pad(i2)
        padded_i1 = self.pad(i1)

        # ------------ Local convolutions ------------

        return self.separable_conv(padded_i2, k2v, k2h) + self.separable_conv(padded_i1, k1v, k1h)

    @staticmethod
    def _check_gradients(func):
        print('Starting gradient check...')
        sep_kernel = config.OUTPUT_1D_KERNEL_SIZE
        inputs = (
            Variable(torch.randn(2, 3, sep_kernel, sep_kernel).cuda(), requires_grad=False),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
        )
        test = gradcheck(func, inputs, eps=1e-3, atol=1e-3, rtol=1e-3)
        print('Gradient check result:', test)

    @staticmethod
    def _conv_module(in_channels, out_channels, kernel, stride, padding, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _kernel_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
            upsample,
            torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        )

    @staticmethod
    def _upsample_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            upsample, torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight, init.calculate_gain('relu'))
