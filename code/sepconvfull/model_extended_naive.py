import sys
import os
curdir = os.path.abspath('.')
sys.path.append(os.path.join(curdir, 'code/sepconvfull'))

import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sepconv
from torch.nn import functional as F
import numpy as np
import utilities



def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)








class SubnetExtended(torch.nn.Module):
    
    def __init__(self, kernel_size=51, d_size=None, d_scale=None, dropout_probs=(0,0)):
        super(SubnetExtended, self).__init__()
        self.kernel_size = kernel_size
        self.d_size = d_size
        self.d_scale = d_scale
        
        self.subnetA = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)            
        )

        def SubnetB(kernel_size, scale, dropout):
            if kernel_size == None or scale == None:
                return None

            return  torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2/scale, mode='bilinear', align_corners=True),
                torch.nn.Dropout2d(p=dropout),
                torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1)
            )

        self.kernel = SubnetB(kernel_size=self.kernel_size, scale=1, dropout=dropout_probs[0])
        self.kernel_d = SubnetB(kernel_size=self.d_size, scale = self.d_scale, dropout=dropout_probs[1])
        
        

        
        
        
    def forward(self, x):
        out = self.subnetA(x)

        out1 = out2 = None

        if self.kernel_size != None:
            out1 = self.kernel(out)
        
        if self.d_size != None:
            out2 = self.kernel_d(out)
        
        return out1, out2



class KernelEstimationExtended(torch.nn.Module):
    def __init__(self, 
                 kl_size, 
                 kl_d_size=None, 
                 kl_d_scale=2, 
                 kq_size=None,
                 kq_d_size=None,
                 kq_d_scale=2,
                 kq_inp=None,
                 input_frames=2):

        super(KernelEstimationExtended, self).__init__()

        self.kl_size = kl_size
        self.kl_d_size = kl_d_size
        self.kl_d_scale = kl_d_scale
        self.kq_size = kq_size
        self.kq_d_size = kq_d_size
        self.kq_d_scale = kq_d_scale
        self.input_frames = input_frames

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


        self.moduleConv1 = Basic(self.input_frames * 3, 32)
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

        # hier 4 modules toevoegen

        self.moduleVertical1 = SubnetExtended(kernel_size=kl_size, d_size=kl_d_size, d_scale=kl_d_scale, dropout_probs=(0, 0))
        self.moduleVertical2 = SubnetExtended(kernel_size=kl_size, d_size=kl_d_size, d_scale=kl_d_scale, dropout_probs=(0, 0))
        self.moduleHorizontal1 = SubnetExtended(kernel_size=kl_size, d_size=kl_d_size, d_scale=kl_d_scale, dropout_probs=(0, 0))
        self.moduleHorizontal2 = SubnetExtended(kernel_size=kl_size, d_size=kl_d_size, d_scale=kl_d_scale, dropout_probs=(0, 0))

        # if we use another 4 filters for quadratic input
        if self.kq_size != None or self.kq_d_size != None:
            self.moduleVerticalQ1 = SubnetExtended(kernel_size=kq_size, d_size=kq_d_size, d_scale=kq_d_scale)
            self.moduleVerticalQ2 = SubnetExtended(kernel_size=kq_size, d_size=kq_d_size, d_scale=kq_d_scale)
            self.moduleHorizontalQ1 = SubnetExtended(kernel_size=kq_size, d_size=kq_d_size, d_scale=kq_d_scale)
            self.moduleHorizontalQ2 = SubnetExtended(kernel_size=kq_size, d_size=kq_d_size, d_scale=kq_d_scale)





    def forward(self, rfields):
        B, F, C, H, W = rfields.shape
        
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

        if self.kq_size != None or self.kq_d_size != None:
            VerticalQ1 = self.moduleVerticalQ1(tensorCombine)
            VerticalQ2 = self.moduleVerticalQ2(tensorCombine)
            HorizontalQ1 = self.moduleHorizontalQ1(tensorCombine)
            HorizontalQ2 = self.moduleHorizontalQ2(tensorCombine)

            return Vertical1, Horizontal1, Vertical2, Horizontal2, VerticalQ1, HorizontalQ1, VerticalQ2, HorizontalQ2


        return Vertical1, Horizontal1, Vertical2, Horizontal2, None, None, None, None




class SepConvNetExtended(torch.nn.Module):
    def __init__(self,
                 kl_size, 
                 kl_d_size=None, 
                 kl_d_scale=2, 
                 kq_size=None,
                 kq_d_size=None,
                 kq_d_scale=2,
                 input_frames=2):
        super(SepConvNetExtended, self).__init__()
        self.kl_size = kl_size
        self.kl_d_size = kl_d_size
        self.kl_d_scale = kl_d_scale
        self.kq_size = kq_size
        self.kq_d_size = kq_d_size
        self.kq_d_scale = kq_d_scale
        self.input_frames = input_frames
        
        if (self.kq_size != None)or(self.kq_d_size != None):
            assert self.input_frames == 4
        

        self.get_kernel = KernelEstimationExtended(
            kl_size=self.kl_size, 
            kl_d_size=self.kl_d_size, 
            kl_d_scale=self.kl_d_scale,
            kq_size=self.kq_size,
            kq_d_size=self.kq_d_size,
            kq_d_scale=self.kq_d_scale,
            input_frames=self.input_frames
        )

        # linear kernel
        kernel_pad_l = int(math.floor(kl_size / 2.0))
        self.modulePad_l = torch.nn.ReplicationPad2d([kernel_pad_l, kernel_pad_l, kernel_pad_l, kernel_pad_l])

        
        # if displacement kernel for linear input
        if self.kl_d_size != None:
            kernel_pad_ld = int(math.floor(kl_d_size / 2.0))
            self.modulePad_ld = torch.nn.ReplicationPad2d([kernel_pad_ld, kernel_pad_ld, kernel_pad_ld, kernel_pad_ld])

            self.down_l = torch.nn.AvgPool2d(kernel_size=kl_d_scale, stride=kl_d_scale)
            self.up_l  = torch.nn.Upsample(scale_factor=kl_d_scale, mode='bilinear')

        # if quadratic input
        if self.kq_size != None:
            kernel_pad_q = int(math.floor(kq_size / 2.0))
            self.modulePad_q = torch.nn.ReplicationPad2d([kernel_pad_q, kernel_pad_q, kernel_pad_q, kernel_pad_q])

        # if displacement kernel for quadratic input
        if self.kq_d_size != None:
            kernel_pad_qd = int(math.floor(kq_d_size / 2.0))
            self.modulePad_qd = torch.nn.ReplicationPad2d([kernel_pad_qd, kernel_pad_qd, kernel_pad_qd, kernel_pad_qd])

            self.down_q = torch.nn.AvgPool2d(kernel_size=kq_d_scale, stride=kq_d_scale)
            self.up_q  = torch.nn.Upsample(scale_factor=kq_d_scale, mode='bilinear')


    def load_weights(self, loss='l1'):
        
        weights = torch.load(f'code/sepconv/network-{loss}.pytorch')
        weights = utilities.convert_weights(weights)

        weights_encoder = {k:v for k,v in weights.items() if 'moduleConv' in k}
        weights_cur_architecture = self.state_dict()
        

        if self.input_frames == 4:
            new_weights_layer1 = weights_cur_architecture['get_kernel.moduleConv1.0.weight']
            new_weights_layer1[:,3:9,:,:] = weights_encoder['get_kernel.moduleConv1.0.weight']
            weights_encoder['get_kernel.moduleConv1.0.weight'] = new_weights_layer1

        weights_cur_architecture.update(weights_encoder)
        self.load_state_dict(weights_cur_architecture)



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

        # get kernels from subnets
        V1, H1, V2, H2, VQ1, HQ1, VQ2, HQ2 = self.interpolation_kernels = self.get_kernel(padded_frames)

        
        frame_before = int(0+f/4)
        frame_after  = int(1+f/4)
            
        tensorDotL = sepconv.FunctionSepconv()(self.modulePad_l(padded_frames[:, frame_before]), V1[0], H1[0])
        tensorDotR = sepconv.FunctionSepconv()(self.modulePad_l(padded_frames[:, frame_after]), V2[0], H2[0])
   
        if self.kl_d_size != None:
            # downscale input frames
            im1d = self.down_l(padded_frames[:, frame_before])
            im2d = self.down_l(padded_frames[:, frame_after])
            
            # convolve and upscale back to original size
            tensorDotL += self.up_l( sepconv.FunctionSepconv()(self.modulePad_ld(im1d), V1[1], H1[1]) )
            tensorDotR += self.up_l( sepconv.FunctionSepconv()(self.modulePad_ld(im2d), V2[1], H2[1]) )


        if self.kq_size != None:
            tensorDotLL = sepconv.FunctionSepconv()(self.modulePad_q(padded_frames[:, 0]), VQ1[0], HQ1[0])
            tensorDotRR = sepconv.FunctionSepconv()(self.modulePad_q(padded_frames[:, 3]), VQ2[0], HQ2[0])
        else:
            tensorDotLL = tensorDotRR = 0

            
        if self.kq_d_size != None:
            im1qd = self.down_q(padded_frames[:, 0])
            im2qd = self.down_q(padded_frames[:, 3])

            tensorDotLL += self.up_q( sepconv.FunctionSepconv()(self.modulePad_qd(im1qd), VQ1[1], HQ1[1]) )
            tensorDotRR += self.up_q( sepconv.FunctionSepconv()(self.modulePad_qd(im2qd), VQ2[1], HQ2[1]) )




        frame_out = tensorDotL + tensorDotR + tensorDotLL + tensorDotRR

        if h_padded:
            frame_out = frame_out[:, :, 0:h, :]
        if w_padded:
            frame_out = frame_out[:, :, :, 0:w]
        
        return frame_out




