import time
import os
from torch.autograd import Variable
import torchvision
import torch
import random
import numpy as np
import numpy
import sys
sys.path.append('E:\scriptieAI\code\DAIN')
import networks
# from my_args import  args # TODO weggehaald
# from scipy.misc import imread, imsave
from imageio import imread
from PIL import Image

from AverageMeter import  *
import shutil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class Args():
    
    def __init__(self):
        self.filter_size = 4
        self.time_step = 0.5
        self.channels = 3
        self.netName = 'DAIN'
        self.use_cuda = True
        
        self.save_which = 1
        self.dtype = torch.cuda.FloatTensor
        
args = Args()

use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))

model = networks.__dict__[args.netName](    channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)

if args.use_cuda:
    print('USING CUDA')
    model = model.cuda()

model.eval()

args.SAVED_MODEL = 'E:/scriptieAI/modelsweights/dain/best.pth'

if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")


model = model.eval()



def interpolate_efficient(inputs):
    # X0 = torch.from_numpy( np.transpose(im1, (2,0,1)).astype('float32')/255.0).type(dtype)
    # X1 = torch.from_numpy( np.transpose(im2, (2,0,1)).astype('float32')/255.0).type(dtype)
    X0 = (inputs[0].permute(2,0,1)/255.).type(dtype)
    X1 = (inputs[1].permute(2,0,1)/255.).type(dtype)

    y_ = torch.FloatTensor()

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))
    
    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])
    
    torch.set_grad_enabled(False)
    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)
    
    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
        
    y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[save_which]
    
    
    if use_cuda:
        # X0 = X0.data.cpu().numpy()
        if not isinstance(y_, list):
            y_ = y_.data.cpu().numpy()
        else:
            y_ = [item.data.cpu().numpy() for item in y_]
        # offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        # filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        # X1 = X1.data.cpu().numpy()
    else:
        # X0 = X0.data.numpy()
        if not isinstance(y_, list):
            y_ = y_.data.numpy()
        else:
            y_ = [item.data.numpy() for item in y_]
        # offset = [offset_i.data.numpy() for offset_i in offset]
        # filter = [filter_i.data.numpy() for filter_i in filter]
        # X1 = X1.data.numpy()
        
    # X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))


    y_ = [np.transpose(255.0 * item.clip(0,1.0)[:, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    # 0 weggehaald als index

    # offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    # filter = [np.transpose(
    #     filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
    #     (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    # X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    del X0; del X1
    return y_[0]


def interpolate_old(im1, im2):
    X0 = torch.from_numpy( np.transpose(im1, (2,0,1)).astype('float32')/255.0).type(dtype)
    X1 = torch.from_numpy( np.transpose(im2, (2,0,1)).astype('float32')/255.0).type(dtype)

    y_ = torch.FloatTensor()

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))
    
    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])
    
    torch.set_grad_enabled(False)
    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)
    
    print('mean', X0.mean())
    
    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
        
    y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[save_which]
    
    
    if use_cuda:
        X0 = X0.data.cpu().numpy()
        if not isinstance(y_, list):
            y_ = y_.data.cpu().numpy()
        else:
            y_ = [item.data.cpu().numpy() for item in y_]
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        X1 = X1.data.cpu().numpy()
    else:
        X0 = X0.data.numpy()
        if not isinstance(y_, list):
            y_ = y_.data.numpy()
        else:
            y_ = [item.data.numpy() for item in y_]
        offset = [offset_i.data.numpy() for offset_i in offset]
        filter = [filter_i.data.numpy() for filter_i in filter]
        X1 = X1.data.numpy()
        
    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))


    y_ = [np.transpose(255.0 * item.clip(0,1.0)[:, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    # 0 weggehaald als index

    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    del X0; del X1
    return y_[0]



if __name__ == '__main__':
    None


