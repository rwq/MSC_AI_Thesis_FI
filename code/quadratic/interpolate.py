import sys
sys.path.append('E:\scriptieAI\code\quadratic')


import models
# import datas
# import configs

# hacky


import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
from tensorboardX import SummaryWriter


import time

import cv2

# loading configures
# parser = argparse.ArgumentParser()
# parser.add_argument('config')
# args = parser.parse_args()
# args = parser.parse_config()

# config = Config.from_file(args.config)
config = Config.from_file('E:/scriptieAI/code/quadratic/configs/config_test.py')
# print(config.mean)
# preparing datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
# trans_old = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])
trans_old = TF.Compose([TF.ToPILImage(), TF.ToTensor(), normalize1, normalize2, ])
trans = TF.Compose([normalize1, normalize2, ])

# trans2 = TF.Compose([normalize1, normalize2, ])
revmean = [-x for x in config.mean]
revstd = [1.0 / x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2])

# testset = datas.AIMSequence(config.testset_root, trans, config.test_size, config.test_crop_size, config.inter_frames)
# sampler = torch.utils.data.SequentialSampler(testset)
# validationloader = torch.utils.data.DataLoader(testset, sampler=sampler, batch_size=1, shuffle=False, num_workers=1)


# model
model = getattr(models, config.model)(config.pwc_path).cuda()
model = nn.DataParallel(model)

dict1 = torch.load(config.checkpoint)
model.load_state_dict(dict1['model_state_dict'])


tot_time = 0
tot_frames = 0

# print('Everything prepared. Ready to test...')

to_img = TF.ToPILImage()

def generate():
    global tot_time, tot_frames
    retImg = []
   
    
    store_path = config.store_path

    with torch.no_grad():
        for validationIndex, validationData in enumerate(validationloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, folder, index = validationData

            # make sure store path exists
            if not os.path.exists(config.store_path + '/' + folder[1][0]):
                os.mkdir(config.store_path + '/' + folder[1][0])

            # if sample consists of four frames (ac-aware)
            if len(sample) is 4:
                frame0 = sample[0]
                frame1 = sample[1]
                frame2 = sample[-2]
                frame3 = sample[-1]

                I0 = frame0.cuda()
                I3 = frame3.cuda()

                I1 = frame1.cuda()
                I2 = frame2.cuda()

                revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/'  + index[1][0] + '.png')
                revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[-2][0] + '/' +  index[-2][0] + '.png')
            # else two frames (linear)
            else:
                frame0 = None
                frame1 = sample[0]
                frame2 = sample[-1]
                frame3 = None

                I0 = None
                I3 = None
                I1 = frame1.cuda()
                I2 = frame2.cuda()
             
                revtrans(I1.clone().cpu()[0]).save(store_path + '/' + folder[0][0] + '/'  + index[0][0] + '.png')
                revtrans(I2.clone().cpu()[0]).save(store_path + '/' + folder[1][0] + '/' +  index[1][0] + '.png')

            
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                print(t)


                # record duration time
                start_time = time.time()

                output = model(I0, I1, I2, I3, t)
                It_warp = output
                
                tot_time += (time.time() - start_time)
                tot_frames += 1
                

                if len(sample) is 4:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[1][0] + '_' + str(tt) + '.png')
                else:
                    revtrans(It_warp.cpu()[0]).save(store_path + '/' + folder[0][0] + '/' + index[0][0] + '_' + str(tt) + '.png')

from PIL import Image







def interpolate(sample, t=.5):

    input_tensors = sample.copy()
    for i in range(len(input_tensors)):
        input_tensors[i] = normalize_batch(input_tensors[i], config.mean, config.std)
    
    # for tensor in input_tensors:
    #     print(tensor.shape, tensor.is_contiguous(), tensor.device)

    # quadratic test
    if len(input_tensors) == 4:
        I0, I1, I2, I3 = input_tensors

    else:
        I0 = None
        I1 = input_tensors[0]
        I2 = input_tensors[-1]
        I3 = None

    model.eval()
    with torch.no_grad():
        output = model(I0, I1, I2, I3, t)
        output = normalize_batch(output, [0., 0., 0.], revstd)
        output = normalize_batch(output, revmean, [1., 1., 1.])

    return output



def normalize_batch(tensor, mean, sd):
    mean = torch.tensor(mean).view(3,1,1).to(tensor.device)
    sd   = torch.tensor(sd).view(3,1,1).to(tensor.device)

    mean = mean.contiguous()
    sd   = sd.contiguous()

    return (tensor-mean) / sd



def interpolate_test(sample, t=.5):

    input_tensors = sample.copy()
    

    for tensor in input_tensors:
        print(tensor.shape, tensor.is_contiguous(), tensor.device)




    # quadratic test
    if len(input_tensors) == 4:
        I0, I1, I2, I3 = input_tensors

    else:
        I0 = None
        I1 = input_tensors[0]
        I2 = input_tensors[-1]
        I3 = None

    model.eval()
    with torch.no_grad():
        output = model(I0, I1, I2, I3, t)
        output = normalize_batch(output, [0., 0., 0.], revstd)
        output = normalize_batch(output, revmean, [1., 1., 1.])

    return output









def interpolate_old(sample, t=.5):
    # for i in range(len(sample)):
    #     sample[i] = sample[i].permute(2,0,1).unsqueeze(0)/255.
    #     print(sample[i].shape)

    # overwrite input
    # samples = []

    input_tensors = sample.copy()
    for i in range(len(input_tensors)):
        input_tensors[i] = trans_old(input_tensors[i].type(torch.uint8))
        input_tensors[i] = torch.Tensor(input_tensors[i]).unsqueeze(0).float()

 
    


    # sample = [img1, img2, img3, img4]
    # quadratic test
    if len(input_tensors) == 4:
        frame0 = input_tensors[0]
        frame1 = input_tensors[1]
        frame2 = input_tensors[-2]
        frame3 = input_tensors[-1]

        I0 = frame0.cuda()
        I3 = frame3.cuda()

        I1 = frame1.cuda()
        I2 = frame2.cuda()
    else:
        frame0 = None
        frame1 = input_tensors[0]
        frame2 = input_tensors[-1]
        frame3 = None

        I0 = None
        I3 = None
        I1 = frame1.cuda()
        I2 = frame2.cuda()

    model.eval()
    with torch.no_grad():
        output = model(I0, I1, I2, I3, t)

        y = revtrans(output.cpu()[0]) * 255

    return y.permute(1,2,0)

                    
def test():

    dict1 = torch.load(config.checkpoint)
    model.load_state_dict(dict1['model_state_dict'])
    print('LOADED')
    if not os.path.exists(config.store_path):
        os.mkdir(config.store_path)
    generate()

# print(testset)


# test()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # # print ('Avg time is {} second'.format(tot_time/tot_frames))
    y = interpolate(None)

    plt.imshow(y)
    plt.show()


