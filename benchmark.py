
import time
import os
from torch.autograd import Variable
import torchvision
import torch
import random
import numpy as np
import numpy


import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
import pickle
import imageio
import dataloader
import metrics
import utilities

'''

Datasets:
* Vimeo90k
* UCF101
* Adobe240

Models:
* QuadraticVideoINterpolation
* DAIN
* SepConv

Metrics
* PSNR
* SSIM
* IE

'''
# DATASETS = ['Adobe240', 'UCF101', 'Vimeo90k']
DATASETS = ['Adobe240']
METHODS = ['DAIN', 'SepConv', 'QVI', 'QVI_linear']
FILEPATH_RESULTS = 'results_new2.pkl'

from code.DAIN.interpolate import interpolate_efficient as interpolate
# from code.quadratic import interpolate
# from code.sepconv import interpolate

class ResultStore:   

    def __init__(self, filepath):
        self.filepath = filepath

        datasets = ['Adobe240', 'UCF101', 'Vimeo90k']
        methods = ['DAIN', 'SepConv', 'QVI', 'QVI_linear']
        metrics = ['psnr', 'ssim', 'ie']

        if not os.path.exists(filepath):
            self.results = defaultdict(dict)
            for mtd in methods:
                for ds in datasets:
                    self.results[mtd][ds] = dict()
                    for mtrc in metrics:                        
                        self.results[mtd][ds][mtrc] = []

            self.save()

        else:
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)

    def save(self):
        with open(self.filepath, 'wb') as f:
            pickle.dump(self.results, f)

    def store(self, method, dataset, values):
        self.results[method][dataset]['ssim'].append(values[0])
        self.results[method][dataset]['psnr'].append(values[1])
        self.results[method][dataset]['ie'].append(values[2])



method = 'DAIN'
N_TEST = 1000
cropper = utilities.Cropper(height=720, width=1280)
results = ResultStore(filepath=FILEPATH_RESULTS)

for dataset in DATASETS:
    print(f'[{time.ctime()}] Dataset: {dataset}')
    is_quadratic = method == 'QVI'
    gen = dataloader.get_datagenerator(dataset, quadratic=is_quadratic)
    
    k = 0
    for inputs, ii in tqdm(gen, total=N_TEST):
        
        cornercrops1 = cropper.crop(inputs[0].numpy())
        cornercrops2 = cropper.crop(inputs[1].numpy())

        cropped_results = []
        for corner1, corner2 in zip(cornercrops1, cornercrops2):
            corner1 = torch.Tensor(np.array(corner1))
            corner2 = torch.Tensor(np.array(corner2))
            result = interpolate([corner1, corner2])
            cropped_results.append(result)
        

        result = torch.Tensor(cropper.decrop(*cropped_results)).int()

        # result = torch.Tensor(interpolate(inputs))#, t=0.5))
        

        # imageio.imsave('test_image1.png', i1.numpy())
        # imageio.imsave('test_image2.png', i2.numpy())
        # imageio.imsave('test_pred.png', result.numpy())
        # imageio.imsave('test_gt.png', ii.numpy())
        result = result.unsqueeze(0)
        ii = ii.unsqueeze(0)

        # compute metrics
        ssim = metrics.ssim(result, ii).item()
        psnr = metrics.psnr(result, ii).item()
        ie = metrics.interpolation_error(result, ii).item()

        results.store(method=method, dataset=dataset, values=[ssim, psnr, ie])
        k+= 1

        if k == N_TEST:
            break




results.save()
