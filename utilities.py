import torchvision
import numpy as np
from PIL import Image
from code.sepconvfull import model
import torch
from collections import defaultdict, OrderedDict
import math

class Cropper:
    
    
    def __init__(self, height, width, delta=40):
        self.h = height
        self.w = width
        self.d = delta
        self.H = int((height/2)+delta)
        self.W = int((width/2)+delta)
        
    def crop(self, img):
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))
        
        TL = torchvision.transforms.functional.crop(img, i=0, j=0, h=self.H, w=self.W)
        TR = torchvision.transforms.functional.crop(img, i=0, j=self.w-self.W, h=self.H, w=self.W)
        BL = torchvision.transforms.functional.crop(img, i=self.h-self.H, j=0, h=self.H, w=self.W)
        BR = torchvision.transforms.functional.crop(img, i=self.h-self.H, j=self.w-self.W, h=self.H, w=self.W)
        
        return TL, TR, BL, BR
    
    def decrop(self, TL, TR, BL, BR):
        R = np.zeros((self.h,self.w, 3))
        d2 = int(self.d/2)
        R[:self.H-d2, :self.W-d2, :] = TL[:self.H-d2, :self.W-d2, :]
        R[:self.H-d2, (self.w-self.W+d2):, :] = TR[:self.H-d2, -(self.W-d2):, :]
        R[(self.h-self.H+d2):, :self.W-d2, :] = BL[-(self.H-d2):, :self.W-d2, :]
        R[(self.h-self.H+d2):, (self.w-self.W+d2):, :] = BR[-(self.H-d2):, -(self.W-d2):, :]
        
        return R


class EarlyStopping:
    '''
    Returns True if valid metric did not improve for the last 'patience' epochs.


    params:
        valid_results: dictionary with the validation metric results per epoch
        patience: number of allowed consecutive epochs without improvement
        mode: whether to maximize or minimize the metric

    '''
    
    def __init__(self, results, patience, objective='min', metric='L1_loss'):
        self.results = results
        self.patience = patience
        self.objective = objective
        self.metric = metric
        
    def stop(self):
        self.new_best()
    
        if self.time_since_best > self.patience:
            return True
        else:
            return False
        

    
    def new_best(self):
        values = self.results.results['valid'][self.metric]
        epochs = list(values.keys())
        means = np.array([np.mean(values[epoch]) for epoch in epochs])
        
        if len(means) <= 1:
            self.time_since_best = 0
            return True
        
        current_epoch = len(means)
        
        if self.objective == 'max':
            self.time_since_best = current_epoch - means.argmax() + 1
        elif self.objective == 'min':
            self.time_since_best = current_epoch - means.argmin() + 1
        else:
            raise NotImplementedError()
            
        if self.time_since_best == 0:
            return True


class ResultStore:
    
    def __init__(self, folds=['train', 'valid'], metrics=['psnr', 'ie', 'L1_loss', 'accuracy'], writer=None):
        self.folds = folds
        self.metrics = metrics
        self.results = dict()
        self.writer = writer
        
        for fold in self.folds:
            self.results[fold] = dict()
            for metric in self.metrics:
                self.results[fold][metric] = defaultdict(list)
        
    def store(self, fold, epoch, value_dict):
        for metric, value in value_dict.items():
            if isinstance(value, list):
                self.results[fold][metric][epoch].extend(value)
            else:
                self.results[fold][metric][epoch].append(value)
        
    def write_tensorboard(self, fold, epoch):
        for metric in self.metrics:
            mean = np.mean(self.results[fold][metric][epoch])
            
            self.writer.add_scalar(f'{metric}/{fold}', mean, epoch)
            self.writer.add_histogram(f'{metric}/{fold}_hist', np.array(self.results[fold][metric][epoch]), epoch)




def convert_weights(weights):
    w = OrderedDict()
    for key in weights:
        new_key = 'get_kernel.'+key
        w[new_key] = weights[key]
        
    return w



def convert_subnet(subnet, kernel_size):
    
    pad = int((kernel_size - 51)/2)
    
    # change second to last layer
    W = subnet[4].weight.data
    b = subnet[4].bias.data

    subnet[4] = nn.Conv2d(64, kernel_size, 3, stride=1, padding=1)

    subnet[4].weight.data.zero_()
    subnet[4].bias.data.zero_()
    subnet[4].weight.data[pad:-pad] = W
    subnet[4].bias.data[pad:-pad] = b
    
    
    # change last layer
    b = subnet[7].bias.data
    W = subnet[7].weight.data

    subnet[7] = nn.Conv2d(kernel_size, kernel_size, 3, stride=1, padding=1)
    
    subnet[7].weight.data.zero_()
    subnet[7].bias.data.zero_()
    
    subnet[7].weight.data[pad:-pad, pad:-pad] = W
    subnet[7].bias.data[pad:-pad] = b
    
    return subnet



def get_sepconv(input_size=2, kernel_size=51, weights='l1'):

    assert weights in ['l1', 'lf']
    
    kernel_size = 61
    assert kernel_size % 2 == 1 and kernel_size >= 51
    kernel_pad = int(math.floor(kernel_size / 2.0))
    

    sepconv = model.SepConvNet(kernel_size=51)

    weights = torch.load(f'code/sepconv/network-{weights}.pytorch')
    weights = convert_weights(weights)

    sepconv.load_state_dict(weights)
    
    
    if kernel_size > 51:
        sepconv.get_kernel.moduleVertical1 = convert_subnet(sepconv.get_kernel.moduleVertical1, kernel_size)
        sepconv.get_kernel.moduleVertical2 = convert_subnet(sepconv.get_kernel.moduleVertical2, kernel_size)
        sepconv.get_kernel.moduleHorizontal1 = convert_subnet(sepconv.get_kernel.moduleHorizontal1, kernel_size)
        sepconv.get_kernel.moduleHorizontal2 = convert_subnet(sepconv.get_kernel.moduleHorizontal2, kernel_size)

        sepconv.modulePad = torch.nn.ReplicationPad2d([kernel_pad, kernel_pad, kernel_pad, kernel_pad])
        

    if input_size == 4:

        # save old weights
        W = sepconv.get_kernel.moduleConv1[0].weight.data
        b = sepconv.get_kernel.moduleConv1[0].bias.data

        # change architecture (6->12 channels)
        sepconv.get_kernel.moduleConv1[0] = torch.nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)

        # replace old weights for first 6 channels, randomly init others
        sepconv.get_kernel.moduleConv1[0].weight.data.zero_()
        sepconv.get_kernel.moduleConv1[0].weight.data[:, 3:9, :, :] = W
        
        sepconv.get_kernel.moduleConv1[0].bias.data.zero_()
        sepconv.get_kernel.moduleConv1[0].bias.data = b
    
        return sepconv
    
    else:
        return sepconv


def create_grid(y_hat, y, **args):
    '''
    Creates image grid by stacking predictions y_hat and 
    ground truth y.
    
    y_hat: predicted inputs, shape B x 3 x H x W
    y: true values, same shape as y_hat
    '''
    
    assert y_hat.shape == y.shape, f'shapes are {y_hat.shape, y.shape}'
    
    inp_tensor = torch.cat([y_hat, y])
    grid = torchvision.utils.make_grid(inp_tensor, **args)
    
    return grid