import torch
from torch.utils.data import Dataset
import os
import imageio
import re
from PIL import Image
import random
import numpy as np
from constants import *
import cv2
import pandas as pd
import albumentations as A
import time
import utilities
import math

HARD_TEST_INSTANCES = {
    'redbull480.mp4': [116, 763, 1089, 1253]
}



def get_frame_by_caption(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame
    

def generate_hard_vimeo(quadratic=False):
    '''generates a small sample of hard instances from the Vimeo-90k testset'''
    hard_vimeo = [1433, 3429, 7680, 3567, 2422, 3254, 2695]
    ds = vimeo90k_dataset(quadratic=quadratic, fold='test')

    for index in hard_vimeo:
        X, y = ds[index]

        X = X.permute(0,3,1,2).float().unsqueeze(dim=0).contiguous()
        y = y.permute(2,0,1).float().contiguous()

        yield X,y


def generate_hard_input(quadratic=False):
    
    if quadratic:
        l = [-3, -1, 1, 3]
    else:
        l = [-1, 1]
    
    
    for filename, frame_indices in HARD_TEST_INSTANCES.items():
        
        cap = cv2.VideoCapture(os.path.join(FP_INPUT_VIDEOS, filename))
        
        
        for index in frame_indices:
            frame_indices = [index+i for i in l]
            X = [get_frame_by_caption(cap, i) for i in frame_indices]
            X = torch.from_numpy(np.array(X))
            X = X.permute(0,3,1,2).float()
            X = X.unsqueeze(dim=0)

            y = get_frame_by_caption(cap, index)
            y = torch.from_numpy(y).permute(2,0,1).float()
            
            yield X, y


def get_filename_parts(file):
    # print('file', file)
    base, f, ext = re.findall(r'(.+__\d+)_(\d+)\.(.+)', file)[0]
    return base, f, ext

def get_basenames(files):
    basenames = []
    exts = []
    for file in files:
        # print('file1', file)
        base, _, ext = get_filename_parts(file)
        basenames.append(base)
        exts.append(ext)
        
    return sorted(list(set(zip(basenames, exts))))





class gopro_dataset(Dataset):

    def __init__(self, quadratic=False, fold='train'):#, transformer=None):
        
        assert fold in ['train', 'validation', 'test']
        
        self.name = 'GoPro'
        self.fold = fold
        self.gt_id = 12
        self.quadratic = quadratic
        if quadratic:
            self.image_ids = [0,8,16,24]
        else:
            self.image_ids = [8,16]
            
        self.folder_paths = self.get_folder_paths()
            
        
    def get_folder_paths(self):
        fold_path = os.path.join(FP_GOPRO, self.fold)
        return [os.path.join(FP_GOPRO, self.fold, f) for f in os.listdir(fold_path)]

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        frames = []
        folder = self.folder_paths[idx]
        
        for i in self.image_ids:
            filepath = os.path.join(folder, f'{i}.jpeg')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)

        frames = torch.stack(frames)

        filepath = os.path.join(folder, f'{self.gt_id}.jpeg')
        y = torch.Tensor(imageio.imread(filepath))
        
        return frames,y


class ucf101_dataset(Dataset):
    '''
    Pytorch Dataset class to be used with dataloader to load
    ucf101 images
    '''

    #TODO handle different sizes?

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        frames = []
        filebase, ext = self.basenames[idx]
        
        for i in self.image_ids:
            filepath = os.path.join(FP_UCF101, f'{filebase}_{i}.{ext}')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)

        frames = torch.stack(frames)

        # load gt
        filepath = os.path.join(FP_UCF101, f'{filebase}_{self.gt_id}.{ext}')
        img = torch.Tensor(imageio.imread(filepath))

        # if self.transformer != None:
        #     frames, y = self.transformer.apply(frames, y)

        return frames, img


    def __init__(self, quadratic=False):#, transformer=None):
        # split into filename and extension
        self.files=get_ucf101_files(), 
        self.basenames = list(get_basenames(files))
        self.transformer = transformer
        self.name = 'UCF101'

        self.gt_id = 2
        if quadratic:
            self.image_ids = [0,1,3,4]
        else:
            self.image_ids = [1,3]


def split_data(dataset, percentages):
    percentages = np.array(percentages)
    assert np.allclose(percentages.sum(), 1)

    N = len(dataset)
    fold_sizes = (percentages * N).round().astype(int)
    fold_sizes[-1] = N-fold_sizes[:-1].sum()
    fold_sizes = fold_sizes.tolist()

    return torch.utils.data.random_split(dataset, fold_sizes)


class large_motion_dataset(Dataset):

    def __init__(self, fold='train', quadratic=False, cropped=True, min_flow=0):#, transformer=None):
        
        self.name = 'LMD2'
        self.quadratic = quadratic
        if cropped:
            self.fold_path = os.path.join(FP_LMD, f'{fold}_cropped')
        else:
            self.fold_path = os.path.join(FP_LMD, fold)
        self.sequences = [os.path.join(self.fold_path, seq) for seq in os.listdir(self.fold_path)]
        
        if quadratic:
            self.image_ids = [0,2,4,6]
        else:
            self.image_ids = [2,4]

        if min_flow>0:
            flow_data = pd.read_csv('hardinstancesinfo/large_motion_dataset2_train_flow2.csv')
            subset = flow_data[flow_data.mean_manh_flow >= min_flow]
            self.sequences = [s for i,s in enumerate(self.sequences) if i in subset.index.tolist()]
            self.weights = subset.mean_manh_flow.tolist()
            

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames = []
        folderpath = self.sequences[idx]

        for i in self.image_ids:
            filepath = os.path.join(folderpath, f'{i}.jpg')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)

        frames = torch.stack(frames)

        filepath = os.path.join(folderpath, '3.jpg')
        y = torch.Tensor(imageio.imread(filepath))


        return frames, y













# middlebury
def middlebury_generator(quadratic=False):

    exclude = ['Dimetrodon', 'Venus'] # only 2 frames
    clips = os.listdir(os.path.join(FP_MIDDLEBURY, 'other-data'))
    clips = [c for c in clips if not c in exclude]

    for clip in clips:
        if quadratic:
            image_ids = [9,10,11,12]
        else:
            image_ids = [10,11]
            
        # load input
        frames = []
        for d in image_ids:
            fp = os.path.join(FP_MIDDLEBURY, 'other-data', clip, f'frame{str(d).zfill(2)}.png')
            img = torch.Tensor(imageio.imread(fp))
            frames.append(img)

        # load gt
        fp = os.path.join(FP_MIDDLEBURY, 'other-gt-interp', clip, f'frame10i11.png')
        img = torch.Tensor(imageio.imread(fp))
    
        yield frames, img



class adobe240_dataset(Dataset):

    def __init__(self, quadratic=False, fold='train'):#, transformer=None):
        
        assert fold in ['train', 'validation', 'test']
        
        self.name = 'Adobe240'
        self.fold = fold
        self.gt_id = 12
        self.quadratic = quadratic
        if quadratic:
            self.image_ids = [0,8,16,24]
        else:
            self.image_ids = [8,16]
            
        self.folder_paths = self.get_folder_paths()
            
        
    def get_folder_paths(self):
        fold_path = os.path.join(FP_ADOBE240, self.fold)
        return [os.path.join(FP_ADOBE240, self.fold, f) for f in os.listdir(fold_path)]

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        frames = []
        folder = self.folder_paths[idx]
        
        for i in self.image_ids:
            filepath = os.path.join(folder, f'{i}.jpeg')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)

        frames = torch.stack(frames)

        filepath = os.path.join(folder, f'{self.gt_id}.jpeg')
        y = torch.Tensor(imageio.imread(filepath))
        
        return frames,y










class vimeo90k_dataset(Dataset):

    def __init__(self, quadratic=False, fold='all', hard=False):#, transformer=None):
        if hard:
            fold='train'

        assert fold in ['train', 'test', 'all', 'valid', 'nontest']
        
        sequences = self.get_vimeo_sequences(fold)
        if hard:
            self.folderpaths = [os.path.join(FP_VIMEO90K_HARD, 'sequences', s) for s in sorted(sequences)]
        else:
            self.folderpaths = [os.path.join(FP_VIMEO90K, 'sequences', s) for s in sorted(sequences)]

        
        self.name = 'Vimeo90K'
        self.quadratic = quadratic

        if quadratic:
            self.image_ids = [1,3,5,7]
            self.gt_id = 4
        else:
            self.image_ids = [3,5]
            self.gt_id = 4


    def __len__(self):
        return len(self.folderpaths)

    def __getitem__(self, idx):
        folderpath = self.folderpaths[idx]
        frames = []

        for i in self.image_ids:
            img = imageio.imread(os.path.join(folderpath, f'im{i}.png'))
            img = torch.Tensor(img)
            frames.append(img)

        frames = torch.stack(frames)

        y = imageio.imread(os.path.join(folderpath, f'im{self.gt_id}.png'))
        y = torch.Tensor(y)



        return frames, y

    def get_vimeo_sequences(self, fold):

        if fold in ['train', 'test']:
            with open(os.path.join(FP_VIMEO90K, 'sep_{0}list.txt'.format(fold)), 'r') as f:
                sequences = f.readlines()
                sequences = [s.strip() for s in sequences]

        elif fold == 'all':

            # sequences = self.get_vimeo_sequences('train') + self.get_vimeo_sequences('test')
            sequences = []
            for folder in os.listdir(os.path.join(FP_VIMEO90K, 'sequences')):
                subfolders =[subfolder for subfolder in os.listdir(os.path.join(FP_VIMEO90K, 'sequences', folder))]
                subfolders = [f'{folder}/{subfolder}' for subfolder in subfolders]
                sequences.extend(subfolders)
        
        elif fold == 'valid':
            all = self.get_vimeo_sequences(fold='all')
            train = self.get_vimeo_sequences(fold='train')
            test = self.get_vimeo_sequences(fold='test')

            sequences = list( set(all)-set(train)-set(test) )

        elif fold == 'nontest':
            all = self.get_vimeo_sequences(fold='all')
            test = self.get_vimeo_sequences(fold='test')

            sequences = list( set(all)-set(test) )
        else:
            raise NotImplementedError()

        return sequences





def get_hard_instances_subset(dataset_name, measure, quantile, indices=None, quadratic=False):
    assert dataset_name in ['Adobe240', 'Vimeo90K']
    assert measure in ['mean_disp', 'max_disp', 'median_disp', 'non_linearity']
    
    # load dataset
    if dataset_name == 'Adobe240':
        ds = adobe240_dataset(quadratic=quadratic)
    elif dataset_name == 'Vimeo90K':
        ds = vimeo90k_dataset(quadratic=quadratic)
        
    # load meta data and subset
    df = pd.read_csv(f'hardinstancesinfo/{dataset_name}.csv')
    if indices != None: df = df[df.index.isin(indices)]
    
    # only keep top (1-q)%
    q = df.quantile(quantile)
    df = df[df[measure] >= q[measure]]
    
    subset = torch.utils.data.Subset(ds, indices=df.index)
    
    return subset


class TransformedDataset(Dataset):

    def __init__(self,
                 dataset,
                 flip_probs = (0.1, 0.1, 0.2),
                 jitter_prob = 0,
                 random_crop=True,
                 random_rescale_prob=0,
                 rescale_distr=(.8, 1.2),
                 crop_size=(128,128),
                #  channels_first=True,
                 normalize=True,
                 hard_alternative_ds=None,
                 hard_prob=0):

        if isinstance(flip_probs, float) or isinstance(flip_probs, int):
            flip_probs = tuple(np.ones(3) * flip_probs)
            
        

        self.flip_probs = flip_probs
        self.jitter_prob = jitter_prob
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.dataset = dataset
        # self.channels_first = channels_first
        self.normalize = normalize
        self.rescale_distr = rescale_distr
        self.random_rescale_prob = random_rescale_prob
        self.hard_alternative_ds = hard_alternative_ds
        self.hard_prob = hard_prob


        if hasattr(self.dataset, 'quadratic'):
            self.quadratic = dataset.quadratic
        else:
            self.quadratic = dataset.dataset.quadratic

        if self.quadratic:
            additional_targets = {'imager':'image', 'y':'image', 'imagell':'image', 'imagerr':'image'}
        else:
            additional_targets = {'imager':'image', 'y':'image'}

        self.jitter = A.Compose([
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=.5),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=50, p=1),
            A.ChannelShuffle(p=.5)
        ], p=1, additional_targets=additional_targets)


    def __getitem__(self, idx):
        use_default = self.hard_alternative_ds == None or random.random() > self.hard_prob

        if use_default:
            frames, y = self.dataset[idx]
            return self.apply(frames, y)
        else:
            frames, y = self.hard_alternative_ds[idx]

            return self.apply(frames, y)

    def __len__(self):
        return self.dataset.__len__()


    def transformation_on_input(self, C):
        C = C.permute(0,2,3,1)
        X, y = C[:-1], C[-1]
        X = X.numpy().astype('uint8')
        y = y.numpy().astype('uint8')
        if self.quadratic:
            transformed = self.jitter(
                image = X[1],
                y = y,
                imager = X[2],
                imagell = X[0],
                imagerr = X[3]
            )
            
            X = torch.stack([torch.from_numpy(transformed[k]) for k in ['imagell', 'image', 'imager', 'imagerr']], dim=0).float()
        else:
            transformed = self.jitter(
                image = X[0],
                y = y,
                imager = X[1]
            )

            X = torch.stack([torch.from_numpy(transformed[k]) for k in ['image', 'imager']], dim=0).float()
        
        y = torch.from_numpy(transformed['y']).float()

        X = X.permute(0,3,1,2)
        y = y.permute(2,0,1).unsqueeze(0)
        C = torch.cat([X, y], dim=0)
            

        return C



    def apply(self, input_frames, label):
        '''
        expects input frames as list of tensors HxWxC and label seperately
        '''
        # combine everything to one tensor
        X = torch.cat([input_frames, label.unsqueeze(0)], dim=0)

        # reverse channels
        X = X.permute(0,3,1,2)

        # scale and random (interesting crop)
        if random.random() < self.random_rescale_prob:
            scale = np.random.uniform(*self.rescale_distr)
            X = torch.nn.functional.interpolate(
                X, 
                scale_factor=(scale,scale), 
                recompute_scale_factor=True,
                mode= 'bilinear',
                align_corners=False
            )





        # random crop
        if self.random_crop:
            _, _, H, W = X.shape
            crop_start_h, crop_start_w = (
                np.random.randint(H-self.crop_size[0]),
                np.random.randint(W-self.crop_size[1])
            )
            crop_end_h, crop_end_w = crop_start_h+self.crop_size[0], crop_start_w+self.crop_size[1]
            X = X[:, :, crop_start_h:crop_end_h, crop_start_w:crop_end_w]


        # apply contrast, brightness saturation and hue change
        if np.random.rand(1) < self.jitter_prob:
            X = self.transformation_on_input(X)


        
        # horizontal/vertical flips
        h_flip, v_flip = np.random.rand(2) < self.flip_probs[:2]
        
        dims_to_flip = []
        if h_flip: dims_to_flip.append(2)
        if v_flip: dims_to_flip.append(3)
        
        if len(dims_to_flip) > 0:
            X = X.flip(dims=dims_to_flip)
        
        
        if self.normalize:
            X.div_(255.)
        
        # split back
        input_frames, y = X[:-1], X[-1]

        # temporal flip
        if np.random.rand(1) < self.flip_probs[-1]:
            input_frames = input_frames.flip(dims=[0])

           
        return input_frames, y









def get_datagenerator(dataset, quadratic=False):
    DATASETS = ['Adobe240', 'UCF101', 'Vimeo90k']

    if dataset == 'Adobe240':
        return adobe240_generator(quadratic=quadratic)
    elif dataset == 'UCF101':
        return ucf101_generator(quadratic=quadratic)
    elif dataset == 'Vimeo90k':
        return vimeo90k_generator(quadratic=quadratic)
    else:
        raise NotImplementedError()

