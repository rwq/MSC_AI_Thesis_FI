import torch
from torch.utils.data import Dataset
import os
import imageio
import re
from PIL import Image
import random
import numpy as np
from constants import FP_UCF101, FP_ADOBE240, FP_VIMEO90K, FP_MIDDLEBURY, FP_LMD, FP_INPUT_VIDEOS
import cv2
import pandas as pd
import albumentations as A
import time


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


def get_ucf101_files():
    return os.listdir(FP_UCF101)

def get_adobe240_files():
    return os.listdir(FP_ADOBE240)




# UCF101
def ucf101_generator(files=get_ucf101_files(), k=1, quadratic=False):
    basenames = get_basenames(files)
    
    if quadratic:
        image_ids = [0,1,3,4]
        gt = 2
    else:
        image_ids = [0,2]
        gt = 1
    
    for filebase, ext in basenames:
        frames = []
        
        for i in image_ids:
            filepath = os.path.join(FP_UCF101, f'{filebase}_{i}.{ext}')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)
            
        # load gt
        filepath = os.path.join(FP_UCF101, f'{filebase}_{gt}.{ext}')
        img = torch.Tensor(imageio.imread(filepath))
        
        yield frames, img



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


    def __init__(self, files=get_ucf101_files(), quadratic=False):#, transformer=None):
        # split into filename and extension
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

    def __init__(self, files=os.listdir(FP_LMD), quadratic=False):#, transformer=None):
        self.basenames = list(get_basenames(files))
        self.name = 'LMD'
        self.gt_id = 2
        self.quadratic = quadratic
        if quadratic:
            self.image_ids = [0,1,3,4]
        else:
            self.image_ids = [1,3]
            

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        frames = []
        filebase, ext = self.basenames[idx]
        
        for i in self.image_ids:
            filepath = os.path.join(FP_LMD, f'{filebase}_{i}.{ext}')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)

        frames = torch.stack(frames)

        filepath = os.path.join(FP_LMD, f'{filebase}_{self.gt_id}.{ext}')
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

# Vimeo90K

# Adobe240
def adobe240_generator(files=get_adobe240_files(), quadratic=False):
    basenames = get_basenames(files)


    if quadratic:
        image_ids = [0,1,3,4]
        gt = 2
    else:
        image_ids = [0,2]
        gt = 1

    for filebase, ext in basenames:
        frames = []
        
        for i in image_ids:
            filepath = os.path.join(FP_ADOBE240, f'{filebase}_{i}.{ext}')
            img = torch.Tensor(imageio.imread(filepath))
            frames.append(img)
            
        # load gt
        filepath = os.path.join(FP_ADOBE240, f'{filebase}_{gt}.{ext}')
        y = torch.Tensor(imageio.imread(filepath))
        
        yield frames, y


class adobe240_dataset(Dataset):

    def __init__(self, files=get_adobe240_files(), quadratic=False):#, transformer=None):
        self.basenames = list(get_basenames(files))
        self.name = 'Adobe240'
        # self.transformer = transformer
        self.gt_id = 2
        self.quadratic = quadratic
        if quadratic:
            self.image_ids = [0,1,3,4]
        else:
            self.image_ids = [1,3]
            

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        frames = []
        filebase, ext = self.basenames[idx]
        
        for i in self.image_ids:
            try:
                filepath = os.path.join(FP_ADOBE240, f'{filebase}_{i}.{ext}')
                img = torch.Tensor(imageio.imread(filepath))
                frames.append(img)
            except:
                print('error adobe240', filebase, i)
                exit()
        frames = torch.stack(frames)

        filepath = os.path.join(FP_ADOBE240, f'{filebase}_{self.gt_id}.{ext}')
        y = torch.Tensor(imageio.imread(filepath))

        # if self.transformer != None:
        #     frames, y = self.transformer.apply(frames, y)

        return frames, y

# def get_folderpaths_vimeo90k():
#     folders = os.listdir(FP_VIMEO90K)

#     folderpaths = []

#     for folder in folders:
#         subfolders = os.listdir(os.path.join(FP_VIMEO90K, folder))

#         for subfolder in subfolders:
#             path = os.path.join(FP_VIMEO90K, folder, subfolder)

#             folderpaths.append(path)
    
#     return folderpaths







class vimeo90k_dataset(Dataset):

    def __init__(self, quadratic=False, fold='all'):#, transformer=None):
        assert fold in ['train', 'test', 'all', 'valid', 'nontest']
        
        sequences = self.get_vimeo_sequences(fold)
        
        self.folderpaths = [os.path.join(FP_VIMEO90K, 'sequences', s) for s in sequences]
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

        try:
            for i in self.image_ids:
                img = imageio.imread(os.path.join(folderpath, f'im{i}.png'))
                img = torch.Tensor(img)
                frames.append(img)

            frames = torch.stack(frames)

            y = imageio.imread(os.path.join(folderpath, f'im{self.gt_id}.png'))
            y = torch.Tensor(y)
        except:
            print('error vimeo', idx, self.folderpaths[idx])


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
                 random_rescale=False,
                 random_rescale_prob=0,
                 rescale_range=(.8, 2.5),
                 crop_size=(128,128),
                 channels_first=True,
                 normalize=True):

        if isinstance(flip_probs, float) or isinstance(flip_probs, int):
            flip_probs = tuple(np.ones(3) * flip_probs)
            
        

        self.flip_probs = flip_probs
        self.jitter_prob = jitter_prob
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.dataset = dataset
        self.channels_first = channels_first
        self.normalize = normalize
        self.random_rescale = random_rescale
        self.rescale_range = rescale_range

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
        try:
            frames, y = self.dataset[idx]
        except:
            print(idx)
            time.sleep(100000)

        return self.apply(frames, y)

    def __len__(self):
        return self.dataset.__len__()


    def apply_tensor(self, X):
        assert X.ndimension() in [3,4]

        if X.ndimension() == 3:
            X = X.unsqueeze(dim=0)

        F, H, W, C = X.shape
        
        # horizontal flip
        h_flip, v_flip = np.random.rand(2) < self.flip_probs[:2]
        
        dims_to_flip = []
        if h_flip: dims_to_flip.append(1)
        if v_flip: dims_to_flip.append(2)
        
        if len(dims_to_flip) > 0:
            X = X.flip(dims=dims_to_flip)
        
        
        # random crop
        if self.random_crop:
            crop_start_h, crop_start_w = [
                np.random.randint(H-self.crop_size[0]),
                np.random.randint(W-self.crop_size[1])
            ]
            crop_end_h, crop_end_w = crop_start_h+self.crop_size[0], crop_start_w+self.crop_size[1]
            X = X[:, crop_start_h:crop_end_h, crop_start_w:crop_end_w, :]

        return X

    def transformation_on_input(self, X,y):
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
            

        return X,y



    def apply(self, input_frames, label):
        '''
        expects input frames as list of tensors HxWxC and label seperately
        '''

        # scale
        if self.random_rescale and random.random() < self.random_rescale_prob:
            X = 


        # crop


        # apply contrast, brightness saturation and hue change
        if np.random.rand(1) < self.jitter_prob:
            input_frames, label = self.transformation_on_input(input_frames,label)


        
        X = torch.cat([input_frames, label.unsqueeze(0)], dim=0)



        X = self.apply_tensor(X)

        if self.channels_first:
            X = X.permute(0,3,1,2)
        
        if self.normalize:
            X.div_(255.)
        
        input_frames, y = X[:-1], X[-1]

        if np.random.rand(1) < self.flip_probs[-1]:
            input_frames = input_frames.flip(dims=[0])
            
        return input_frames, y





# class Transformer():
    
#     def __init__(self, h_flip_prob = 0.1, v_flip_prob = 0.1, random_crop=True, crop_size=(128,128)):
#         self.h_flip_prob = h_flip_prob
#         self.v_flip_prob = v_flip_prob
#         self.random_crop = random_crop
#         self.crop_size = crop_size

#     def apply_tensor(self, X):
#         assert X.ndim in [3,4]

#         if X.ndim == 3:
#             X = X.unsqueeze(dim=0)

#         B, H, W, C = X.shape

        
#         # horizontal flip
#         h_flip, v_flip = np.random.rand(2) < [self.h_flip_prob, self.v_flip_prob]
        
#         dims_to_flip = []
#         if h_flip: dims_to_flip.append(1)
#         if v_flip: dims_to_flip.append(2)
        
#         X = X.flip(dims=dims_to_flip)
        
        
#         # random crop
#         if self.random_crop:
#             crop_start_h, crop_start_w = [
#                 np.random.randint(H-self.crop_size[0]),
#                 np.random.randint(W-self.crop_size[1])
#             ]
#             crop_end_h, crop_end_w = crop_start_h+self.crop_size[0], crop_start_w+self.crop_size[1]
#             X = X[:, crop_start_h:crop_end_h, crop_start_w:crop_end_w, :]

#         return X
        
#     def apply(self, input_frames, label):
#         '''
#         expects input frames as list of tensors HxWxC and label seperately
#         '''
#         X = torch.stack([*input_frames, label], dim=0)
        
#         X = self.apply_tensor(X)
            
#         X = X.unbind(dim=0)
        
#         input_frames, y = X[:-1], X[-1]
            
#         return input_frames, y



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

if __name__ == '__main__':

    files = get_adobe240_files()
    # print(files[0])
    gen = adobe240_generator(files)

    print(next(gen))