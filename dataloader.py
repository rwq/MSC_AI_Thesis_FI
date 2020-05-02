import torch
from torch.utils.data import Dataset
import os
import imageio
import re
from PIL import Image
import random
import numpy as np
from constants import BASEDIR_DATA
import cv2


UCF101 = 'created_datasets/UCF101'
ADOBE240 = 'created_datasets/ADOBE240'
VIMEO90K = 'created_datasets/vimeo_test_clean/sequences'
MIDDLEBURY = 'datasets/MiddleBurySet/'


FP_UCF101 = os.path.join(BASEDIR_DATA, UCF101)
FP_ADOBE240 = os.path.join(BASEDIR_DATA, ADOBE240)
FP_VIMEO90K = os.path.join(BASEDIR_DATA, VIMEO90K)
FP_MIDDLEBURY = os.path.join(BASEDIR_DATA, MIDDLEBURY)

HARD_TEST_INSTANCES = {
    'input_videos/redbull480.mp4': [116, 763, 1089, 1253]
}



def get_frame_by_caption(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame
    
    

def generate_hard_input(quadratic=False):
    
    if quadratic:
        l = [-2, -1, 1, 2]
    else:
        l = [-1, 1]
    
    
    for filename, frame_indices in HARD_TEST_INSTANCES.items():
        
        cap = cv2.VideoCapture(os.path.join(BASEDIR_DATA, filename))
    
        
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
        self.gt_id = 2
        if quadratic:
            self.image_ids = [0,1,3,4]
        else:
            self.image_ids = [1,3]














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
        # self.transformer = transformer
        self.gt_id = 2
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

def get_folderpaths_vimeo90k():
    folders = os.listdir(FP_VIMEO90K)

    folderpaths = []

    for folder in folders:
        subfolders = os.listdir(os.path.join(FP_VIMEO90K, folder))

        for subfolder in subfolders:
            path = os.path.join(FP_VIMEO90K, folder, subfolder)

            folderpaths.append(path)
    
    return folderpaths


# def vimeo90k_generator(folderpaths=get_folderpaths_vimeo90k(), k=1, quadratic=False):
    
#     if quadratic:
#         image_ids = [2,3,5,6]
#         gt = 4
#     else:
#         image_ids = [3,5]
#         gt = 4
        
#     random.shuffle(folderpaths)
    
    
#     for folderpath in folderpaths:
#         # filepath = os.path.join(folderpath, 'im{0}.png')
#         filepath = os.path.join('E:/scriptieAI/created_datasets/vimeo_test_clean/sequences/00006/0043/', 'im{0}.png')
#         X = []
        
#         for ind in image_ids:
#             img = Image.open(filepath.format(ind))
#             img = torch.Tensor(np.array(img))
#             X.append(img)
#         img = Image.open(filepath.format(gt))
#         y = torch.Tensor(np.array(img))   
        
#         yield X, y


class vimeo90k_dataset(Dataset):

    def __init__(self, folderpaths=get_folderpaths_vimeo90k(), quadratic=False):#, transformer=None):
        self.folderpaths = folderpaths
        # self.transformer = transformer

        if quadratic:
            self.image_ids = [2,3,5,6]
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



class TransformedDataset(Dataset):

    def __init__(self,
                 dataset,
                 h_flip_prob = 0.1,
                 v_flip_prob = 0.1,
                 random_crop=True,
                 crop_size=(128,128),
                 channels_first=True,
                 normalize=True):

        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.dataset = dataset
        self.channels_first = channels_first
        self.normalize = normalize

    def __getitem__(self, idx):
        frames, y = self.dataset[idx]

        return self.apply(frames, y)

    def __len__(self):
        return self.dataset.__len__()


    def apply_tensor(self, X):
        assert X.ndim in [3,4]

        if X.ndim == 3:
            X = X.unsqueeze(dim=0)

        B, H, W, C = X.shape
        
        # horizontal flip
        h_flip, v_flip = np.random.rand(2) < [self.h_flip_prob, self.v_flip_prob]
        
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


    def apply(self, input_frames, label):
        '''
        expects input frames as list of tensors HxWxC and label seperately
        '''

        X = torch.cat([input_frames, label.unsqueeze(0)], axis=0)

        X = self.apply_tensor(X)

        if self.channels_first:
            X = X.permute(0,3,1,2)
        
        if self.normalize:
            X.div_(255.)
        
        input_frames, y = X[:-1], X[-1]
            
        return input_frames, y





class Transformer():
    
    def __init__(self, h_flip_prob = 0.1, v_flip_prob = 0.1, random_crop=True, crop_size=(128,128)):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.random_crop = random_crop
        self.crop_size = crop_size

    def apply_tensor(self, X):
        assert X.ndim in [3,4]

        if X.ndim == 3:
            X = X.unsqueeze(dim=0)

        B, H, W, C = X.shape

        
        # horizontal flip
        h_flip, v_flip = np.random.rand(2) < [self.h_flip_prob, self.v_flip_prob]
        
        dims_to_flip = []
        if h_flip: dims_to_flip.append(1)
        if v_flip: dims_to_flip.append(2)
        
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
        
    def apply(self, input_frames, label):
        '''
        expects input frames as list of tensors HxWxC and label seperately
        '''
        X = torch.stack([*input_frames, label], axis=0)
        
        X = self.apply_tensor(X)
            
        X = X.unbind(dim=0)
        
        input_frames, y = X[:-1], X[-1]
            
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

if __name__ == '__main__':

    files = get_adobe240_files()
    # print(files[0])
    gen = adobe240_generator(files)

    print(next(gen))