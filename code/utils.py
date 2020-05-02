import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from tqdm import tqdm
from PIL import Image
import re

# todo
# v datagenerator (batch_size)
# v datacreator 
#   train/valid/test split
#   seed?


def generator(folder, batch_size = 4, k=None):
    
    raw_files = os.listdir(folder)

    # infer k from files
    if k == None:
        ks = [int(re.findall(r'.+\.mp4_\d+_(\d+)\.png',f)[0]) for f in raw_files]
        k = max(ks)-1

        assert len(raw_files) % (k+2) == 0, f'not all generated datapoints of equal length, {len(raw_files), k}'

    files = [
        [raw_files[i+j] for i in range(3)] for j in range(len(raw_files)-2)
    ]        
    
    files = files[:1]
    print(files)
    random.shuffle(files)

    n_batches = np.ceil(len(files)/batch_size).astype(int)

    for b in range(n_batches):

        batch = files[b*batch_size:(b+1)*batch_size]
        batch_result = []
        for frameset in batch:

            imgs = []
            for file in frameset:
                img = np.array(Image.open(os.path.join(folder, file)))
                imgs.append(img)
            imgs = np.stack(imgs)
            batch_result.append(imgs)

        yield np.stack(batch_result)



def train_valid_test_split(folderpath):
    pass



def data_creator(folderpath_in, folderpath_out, k=1):
    '''
    Goes through all files in 'folderpath_in' and 
    creates random sequences of frames of length k+2

    k: number of interpolated frames

    '''

    for file in tqdm(os.listdir(folderpath_in)):
        # load video
        cap = cv2.VideoCapture(os.path.join(folderpath_in, file))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        domain = [[i+j for j in range(k+2)] for i in range(n_frames-k-1)]
        to_sample = []

        while len(domain) > 0:
            datapoint = random.choice(domain)
            domain = [x for x in domain if sum([x.count(y) for y in datapoint]) == 0]

            to_sample.append(datapoint)

        random.shuffle(to_sample)

        for i, datapoint in enumerate(to_sample):
            frames = _get_frames_by_indices(cap, datapoint)

            for j, frame in enumerate(frames):
                filepath_out = os.path.join(folderpath_out, f'{file}_{i}_{j}.png')
                frame = np.squeeze(frame)
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                im = Image.fromarray(frame)
                im.save(filepath_out)



def _get_frames_by_indices(cap, indices):
    frames = []
    
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        _, image = cap.read()
        frames.append(image)
        
    return frames



# data_creator('E:\scriptieAI\demo_data', 'E:\scriptieAI\output_data')
print(next(generator('E:\scriptieAI\output_data')).shape)