import torch
import dataloader
import os
from PIL import Image
from torchvision import transforms, utils
import numpy as np
import metrics
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import utilities
import time


# load model
checkpoint= torch.load('models_test/extra_folder/296_checkpoint_1594777175_seed_0_lr=0.001_lr2=0.0001_loss=l1_pretrain=1_kq_d_size=45_kq_d_scale=2_kq_size=45_optimizer=adamax_input_size=4_kl_size=51_kl_d_size=45_kl_d_scale=2')
model = checkpoint['last_model'].cuda().eval()


DATA_DIR = 'Created_datasets/MiddleBury'
OUTPUT_FOLDER = os.path.join(DATA_DIR, 'output_test_3')
to_tensor = transforms.ToTensor()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


sequences = [
    'Mequon',
    'Schefflera',
    'Urban',
    'Teddy',
    'Backyard',
    'Basketball',
    'Dumptruck',
    'Evergreen'
]

with torch.no_grad():
    for seq in sequences: # 
        t0 = time.time()
                   
        
        folderpath = os.path.join(DATA_DIR, seq)
        
        if seq == 'Teddy':
            im1 = to_tensor(Image.open(os.path.join(folderpath, 'frame10.png')))
            im2 = to_tensor(Image.open(os.path.join(folderpath, 'frame11.png')))
            X = torch.stack([im1, im2])
            
            X = torch.cat([
                torch.zeros(1,3,360,420),
                X,
                torch.zeros(1,3,360,420)                
            ], axis=0).unsqueeze(0).cuda()

            
        else:
            
            im1 = to_tensor(Image.open(os.path.join(folderpath, 'frame09.png')))
            im2 = to_tensor(Image.open(os.path.join(folderpath, 'frame10.png')))
            im3 = to_tensor(Image.open(os.path.join(folderpath, 'frame11.png')))
            im4 = to_tensor(Image.open(os.path.join(folderpath, 'frame12.png')))
            
            X = torch.stack([im1, im2, im3, im4]).unsqueeze(0).cuda()
            
            
        y_hat = model(X).clamp(0,1).mul(255).cpu().detach().int().squeeze(0).permute(1,2,0)
        y_hat = y_hat.numpy().astype(np.uint8)
        print(seq)
        t1 = time.time()
        output_folder_sequence = os.path.join(OUTPUT_FOLDER, seq)
        os.makedirs(output_folder_sequence, exist_ok=True)
        imageio.imwrite(os.path.join(output_folder_sequence, f'frame10i11.png'), im=y_hat)
        
        if seq=='Urban':
            print(f'Running time Urban: {t1-t0} seconds')
        

