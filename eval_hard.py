import torch
import numpy as np
import dataloader
import metrics
from tqdm import tqdm
from collections import defaultdict
import imageio

import pandas as pd
import time
import os
import argparse


hard_vimeo = [1433, 3429, 7680, 3567, 2422, 3254, 2695]


OUTPUT_FOLDER = 'visual_comparison'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)




def evaluate(model_name):
    
    print(f'[{time.ctime()}] Start evaluating {model_name}')
    
    quadratic = 'quad' in model_name
    
    if model_name == 'sepconv-l1':
        import utilities
        model = utilities.get_sepconv(weights='l1').cuda()
    elif model_name == 'sepconv-lf':
        import utilities
        model = utilities.get_sepconv(weights='lf').cuda()
    elif model_name == 'qvi-lin' or model_name =='qvi-quad':
        from code.quadratic.interpolate import interpolate as model
    elif model_name == 'dain':
        from code.DAIN.interpolate import interpolate_efficient as model
    else:
        raise NotImplementedError()
        
    torch.manual_seed(42)
    np.random.seed(42)
    

    ds = dataloader.vimeo90k_dataset(quadratic=quadratic, fold='test')
    # ds = torch.utils.data.Subset(ds, indices=hard_vimeo)
    ds = dataloader.TransformedDataset(ds, normalize=True, random_crop=False, flip_probs=0)    
    
    _, _, test = dataloader.split_data(ds, [0, 0, 1])
    
    
    with torch.no_grad():
        for i,index in enumerate(tqdm(hard_vimeo)):
            X,_ = ds[index]
            X = X.cuda().unsqueeze(0).contiguous()
            
            y_hat = model(X).clamp(0,1)

            y_hat.mul_(255)
            img = y_hat.squeeze(0).permute(1,2,0).detach().cpu().numpy().astype('uint8')

            filepath_out = os.path.join(OUTPUT_FOLDER, '{}_{}.png'.format(model_name, index))
            imageio.imwrite(filepath_out, img)


        
        
    
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str)

    FLAGS, unparsed = parser.parse_known_args()

    assert FLAGS.model in ['sepconv-l1', 'sepconv-lf', 'qvi-lin', 'qvi-quad', 'dain']
    
    evaluate(FLAGS.model)
    

