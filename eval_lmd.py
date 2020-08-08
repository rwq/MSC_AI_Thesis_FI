import torch
import numpy as np
import dataloader
import metrics
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import time
import os
import argparse





OUTPUT_FOLDER = 'results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)




def evaluate(model_name, dataset, quadratic=False):
    
    print(f'[{time.ctime()}] Start evaluating {model_name} on {dataset}')
    
    # quadratic = 'quad' in model_name
    
    if model_name == 'sepconv':
        import utilities
        model = utilities.get_sepconv(weights='l1').cuda()
    elif model_name == 'qvi-lin' or model_name =='qvi-quad':
        from code.quadratic.interpolate import interpolate as model
    elif model_name == 'dain':
        from code.DAIN.interpolate import interpolate_efficient as model
    elif model_name == 'sepconv2':
        checkpoint= torch.load('models/checkpoint_1593886534_seed_0_optimizer=adamax_input_size=4_lr=0.001_lr2=0.0001_weights=None_kernel_size=45_loss=l1_pretrain=1_kernel_size_d=31_kernel_size_scale=4_kernel_size_qd=25_kernel_size_qd_scale=4')
        model = checkpoint['last_model'].cuda().eval()
    else:
        raise NotImplementedError()
        
    torch.manual_seed(42)
    np.random.seed(42)
    # model = model.eval()
    
    results = defaultdict(list)
    if dataset == 'lmd':
        ds = dataloader.large_motion_dataset2(quadratic=quadratic, fold='test', cropped=False)
    elif dataset == 'adobe240':
        ds = dataloader.adobe240_dataset(quadratic=quadratic, fold='test')
    elif dataset == 'gopro':
        ds = dataloader.gopro_dataset(quadratic=quadratic, fold='test')
    elif dataset == 'vimeo90k':
        ds = dataloader.vimeo90k_dataset(quadratic=quadratic, fold='test')
    else:
        raise NotImplementedError()

    ds = dataloader.TransformedDataset(ds, normalize=True, random_crop=False, flip_probs=0)    
    
    _, _, test = dataloader.split_data(ds, [0, 0, 1])
    
    data_loader = torch.utils.data.DataLoader(test, batch_size=1)
    with torch.no_grad():
        for X,y in tqdm(data_loader, total=len(data_loader)):
            X = X.cuda()
            y = y.cuda()
            
            y_hat = model(X).clamp(0,1)
            
            y.mul_(255)
            y_hat.mul_(255)
            
            results['psnr'].extend(metrics.psnr(y_hat, y))
            results['ie'].extend(metrics.interpolation_error(y_hat, y))
            results['ssim'].extend(metrics.ssim(y_hat, y))
    
    # store in dataframe
    results = pd.DataFrame(results)
    results['model'] = model_name
    results['dataset'] = dataset
    
    return results
        
    
    
    
    
    
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--model', required=True, type=str)
    # parser.add_argument('--dataset', default='lmd', type=str)

    # FLAGS, unparsed = parser.parse_known_args()

    # assert FLAGS.model in ['sepconv', 'qvi-lin', 'qvi-quad', 'dain']
    
    for model in ['dain']:#, 'qvi-lin', 'qvi-quad']:
        for dataset in ['gopro']:
            results = evaluate(model, dataset, True)
            
            filepath_out = os.path.join(OUTPUT_FOLDER, f'results_{dataset}_{model}_eval.csv')
            results.to_csv(filepath_out, index=None)

