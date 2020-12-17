from code.sepconvfull.model_extended_efficient import SepConvNetExtended
import dataloader
import torch
from torch.utils.tensorboard import SummaryWriter 
import discriminator
from tqdm import tqdm
from collections import defaultdict
import metrics
import numpy as np
import hyperopt
from hyperopt import hp, space_eval, Trials, fmin, STATUS_OK, tpe
import time
from itertools import product
import os
from utilities import ResultStore, EarlyStopping, get_sepconv
import argparse
import losses
from constants import MODEL_FOLDER
import utilities
import time
from pytorch_ranger import Ranger
import sys
import pandas as pd
# from code.sepconvfull.model import SepConvNetExtended
import random
import pickle

os.makedirs(MODEL_FOLDER, exist_ok=True)



FOLDS = [
    'train_fold', 'valid_vimeo',
    'valid_lmd', 'test_vimeo', 'test_disp', 'test_nonlin'
]

def visual_evaluation_vimeo(model, quadratic, writer, epoch):
    gen = dataloader.generate_hard_vimeo(quadratic=quadratic)

    y_hats = []
    ys = []
    for X, y in gen:
        X = X.cuda() / 255.
        y = y / 255.
        
        y_hat = model(X).detach().cpu().squeeze(dim=0)
        y_hats.append(y_hat)
        ys.append(y)

    y_hats = torch.stack(y_hats).clamp(0,1)
    ys = torch.stack(ys)

    # create image grid
    grid = utilities.create_grid(y_hats, ys, padding=20, nrow=7)

    writer.add_image('hard_examples_vimeo', img_tensor=grid, global_step=epoch)

def visual_evaluation(model, quadratic, writer, epoch):
    gen = dataloader.generate_hard_input(quadratic=quadratic)

    y_hats = []
    ys = []
    for X, y in gen:
        X = X.cuda() / 255.
        y = y / 255.
        
        y_hat = model(X).detach().cpu().squeeze(dim=0)
        y_hats.append(y_hat)
        ys.append(y)

    y_hats = torch.stack(y_hats).clamp(0,1)
    ys = torch.stack(ys)

    # create image grid
    grid = utilities.create_grid(y_hats, ys, padding=20, nrow=4)

    writer.add_image('hard_examples', img_tensor=grid, global_step=epoch)



def train(params, n_epochs, verbose=True):
    # init interpolation model
    timestamp = int(time.time())
    formatted_params = '_'.join(f'{k}={v}' for k,v in params.items())

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)


    G = SepConvNetExtended(
        kl_size=params['kl_size'], 
        kq_size=params['kq_size'],
        kl_d_size=params['kl_d_size'], 
        kl_d_scale=params['kl_d_scale'],
        kq_d_scale=params['kq_d_scale'],
        kq_d_size=params['kq_d_size'],
        input_frames=params['input_size']
    )
        
        
    if params['pretrain'] in [1,2]:
        print('LOADING L1')
        G.load_weights('l1')
    
    name = f'{timestamp}_seed_{FLAGS.seed}_{formatted_params}'
    G = torch.nn.DataParallel(G).cuda()

    # optimizer = torch.optim.Adamax(G.parameters(), lr=params['lr'], betas=(.9, .999))
    if params['optimizer'] == 'ranger':
        optimizer = Ranger([
            {'params': [p for l,p in G.named_parameters() if 'moduleConv' not in l]},
            {'params': [p for l,p in G.named_parameters() if 'moduleConv' in l], 'lr': params['lr2']}
        ], lr=params['lr'], betas=(.95, .999))

    elif params['optimizer'] == 'adamax':
        optimizer = torch.optim.Adamax([
            {'params': [p for l,p in G.named_parameters() if 'moduleConv' not in l]},
            {'params': [p for l,p in G.named_parameters() if 'moduleConv' in l], 'lr': params['lr2']}
        ], lr=params['lr'], betas=(.9, .999))

    else:
        raise NotImplementedError()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-FLAGS.warmup, eta_min=3e-5, last_epoch=-1)
    start_epoch = 0

    print('SETTINGS:')
    print(params)
    print('NAME:')
    print(name)
    sys.stdout.flush()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    quadratic = params['input_size'] == 4
    
    L1_loss = torch.nn.L1Loss()

    ds_train_lmd = dataloader.large_motion_dataset(quadratic = quadratic, cropped=True, fold='train', min_flow=6)
    ds_valid_lmd = dataloader.large_motion_dataset(quadratic = quadratic, cropped=True, fold='valid')
    
    ds_vimeo_train = dataloader.vimeo90k_dataset(fold='train', quadratic = quadratic)
    ds_vimeo_test = dataloader.vimeo90k_dataset(fold='test', quadratic = quadratic)

    # train, test_lmd = dataloader.split_data(ds_lmd, [.9, .1])
    train_vimeo, valid_vimeo = dataloader.split_data(ds_vimeo_train, [.9, .1])

    train_settings = {
        'flip_probs':FLAGS.flip_probs,
        'normalize':True,
        'crop_size':(params['crop_size'], params['crop_size']),
        'jitter_prob':FLAGS.jitter_prob,
        'random_rescale_prob':FLAGS.random_rescale_prob
        # 'rescale_distr':(.8, 1.2),
        
    }

    valid_settings = {
        'flip_probs':0,
        'random_rescale_prob':0,
        'random_crop':False,
        'normalize':True
    }

    train_lmd = dataloader.TransformedDataset(ds_train_lmd, **train_settings)
    valid_lmd = dataloader.TransformedDataset(ds_valid_lmd, **valid_settings)

    train_vimeo = dataloader.TransformedDataset(train_vimeo, **train_settings)
    valid_vimeo = dataloader.TransformedDataset(valid_vimeo, **valid_settings)
    test_vimeo  = dataloader.TransformedDataset(ds_vimeo_test, **valid_settings)

    train_data = torch.utils.data.ConcatDataset([train_lmd, train_vimeo])

    

    # displacement
    df = pd.read_csv(f'hardinstancesinfo/vimeo90k_test_flow.csv')
    test_disp = torch.utils.data.Subset(ds_vimeo_test, indices = df[df.mean_manh_flow>=df.quantile(.9).mean_manh_flow].index.tolist())
    test_disp = dataloader.TransformedDataset(test_disp, **valid_settings)
    test_disp   = torch.utils.data.DataLoader(test_disp, batch_size=4, pin_memory=True)

    # nonlinearity
    df = pd.read_csv(f'hardinstancesinfo/Vimeo90K_test.csv')
    test_nonlin = torch.utils.data.Subset(ds_vimeo_test, indices = df[df.non_linearity>=df.quantile(.9).non_linearity].index.tolist())
    test_nonlin = dataloader.TransformedDataset(test_nonlin, **valid_settings)
    test_nonlin = torch.utils.data.DataLoader(test_nonlin, batch_size=4, pin_memory=True)

    # create weights for train sampler
    df_vim = pd.read_csv(f'hardinstancesinfo/vimeo90k_train_flow.csv')
    weights_vim = df_vim[df_vim.index.isin(train_vimeo.dataset.indices)].mean_manh_flow.tolist()
    weights_lmd = ds_train_lmd.weights
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_lmd+weights_vim, FLAGS.num_train_samples, replacement=False)

    

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, pin_memory=True, shuffle=False, sampler=train_sampler, num_workers=FLAGS.num_workers)
    valid_dl_vim = torch.utils.data.DataLoader(valid_vimeo, batch_size=4, pin_memory=True, num_workers=FLAGS.num_workers)
    valid_dl_lmd = torch.utils.data.DataLoader(valid_lmd, batch_size=4, pin_memory=True, num_workers=FLAGS.num_workers)
    test_dl_vim = torch.utils.data.DataLoader(test_vimeo, batch_size=4, pin_memory=True, num_workers=FLAGS.num_workers)

    # metrics    
    writer = SummaryWriter(f'runs/final_exp/full_run_hyperopt/{name}')
    
    results = ResultStore(
        writer=writer,
        metrics = ['psnr', 'ssim', 'ie', 'L1_loss'],
        folds=FOLDS
    )

    

    early_stopping_metric = 'L1_loss'
    early_stopping = EarlyStopping(results, patience=FLAGS.patience, metric=early_stopping_metric, fold='valid_vimeo')

    def do_epoch(dataloader, fold, epoch, train=False):
        assert fold in FOLDS

        if verbose:
            pb = tqdm(desc=f'{fold} {epoch+1}/{n_epochs}', total=len(dataloader), leave=True, position=0)
        
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()    

            y_hat = G(X)

            l1_loss = L1_loss(y_hat, y)

            if train:                
                optimizer.zero_grad()
                l1_loss.backward()
                optimizer.step()            
            

            # compute metrics
            y_hat = (y_hat * 255).clamp(0,255)
            y = (y * 255).clamp(0,255)

            psnr = metrics.psnr(y_hat, y)
            ssim = metrics.ssim(y_hat, y)
            ie = metrics.interpolation_error(y_hat, y)
                   
            results.store(fold, epoch, {
                'L1_loss':l1_loss.item(),
                'psnr':psnr,
                'ssim': ssim,
                'ie':ie
            })
            
            if verbose: pb.update()       

        # update tensorboard
        results.write_tensorboard(fold, epoch)
        sys.stdout.flush()




    start_time = time.time()
    for epoch in range(start_epoch, n_epochs):

        G.train()
        do_epoch(train_dl, 'train_fold', epoch, train=True)
        

        if epoch >= FLAGS.warmup-1:
            scheduler.step()   

        G.eval()
        with torch.no_grad():
            do_epoch(valid_dl_vim, 'valid_vimeo', epoch)
            do_epoch(valid_dl_lmd, 'valid_lmd', epoch)
            

        if (early_stopping.stop() and epoch >= FLAGS.min_epochs) or epoch % FLAGS.test_every == 0 or epoch+1 == n_epochs:
            with torch.no_grad():
                do_epoch(test_disp, 'test_disp', epoch)
                do_epoch(test_nonlin, 'test_nonlin', epoch)
                
                do_epoch(test_dl_vim, 'test_vimeo', epoch)

            visual_evaluation(
                model=G,
                quadratic = params['input_size'] ==4, 
                writer=writer,
                epoch=epoch
            )

            visual_evaluation_vimeo(
                model=G,
                quadratic = params['input_size'] ==4, 
                writer=writer,
                epoch=epoch
            )

        # save model if new best
        if early_stopping.new_best():
            filepath_out = os.path.join(MODEL_FOLDER, '{0}_{1}')
            torch.save(G, filepath_out.format('generator', name))
            
        # save last model state
        checkpoint = {'last_model':G, 'epoch':epoch, 'optimizer':optimizer.state_dict(), 'name':name, 'scheduler':scheduler.state_dict()}
        torch.save(checkpoint, filepath_out.format('checkpoint', name))

        if early_stopping.stop() and epoch >= FLAGS.min_epochs:
            break

        torch.cuda.empty_cache()
    
            
    end_time = time.time()
    # free memory
    del G
    torch.cuda.empty_cache()
    time_elapsed = end_time - start_time
    print(f'Ran {n_epochs} epochs in {round(time_elapsed, 1)} seconds')
    
    return results


def objective(params):
    '''Minimize loss on validation set'''
    
    results = train(params, n_epochs=FLAGS.n_epochs, verbose=True)
    losses = results.results['valid_vimeo']['L1_loss']
    # max_epoch = np.max(list(losses.keys()))
    mean_losses = [np.mean(losses[epoch]) for epoch in losses.keys()]
    best_validation_loss = np.min(mean_losses)

    return best_validation_loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--n_epochs', type = int, default = 40)
    parser.add_argument('--min_epochs', type = int, default = 30)
    parser.add_argument('--patience', type = int, default = 10)
    parser.add_argument('--test_every', type = int, default = 5)
    parser.add_argument('--num_train_samples', type = int, default = 50000)
    parser.add_argument('--num_workers', type = int, default = 5)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--jitter_prob', type = float, default = .2)
    parser.add_argument('--random_rescale_prob', type = float, default = 0)
    parser.add_argument('--flip_probs', type = float, default = .5)
    parser.add_argument('--warmup', type = int, default = 20)
    

    FLAGS, unparsed = parser.parse_known_args()

    paramer_space = {
        'lr': 1e-3,
        'lr2': 1e-4,
        'loss':'l1',
        'pretrain':1,
        'kl_d_size':25, 
        'kl_d_scale':4,
        'kq_d_size':25, 
        'kq_d_scale':4,
        'kq_size':None,
        'input_size': 4,
        'crop_size':hp.choice('crop_size', [128,150,160]),
        'optimizer':hp.choice('optimizer', ['adamax', 'ranger']),
        'kl_size':hp.choice('kl_size', [41, 45, 51])
    }

    trials = Trials()

    optimized = fmin(
        fn = objective,
        space = paramer_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=10,
        rstate= np.random.RandomState(FLAGS.seed)
    )

    print('BEST PARAMETER SETTINGS:')
    print(space_eval(paramer_space, optimized))
    
    with open('results/hp_opt_trials.pickle', 'wb') as f:
        pickle.dump(trials, f)




    
    
