from code.sepconvfull import model
import dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
import discriminator
from tqdm import tqdm
from collections import defaultdict
import metrics
import numpy as np
import hyperopt
from hyperopt import hp, space_eval, Trials
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


os.makedirs(MODEL_FOLDER, exist_ok=True)




FOLDS = [
    'train_adobe', 'valid_adobe', 'test_adobe'
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




def objective(params):
    '''Minimize loss on validation set'''
    
    _, _, resultstore = train(params, n_epochs=5, verbose=False)
    
    losses = resultstore.results['valid']['L1_loss']
    max_epoch = np.max(list(losses.keys()))
    validation_loss = np.mean(losses[max_epoch])
    
    return validation_loss


def train(params, n_epochs, verbose=True):

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # init interpolation model
    if FLAGS.filename == None:        
        G = get_sepconv(input_size=params['input_size'], weights=params["weights"], kernel_size=params['kernel_size'], random_output_kernel=True)
        G = G.cuda()

        # optimizer = Ranger(G.parameters(), lr=params['lr'])
        optimizer = torch.optim.Adam(G.parameters(), lr=params['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-FLAGS.warmup)
        start_epoch = 0
        name = f'{int(time.time())}_lr_{params["lr"]}_F_{params["input_size"]}_K_{params["kernel_size"]}_{params["weights"]}_loss_{params["loss"]}'
    else:
        checkpoint = torch.load(FLAGS.filename)
        G = checkpoint['last_model'].cuda()
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        start_epoch = checkpoint['epoch']+1
        name = checkpoint['name']

        # optimizer = Ranger(G.parameters())
        optimizer = torch.optim.Adam(G.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])


    
    loss_network = losses.LossNetwork(layers=[9,16,26]).cuda() #9, 16, 26
    Perc_loss = losses.PerceptualLoss(loss_network, epoch=start_epoch, scheme=params['loss'], n_epochs=n_epochs, include_input=True)
    L1_loss = torch.nn.L1Loss()

    quadratic = params['input_size'] == 4

    ds_adobe = dataloader.adobe240_dataset(quadratic = quadratic)
    # ds_lmd   = dataloader.large_motion_dataset(quadratic = quadratic)
    # ds_vimeo_train = dataloader.vimeo90k_dataset(fold='train', quadratic = quadratic)

    train_adb, valid_adb, test_adb = dataloader.split_data(ds_adobe, [.8, .1, .1])
    # train_lmd, valid_lmd, test_lmd = dataloader.split_data(ds_lmd, [.8, .1, .1])
    # train_vimeo, valid_vimeo, test_vimeo = dataloader.split_data(ds_vimeo_train, [.95, .025, .025])

    train_adb = dataloader.TransformedDataset(train_adb, crop_size=(FLAGS.crop_size,FLAGS.crop_size), normalize=True)
    valid_adb = dataloader.TransformedDataset(valid_adb, crop_size=(480,480), normalize=True)
    test_adb  = dataloader.TransformedDataset(test_adb, random_crop=False, normalize=True)
    
    # train_lmd = dataloader.TransformedDataset(train_lmd, crop_size=(FLAGS.crop_size,FLAGS.crop_size), normalize=True)
    # valid_lmd = dataloader.TransformedDataset(valid_lmd, crop_size=(400,400), normalize=True)
    # test_lmd  = dataloader.TransformedDataset(test_lmd, random_crop=False, normalize=True)

    # train_vimeo = dataloader.TransformedDataset(train_vimeo, crop_size=(FLAGS.crop_size,FLAGS.crop_size), normalize=True)
    # valid_vimeo = dataloader.TransformedDataset(valid_vimeo, random_crop=False, normalize=True)
    # test_vimeo  = dataloader.TransformedDataset(test_vimeo, random_crop=False, normalize=True)


    # test_disp   = dataloader.get_hard_instances_subset('Adobe240', measure='max_disp', quantile=0.9, indices=test_adb.dataset.indices, quadratic=quadratic)
    # test_nonlin = dataloader.get_hard_instances_subset('Adobe240', measure='non_linearity', quantile=0.9, indices=test_adb.dataset.indices, quadratic=quadratic)
    # test_disp = dataloader.TransformedDataset(test_disp, random_crop=False, flip_probs=(0,0,0), normalize=True)
    # test_nonlin = dataloader.TransformedDataset(test_nonlin, random_crop=False, flip_probs=(0,0,0), normalize=True)
    # test_disp   = torch.utils.data.DataLoader(test_disp, batch_size=2, pin_memory=True)
    # test_nonlin = torch.utils.data.DataLoader(test_nonlin, batch_size=2, pin_memory=True)

    # create weights for train sampler
    # df = pd.read_csv(f'hardinstancesinfo/Adobe240.csv')
    # df.non_linearity = df.non_linearity.fillna(1)
    # weights = (df.mean_disp + np.sqrt(df.non_linearity))[train_adb.dataset.indices].tolist()

    # adb_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, FLAGS.num_train_samples, replacement=False)
    # lmd_sampler = torch.utils.data.RandomSampler(train_lmd, replacement=True, num_samples = FLAGS.num_train_samples)
    # vim_sampler = torch.utils.data.RandomSampler(train_vimeo, replacement=True, num_samples = FLAGS.num_train_samples_vimeo)

    train_dl_adb = torch.utils.data.DataLoader(train_adb, batch_size=FLAGS.batch_size, pin_memory=True, shuffle=False)#, sampler=adb_sampler)
    valid_dl_adb = torch.utils.data.DataLoader(valid_adb, batch_size=4, pin_memory=True)
    test_dl_adb  = torch.utils.data.DataLoader(test_adb, batch_size=2, pin_memory=True)

    # train_dl_lmd = torch.utils.data.DataLoader(train_lmd, batch_size=FLAGS.batch_size, pin_memory=True, shuffle=False, sampler=lmd_sampler)
    # valid_dl_lmd = torch.utils.data.DataLoader(valid_lmd, batch_size=4, pin_memory=True)
    # test_dl_lmd = torch.utils.data.DataLoader(test_lmd, batch_size=2, pin_memory=True)

    # train_dl_vim = torch.utils.data.DataLoader(train_vimeo, batch_size=FLAGS.batch_size, pin_memory=True, shuffle=False, sampler=vim_sampler)
    # valid_dl_vim = torch.utils.data.DataLoader(valid_vimeo, batch_size=4, pin_memory=True)
    # test_dl_vim = torch.utils.data.DataLoader(test_vimeo, batch_size=2, pin_memory=True)

    # metrics    
    writer = SummaryWriter(f'runs/experiment_loss_adobe2/{name}')
    
    results = ResultStore(
        writer=writer,
        metrics = ['psnr', 'ssim', 'ie', 'L1_loss', 'mspl', *[f'mspl_{i}' for i in range(4)]],
        folds=FOLDS
    )

    early_stopping_metric = 'mspl_2' if params['weights'] == 'lf' else 'L1_loss'
    early_stopping = EarlyStopping(results, patience=FLAGS.patience, metric=early_stopping_metric, fold='valid_adobe')

    def do_epoch(dataloader, fold, epoch):
        assert fold in FOLDS

        if verbose:
            pb = tqdm(desc=f'{fold} {epoch+1}/{n_epochs}', total=len(dataloader), leave=True, position=0)
        
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()    

            y_hat = G(X)

            # compute losses
            loss, losses = Perc_loss(y_hat, y)
            l1_loss = L1_loss(y_hat, y)

            if 'train' in fold:
                optimizer.zero_grad()

                if params['loss'] == 'lf':
                    losses[-1].backward()
                elif params['loss'] == 'l1':
                    l1_loss.backward()
                elif 'lp' in params['loss']:
                    loss.backward()
                else:
                    raise NotImplementedError()

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
                'ie':ie,
                'mspl':loss.item(),
                **{f'mspl_{i}':losses[i].item() for i in range(4)}
            })
            
            if verbose: pb.update()       

        # update tensorboard
        results.write_tensorboard(fold, epoch)




    start_time = time.time()
    for epoch in range(start_epoch, n_epochs):
        
        G.train()

        print(f'epoch: {epoch}, alphas: {Perc_loss.weights}')

        sys.stdout.flush()
        do_epoch(train_dl_adb, 'train_adobe', epoch)
        # do_epoch(train_dl_lmd, 'train_lmd', epoch)
        # do_epoch(train_dl_vim, 'train_vimeo', epoch)
        
        if epoch >= FLAGS.warmup-1:
            scheduler.step()    

        G.eval()
        with torch.no_grad():
            do_epoch(valid_dl_adb, 'valid_adobe', epoch)
            # do_epoch(valid_dl_lmd, 'valid_lmd', epoch)
            # do_epoch(valid_dl_vim, 'valid_vimeo', epoch)

        if early_stopping.stop() or epoch % FLAGS.test_every == 0 or epoch+1 == n_epochs:
            with torch.no_grad():
                do_epoch(test_dl_adb, 'test_adobe', epoch)
                # do_epoch(test_dl_lmd, 'test_lmd', epoch)
                # do_epoch(test_dl_vim, 'test_vimeo', epoch)
                # do_epoch(test_disp, 'test_disp', epoch)
                # do_epoch(test_nonlin, 'test_nonlin', epoch)

            visual_evaluation(
                model=G,
                quadratic = params['input_size'] ==4, 
                writer=writer,
                epoch=epoch
            )

        Perc_loss.next_epoch()

        # save model if new best
        if early_stopping.new_best():
            filepath_out = os.path.join(MODEL_FOLDER, '{0}_{1}')
            torch.save(G, filepath_out.format('generator', name))
            
        # save last model state
        checkpoint = {'last_model':G, 'epoch':epoch, 'optimizer':optimizer.state_dict(), 'scheduler':scheduler, 'name':name}
        torch.save(checkpoint, filepath_out.format('checkpoint', name))

        if early_stopping.stop():
            break
    
            
    end_time = time.time()
    # free memory
    del G
    torch.cuda.empty_cache()
    time_elapsed = end_time - start_time
    print(f'Ran {n_epochs} epochs in {round(time_elapsed, 1)} seconds')
    
    return results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hp_ind', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--crop_size', type = int, default = 150)
    parser.add_argument('--n_epochs', type = int, default = 150)
    parser.add_argument('--patience', type = int, default = 50)
    parser.add_argument('--test_every', type = int, default = 10)
    parser.add_argument('--warmup', type = int, default = 5)
    parser.add_argument('--num_train_samples', type = int, default = 4000)
    parser.add_argument('--num_train_samples_vimeo', type = int, default = 20000)
    parser.add_argument('--filename', type = str, default = None)
    parser.add_argument('--seed', type = int, default = 42)


    FLAGS, unparsed = parser.parse_known_args()

    # parameter options for input size = 2
    parameter_space = {
         'input_size': [2],
         'lr': [1e-4],
         'weights': [None],
         'kernel_size': [51],
         'loss':['l1', 'lf', 'lp-constant', 'lp-cosine', 'lp-random']
    }

    param_combinations = product(*parameter_space.values())
    param_combinations = [dict(zip(parameter_space, values)) for values in param_combinations]
    
    # for p in param_combinations:
    #     print(p)

    parameters = param_combinations[FLAGS.hp_ind]
    print(time.ctime(), parameters)
    sys.stdout.flush()
    
    results = train(
        parameters,
        n_epochs=FLAGS.n_epochs,
        verbose=True
    )

    