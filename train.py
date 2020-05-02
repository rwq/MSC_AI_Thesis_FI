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



MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)


torch.manual_seed(42)
np.random.seed(42)


def objective(params):
    '''Minimize loss on validation set'''
    
    _, _, resultstore = train(params, n_epochs=5, verbose=False)
    
    losses = resultstore.results['valid']['L1_loss']
    max_epoch = np.max(list(losses.keys()))
    validation_loss = np.mean(losses[max_epoch])
    
    return validation_loss


def train(params, n_epochs, verbose=True):




    
    # init interpolation model
    G = get_sepconv(input_size = params['input_size'])
    G = G.cuda()
    G = G.eval()

    # init discriminator
    D = discriminator.Discriminator(input_size=params['input_size']).cuda()
    
    ds = dataloader.adobe240_dataset( quadratic = params['input_size'] == 4 )
    
    N_train = int(len(ds) * 0.8)
    N_valid = int(len(ds) * 0.1)
    N_test = len(ds) - N_train - N_valid

    train, valid, test = torch.utils.data.random_split(ds, [N_train, N_valid, N_test])

    train = dataloader.TransformedDataset(train, crop_size=(FLAGS.crop_size,FLAGS.crop_size), normalize=True)
    valid = dataloader.TransformedDataset(valid, crop_size=(FLAGS.crop_size,FLAGS.crop_size), normalize=True)
    test  = dataloader.TransformedDataset(test, random_crop=False, normalize=True)

    train_dl = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, pin_memory=True, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, pin_memory=True)
    test_dl  = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, pin_memory=True)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])#$, amsgrad=params['amsgrad'])
    optimizer_D = torch.optim.Adam(D.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])#, amsgrad=params['amsgrad'])
    critereon = torch.nn.L1Loss()

    # metrics
    name = f'{int(time.time())}_{params["lr"]}_{params["weight_decay"]}_{params["wgan"]}_{params["input_size"]}'
    writer = SummaryWriter(f'runs/{name}') #TODO hp
    
    results = ResultStore(
        writer=writer,
        metrics = ['psnr', 'ie', 'L1_loss', 'accuracy', 'G_loss', 'D_loss'],
        folds=['train', 'valid', 'test']
    )

    early_stopping = EarlyStopping(results, patience=FLAGS.patience, metric='L1_loss')
    



    def do_epoch(dataloader, fold, epoch):
        assert fold in ['train', 'valid', 'test']

        if verbose:
            pb = tqdm(desc=f'{fold} {epoch+1}/{FLAGS.n_epochs}', total=len(dataloader), leave=True, position=0)
        
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()    
            # train generator      

            y_hat = G(X)
            l1_loss = critereon(y_hat, y)
            
            
            loss = l1_loss - D(X, y_hat).sigmoid().log().mean()
            # generator loss
            if params['wgan']:
                G_loss = l1_loss - D(X, y_hat).mean()
            else:
                G_loss = l1_loss - D(X, y_hat).sigmoid().log().mean()
                
            if fold == 'train':
                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

            # train discriminator
            y_hat = y_hat.detach()
            
            
            if params['wgan']:
                D_loss = torch.mean( D(X, y_hat) - D(X, y) )
            else:
                D_loss = -torch.log(1 - D(X, y_hat).sigmoid()).mean() - D(X, y).sigmoid().log().mean()
            
            
            # compute psnr
            y_hat = (y_hat * 255).clamp(0,255)
            y = (y * 255).clamp(0,255)

            psnr = metrics.psnr(y_hat, y)
            ie = metrics.interpolation_error(y_hat, y)
            
            
            correct_preds = (D(X, y_hat).sigmoid().round() == 0).flatten().int().detach().cpu().tolist()
            correct_preds.extend((D(X, y).sigmoid().round() == 1).flatten().int().detach().cpu().tolist())
          
            results.store(fold, epoch, {
                    'L1_loss':l1_loss.item(),
                    'psnr':psnr,
                    'ie':ie,
                    'accuracy':correct_preds,
                    'D_loss':D_loss.item(),
                    'G_loss':G_loss.item()
                })

            if fold == 'train':
                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()
                
                if params['wgan']:
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)
            
            if verbose: pb.update()                

            if i == 50:
                break

        # update tensorboard
        results.write_tensorboard(fold, epoch)





    for epoch in range(FLAGS.n_epochs):

        G.train()
        D.train()
        
        do_epoch(train_dl, 'train', epoch)               

        G.eval()
        D.eval()

        with torch.no_grad():
            do_epoch(valid_dl, 'valid', epoch)


        if early_stopping.stop() or epoch % FLAGS.test_every == 0:
            with torch.no_grad():
                do_epoch(test_dl, 'test', epoch)
        
        
        # save model if new best
        if early_stopping.new_best():
            filepath_out = os.path.join(MODEL_FOLDER, '{0}_{1}')
            torch.save(G, filepath_out.format('generator', name))
            torch.save(D, filepath_out.format('discriminator', name))
            
        
        if early_stopping.stop():
            break
    
            
            
    # free memory
    del G
    del D
    torch.cuda.empty_cache()
    
    return results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hp_start', type = int, default = 0)
    parser.add_argument('--hp_end', type = int)
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--crop_size', type = int, default = 128)
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--patience', type = int, default = 10)
    parser.add_argument('--test_every', type = int, default = 5)



    FLAGS, unparsed = parser.parse_known_args()

    parameter_space = {
         'input_size': [2, 4],
         'lr': [1e-5, 1e-4],
         'weight_decay': [0],
         'wgan': [True, False],   
    }

    param_combinations = product(*parameter_space.values())
    param_combinations = [dict(zip(parameter_space, values)) for values in param_combinations]


    for parameters in param_combinations[FLAGS.hp_start:FLAGS.hp_end]:
        print(time.ctime(), parameters)
        
        results = train(
            parameters,
            n_epochs=10,
            verbose=True
        )

    