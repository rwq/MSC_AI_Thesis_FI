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


os.makedirs(MODEL_FOLDER, exist_ok=True)


torch.manual_seed(42)
np.random.seed(42)



def visual_evaluation(model, quadratic, writer):
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

    writer.add_image('hard_examples', img_tensor=grid)




def objective(params):
    '''Minimize loss on validation set'''
    
    _, _, resultstore = train(params, n_epochs=5, verbose=False)
    
    losses = resultstore.results['valid']['L1_loss']
    max_epoch = np.max(list(losses.keys()))
    validation_loss = np.mean(losses[max_epoch])
    
    return validation_loss


def train(params, n_epochs, verbose=True):
    
    # init interpolation model
    G = get_sepconv(input_size = params['input_size'], weights= params["weights"])
    G = G.cuda().eval()

    loss_network = losses.LossNetwork(layers=[9,16]).cuda()
    critereon = losses.PerceptualLoss(loss_network)
    l1_loss = torch.nn.L1Loss()
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

    optimizer = torch.optim.Adam(G.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # metrics
    name = f'{int(time.time())}_{params["lr"]}_{params["weight_decay"]}_{params["input_size"]}_{params["weights"]}'
    writer = SummaryWriter(f'runs/perceptual_loss_3/{name}') #TODO hp
    
    results = ResultStore(
        writer=writer,
        metrics = ['psnr', 'ie', 'L1_loss', 'perceptual_loss'],
        folds=['train', 'valid', 'test']
    )

    early_stopping = EarlyStopping(results, patience=FLAGS.patience, metric='perceptual_loss')
    



    def do_epoch(dataloader, fold, epoch):
        assert fold in ['train', 'valid', 'test']

        if verbose:
            pb = tqdm(desc=f'{fold} {epoch+1}/{n_epochs}', total=len(dataloader), leave=True, position=0)
        
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()    

            y_hat = G(X)
            loss = critereon(y_hat, y)


            if fold == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            
            

            # compute metrics
            y_hat = (y_hat * 255).clamp(0,255)
            y = (y * 255).clamp(0,255)

            psnr = metrics.psnr(y_hat, y)
            ie = metrics.interpolation_error(y_hat, y)
            l1 = l1_loss(y_hat, y)
            
         
            results.store(fold, epoch, {
                    'L1_loss':l1.item(),
                    'psnr':psnr,
                    'ie':ie,
                    'perceptual_loss':loss.item()
            })
            
            if verbose: pb.update()
       

        # update tensorboard
        results.write_tensorboard(fold, epoch)





    for epoch in range(n_epochs):

        G.train()
        do_epoch(train_dl, 'train', epoch)               

        G.eval()
        with torch.no_grad():
            do_epoch(valid_dl, 'valid', epoch)

        if early_stopping.stop() or epoch % FLAGS.test_every == 0:
            with torch.no_grad():
                do_epoch(test_dl, 'test', epoch)

            visual_evaluation(
                model=G,
                quadratic = params['input_size'] ==4, 
                writer=writer
            )
        
        
        # save model if new best
        if early_stopping.new_best():
            filepath_out = os.path.join(MODEL_FOLDER, '{0}_{1}')
            torch.save(G, filepath_out.format('generator', name))
            
        
        if early_stopping.stop():
            break
    
            
            
    # free memory
    del G
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
         'weights': ['lf']
    }

    param_combinations = product(*parameter_space.values())
    param_combinations = [dict(zip(parameter_space, values)) for values in param_combinations]


    for parameters in param_combinations[FLAGS.hp_start:FLAGS.hp_end]:
        print(time.ctime(), parameters)
        
        results = train(
            parameters,
            n_epochs=FLAGS.n_epochs,
            verbose=True
        )

    