import torch
import numpy as np
from pytorch_msssim import ssim as mssim

def psnr(y_hat, y, R=255.0):
    '''
    Both inputs should be of shape:
    batch_size x n_channels x height x width
    
    formula from:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio    
    '''
    
    assert y_hat.size() == y.size(), 'Input dimensions must match.'

    
    # R = max(y.max(), y_hat.max()).float()
    batch_mse = (y_hat-y).pow(2).float().mean(dim=(1,2,3))
    
    # clamp to prevent log 0
    batch_psnr = 20 * np.log10(R) - 10 * torch.log10(batch_mse.clamp(1e-10))
    
    return batch_psnr.tolist()


def interpolation_error(y_hat, y):
    '''
    Computes pixel-wise RMSE over batches.
    Both inputs should be of shape:
    batch_size x n_channels x height x width
    '''
    
    assert y_hat.size() == y.size(), \
        f'Input dimensions must match.  DIM1: {y_hat.size()}, DIM2: {y.size()}'
    

    mse = (y_hat-y).pow(2).mean(dim=(1,2,3))
    
    return mse.sqrt().tolist()



def std(tensor, dim, keepdim):
    device = tensor.device
    numpy_result = tensor.detach().cpu().numpy().std(axis=dim, keepdims=keepdim, ddof=1)
    return torch.from_numpy(numpy_result).to(device)



def ssim(im1, im2):
    '''
    Computes the SSIM for color images.
    Both inputs should be of shape:
    batch_size x n_channels x height x width

    https://github.com/VainF/pytorch-msssim
    '''
    
    return mssim(im1, im2, win_size=11, data_range=255, size_average=False).tolist()
    
    
    
