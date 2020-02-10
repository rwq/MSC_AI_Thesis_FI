import torch
import numpy as np

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
    
    batch_psnr = 20 * np.log10(R) - 10 * torch.log10(batch_mse)
    
    return batch_psnr


def interpolation_error(y_hat, y):
    '''
    Computes pixel-wise RMSE over batches.
    Both inputs should be of shape:
    batch_size x n_channels x height x width
    '''
    
    assert y_hat.size() == y.size(), \
        f'Input dimensions must match.  DIM1: {y_hat.size()}, DIM2: {y.size()}'
    

    mse = (y_hat-y).pow(2).mean(dim=(1,2,3))
    
    return mse.sqrt()


def ssim(im1, im2, c1=1., c2=1., c3=1.):
    '''
    Computes the SSIM for color images.
    Both inputs should be of shape:
    batch_size x n_channels x height x width
    '''
    
    assert im1.size() == im2.size(),  \
        f'Input dimensions must match.  DIM1: {im1.size()}, DIM2: {im2.size()}'
    
    
    im1 = im1.float()
    im2 = im2.float()
    
    # compute image statistics
    mu1 = im1.mean(dim=(1,2,3), keepdims=True)
    mu2 = im2.mean(dim=(1,2,3), keepdims=True)
    sig1 = im1.std(dim=(1,2,3), keepdims=True)
    sig2 = im2.std(dim=(1,2,3), keepdims=True)
    
    # calculate covariance
    e1 = (im1-mu1).reshape(im1.size(0), -1)
    e2 = (im2-mu2).reshape(im2.size(0), -1)
    cov = (e1 * e2).sum(dim=1) / (e1.size(1)-1)
    
    # remove dimensions
    mu1 = mu1.squeeze()
    mu2 = mu2.squeeze()
    sig1 = sig1.squeeze()
    sig2 = sig2.squeeze()
    
    # compute ssim
    L = (2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1)
    C = (2 * sig1 * sig2 + c2) / (sig1**2 + sig2**2 + c2)
    S = (cov + c3) / (sig1 * sig2 + c3)
    
    return L * C * S