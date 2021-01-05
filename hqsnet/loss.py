"""
Loss for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from . import utils

def data_consistency(k, k0, mask, lmbda=0):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    mask = mask.unsqueeze(-1)
    mask = mask.expand_as(k)

    return (1 - mask) * k + mask * (lmbda*k + k0) / (1 + lmbda)

def get_tv(x):
    """Total variation loss

    Parameters
    ----------
    x : torch.Tensor (batch_size, img_height, img_width, 2)
        Input image

    Returns
    ----------
    tv_loss : TV loss
    """
    tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs())
    tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs())
    return tv_x + tv_y

def get_wavelets(x, device):
    """L1-penalty on wavelets

    Parameters
    ----------
    x : torch.Tensor (batch_size, img_height, img_width, 2)
        Input image

    Returns
    ----------
    tv_loss : wavelets loss
    """
    xfm = DWTForward(J=3, mode='zero', wave='db4').to(device) # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(x)

    batch_size = x.shape[0]
    channels = x.shape[1]
    rows = nextPowerOf2(Yh[0].shape[-2]*2)
    cols = nextPowerOf2(Yh[0].shape[-1]*2)
    wavelets = torch.zeros(batch_size, channels, rows, cols).to(device)
    # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
    for i, band in enumerate(Yh):
        irow = rows // 2**(i+1)
        icol = cols // 2**(i+1)
        wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

    wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
    return wavelets

def nextPowerOf2(n):
    """Get next power of 2"""
    count = 0;

    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;

def loss_with_reg(z, x, w_coeff, tv_coeff, lmbda, device):
    '''
    z is learned variable, output of model
    x is proximity variable
    '''
    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
    dc = lmbda*l2(z, x)

    # Regularization
    z = z.permute(0, 3, 1, 2)
    tv = get_tv(z)
    wavelets = get_wavelets(z, device)
    l1_wavelet = l1(wavelets, torch.zeros_like(wavelets)) # we want L1 value by itself, not the error

    reg = w_coeff*l1_wavelet + tv_coeff*tv

    loss = dc + reg

    return loss, dc, reg

def unsup_loss(x_hat, y, mask, alpha, beta, device):
    '''Unsupervised loss for amortized optimization
    Loss = DC + (1-alpha)*beta * Reg1 + (1-alpha)*(1-beta) * Reg2

    Parameters
    ----------
    x_hat : torch.Tensor (batch_size, img_height, img_width, 2)
        Reconstructed image
    y : torch.Tensor (batch_size, img_height, img_width, 2)
        Under-sampled measurement
    mask : torch.Tensor (img_height, img_width)
        Under-sampling mask
    alpha : float
        Wavelet regularization weighting coefficient
    beta : float
        TV regularization weighting coefficient
    device : str
        Pytorch device string

    Returns
    ----------
    loss : total amortized loss
    dc : dc loss
    '''
    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
 
    mask_expand = mask.unsqueeze(2)
 
    # Data consistency term
    Fx_hat = utils.fft(x_hat)
    UFx_hat = Fx_hat * mask_expand
    dc = l2(UFx_hat, y)

    # Regularization
    x_hat = x_hat.permute(0, 3, 1, 2)
    tv = get_tv(x_hat)
    wavelets = get_wavelets(x_hat, device)
    l1_wavelet = l1(wavelets, torch.zeros_like(wavelets)) # we want L1 value by itself, not the error
 
    # Total loss
    loss = dc + alpha*l1_wavelet + beta*tv

    return loss
