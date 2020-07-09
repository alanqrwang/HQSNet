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
    x = x.float()
    tv_x = torch.sum(((x[:, 0, :, :-1] - x[:, 0, :, 1:])**2 + (x[:, 1, :, :-1] - x[:, 1, :, 1:])**2)**0.5)
    tv_y = torch.sum(((x[:, 0, :-1, :] - x[:, 0, 1:, :])**2 + (x[:, 1, :-1, :] - x[:, 1, 1:, :])**2)**0.5)

    return tv_x + tv_y

def get_wavelets(x, device):
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
    count = 0;

    # First n in the below  
    # condition is for the  
    # case where n is 0 
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

def final_loss(x_hat, y, mask, device, reg_only=False):
    w_coeff, tv_coeff = utils.get_reg_coeff()

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
 
    reg = w_coeff*l1_wavelet + tv_coeff*tv

    if reg_only:
        loss = reg
    else:
        loss = dc + reg

    return loss

def get_loss(x_hat, gt, y, mask, device, strategy, batch_idx, epoch, phase, max_batch):
    if strategy == 'unsup':
        loss = final_loss(x_hat, y, mask, device)
    elif strategy == 'sup':
        loss = nn.MSELoss()(x_hat, gt)

    # Trains supervised for 20 epochs, and only on first 1/10 of the dataset
    # Trains unsupervised for rest of the epochs, and on 9/10 of the dataset
    elif strategy == 'refine':
        if phase == 'train' and epoch <= 100:
            if batch_idx < max_batch // 2:
                loss = nn.MSELoss()(x_hat, gt)
                print('sup on batch_idx', str(batch_idx))
            else:
                loss = None
                print('sup skipping batch_idx', str(batch_idx))
        elif phase == 'train' and epoch > 100:
            if batch_idx >= max_batch // 2:
                loss = final_loss(x_hat, y, mask, device)
                print('unsup on batch_idx', str(batch_idx))
            else:
                loss = None
                print('unsup skipping batch_idx', str(batch_idx))
        else:
            loss = nn.MSELoss()(x_hat, gt)
            print('validation in refine')

    return loss
