import torch
import numpy as np
import myutils

def abs(arr):
    # Expects input of size (N, l, w, 2)
    assert arr.shape[-1] == 2
    return torch.norm(arr, dim=3)

def scale(y, y_zf):
    # print('ifft', torch.sum(y_zf))
    flat_yzf = torch.flatten(abs(y_zf), start_dim=1, end_dim=2)
    # print('flat_yzf', torch.sum(flat_yzf))
    max_val_per_batch, _ = torch.max(flat_yzf, dim=1, keepdim=True)
    # print('max_val', torch.sum(max_val_per_batch))
    y = y / max_val_per_batch.view(len(y), 1, 1, 1)
    y_zf = y_zf / max_val_per_batch.view(len(y), 1, 1, 1)
    return y, y_zf

def fft(x):
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    return torch.ifft(x, signal_ndim=2, normalized=True)

def get_reg_coeff():
    return 0.002, 0.005

def get_lmbda():
    return 1.8

def get_data(data_path, N=None):
    print('Loading from', data_path)
    xdata = np.load(data_path)
    assert len(xdata.shape) == 4
    print('data shapes:', xdata.shape)
    return xdata

def get_mask(maskname, centered=False):
    mask = np.load('/nfs02/users/aw847/data/Masks/poisson_disk_%s_256_256.npy' % maskname)
    if not centered:
        return np.fft.fftshift(mask)
    else:
        return mask

def normalize_recons(recons):
    recons = myutils.array.make_imshowable(recons)
    recons = myutils.array.normalize(recons)
    return recons
