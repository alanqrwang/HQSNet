"""
Utility functions for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
import torch
import numpy as np

def add_bool_arg(parser, name, default=True):
    """Add boolean argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def absval(arr):
    """
    Takes absolute value of last dimension, if complex.
    Input dims:  (N, l, w, 2)
    Output dims: (N, l, w)
    """
    # Expects input of size (N, l, w, 2)
    assert arr.shape[-1] == 2
    return torch.norm(arr, dim=3)

def scale(y, y_zf):
    """Scales inputs for numerical stability"""
    flat_yzf = torch.flatten(absval(y_zf), start_dim=1, end_dim=2)
    max_val_per_batch, _ = torch.max(flat_yzf, dim=1, keepdim=True)
    y = y / max_val_per_batch.view(len(y), 1, 1, 1)
    y_zf = y_zf / max_val_per_batch.view(len(y), 1, 1, 1)
    return y, y_zf

def fft(x):
    """Normalized 2D Fast Fourier Transform"""
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform"""
    return torch.ifft(x, signal_ndim=2, normalized=True)

def get_mask(centered=False):
    mask = np.load('data/mask.npy')
    if not centered:
        return np.fft.fftshift(mask)
    else:
        return mask

def _normalize(arr):
    """Normalizes a batch of images into range [0, 1]"""
    if torch.is_tensor(arr):
        if len(arr.shape) > 2:
            res = torch.zeros_like(arr)
            for i in range(len(arr)):
                res[i] = (arr[i] - torch.min(arr[i])) / (torch.max(arr[i]) - torch.min(arr[i]))
            return res
        else:
            return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))

    else:
        if len(arr.shape) > 2:
            res = np.zeros_like(arr)
            for i in range(len(arr)):
                res[i] = (arr[i] - np.min(arr[i])) / np.ptp(arr[i])
            return res
        else:
            return (arr - np.min(arr)) / np.ptp(arr)

def normalize_recons(recons):
    recons = absval(recons)
    recons = _normalize(recons)
    return recons

def get_data(data_path, N=None):
    print('Loading from', data_path)
    xdata = np.load(data_path)
    assert len(xdata.shape) == 4
    print('Shape:', xdata.shape)
    return xdata

def get_train_gt():
    gt_path = 'data/example_x.npy'
    gt = get_data(gt_path)
    return gt

def get_train_data():
    data_path = 'data/example_y.npy'
    data = get_data(data_path)
    return data
