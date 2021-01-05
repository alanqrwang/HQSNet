"""
Model architecture for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
from . import utils
from . import loss as losslayer
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Upsample(nn.Module):
    """Upsamples input multi-channel image"""
    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class ResBlock(nn.Module):
    '''5-layer CNN with residual output'''
    def __init__(self, n_ch_in=2, n_ch_out=2, nf=64, ks=3):
        '''
        Parameters
        ----------
        n_ch_in : int
            Number of input channels
        n_ch_out : int
            Number of output channels
        nf : int
            Number of hidden channels
        ks : int
            Kernel size
        '''
        super(ResBlock, self).__init__()
        self.n_ch_out = n_ch_out

        self.conv1 = nn.Conv2d(n_ch_in, nf, ks, padding = ks//2)
        self.conv2 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv5 = nn.Conv2d(nf, n_ch_out, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out = self.relu(conv3_out)

        conv4_out = self.conv4(conv3_out)
        conv4_out = self.relu(conv4_out)

        conv5_out = self.conv5(conv4_out)

        x_res = x[:,:self.n_ch_out,:,:] + conv5_out
        return x_res

class HQSNet(nn.Module):
    """HQSNet model architecture"""
    def __init__(self, K, mask, lmbda, device, n_hidden=64):
        """
        Parameters
        ----------
        K : int
            Number of unrolled iterations
        mask : torch.Tensor (img_height, img_width)
            Under-sampling mask
        lmbda : float
            Lambda value
        device : str
            Pytorch device string
        n_hidden : int
            Number of hidden dimensions
        """
        super(HQSNet, self).__init__()

        self.mask = mask
        self.lmbda = lmbda
        self.resblocks = nn.ModuleList()
        self.device = device
            
        for i in range(K):
            resblock = ResBlock(n_ch_in=2, nf=n_hidden)
            self.resblocks.append(resblock)

        self.block_final = ResBlock(n_ch_in=2, nf=n_hidden)


    def forward(self, x, y):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Zero-filled reconstruction
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurements
        """
        for i in range(len(self.resblocks)):
            # z-minimization
            x = x.permute(0, 3, 1, 2)
            
            z = self.resblocks[i](x)
            
            z = z.permute(0, 2, 3, 1)
            
            # x-minimization
            z_ksp = utils.fft(z)
            x_ksp = losslayer.data_consistency(z_ksp, y, self.mask, self.lmbda)
            x = utils.ifft(x_ksp)

        x = x.permute(0, 3, 1, 2)
        x = self.block_final(x)
        x = x.permute(0, 2, 3, 1)
        return x
