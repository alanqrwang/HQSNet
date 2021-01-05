"""
Training loop for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
from . import loss as losslayer
from . import utils, model
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt

def train_hqsnet(reconnet, optimizer, dataloaders, num_epochs, device, mask, w_coeff, tv_coeff):
    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train %d/%d' % (epoch, num_epochs))
                reconnet.train()
            elif phase == 'val':
                print('Validate %d/%d' % (epoch, num_epochs))
                reconnet.eval()

            epoch_loss = 0
            epoch_samples = 0

            for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                y = y.float().to(device)
                gt = gt.float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    zf = utils.ifft(y)
                    y, zf = utils.scale(y, zf)

                    alpha = torch.tensor(tv_coeff).to(device)
                    beta = torch.tensor(w_coeff).to(device)

                    x_hat = reconnet(zf, y)
                    loss = losslayer.unsup_loss(x_hat, y, mask, alpha, beta, device)

                    if phase == 'train' and loss is not None:
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.data.cpu().numpy()

                epoch_samples += len(y)

            epoch_loss /= epoch_samples
            if phase == 'train':
                train_epoch_loss = epoch_loss
            if phase == 'val':
                val_epoch_loss = epoch_loss
        # Optionally save checkpoints here
    return reconnet

