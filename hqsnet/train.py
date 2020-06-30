from . import loss as losslayer
from . import utils
from . import model
import myutils
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_hqsnet(model, optimizer, dataloaders, num_epochs, device, w_coeff, tv_coeff, mask, unsupervised=True):
    loss_list = []
    val_loss_list = []

    best_val_loss = 1e10
    best_epoch = 0
    
    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train %d/%d, unsupervised=%s' % (epoch, num_epochs, unsupervised))
                model.train()
            elif phase == 'val':
                print('Validate %d/%d, unsupervised=%s' % (epoch, num_epochs, unsupervised))
                model.eval()

            epoch_loss = 0
            epoch_samples = 0

            for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                y = y.float().to(device)
                gt = gt.float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    zf = utils.ifft(y)
                    if unsupervised:
                        y, zf = utils.scale(y, zf)
 
                    x_hat = model(zf, y)

                    if unsupervised:
                        loss = losslayer.final_loss(x_hat, y, mask, w_coeff, tv_coeff, device)
                    else:
                        loss = nn.MSELoss()(x_hat, gt)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.data.cpu().numpy() * len(y)

                epoch_samples += len(y)

            epoch_loss /= epoch_samples
            if phase == 'train':
                train_epoch_loss = epoch_loss
            if phase == 'val':
                val_epoch_loss = epoch_loss
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_epoch = epoch
            
        print('Best loss: %s, Epoch: %s' % (best_val_loss, best_epoch))
        # Optionally save checkpoints here, e.g.:
        # save_checkpoint(epoch, resblocks.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, filename, log_interval)

    return resblocks, loss_list
