from . import loss as losslayer
from . import utils
from . import model
import myutils
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np

def train_hqsnet(model, optimizer, dataloaders, num_epochs, device, w_coeff, tv_coeff, mask, filename, strategy, log_interval=1):
    loss_list = []
    val_loss_list = []

    best_val_loss = 1e10
    best_epoch = 0
    
    for epoch in range(1, num_epochs+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train %d/%d, strategy=%s' % (epoch, num_epochs, strategy))
                model.train()
            elif phase == 'val':
                print('Validate %d/%d, strategy=%s' % (epoch, num_epochs, strategy))
                model.eval()

            epoch_loss = 0
            epoch_samples = 0

            for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                y = y.float().to(device)
                gt = gt.float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    zf = utils.ifft(y)
                    if strategy == 'unsup':
                        y, zf = utils.scale(y, zf)
 
                    x_hat = model(zf, y)
                    loss = losslayer.get_loss(x_hat, gt, y, mask, device, strategy, batch_idx, epoch, phase, len(dataloaders[phase]))

                    if phase == 'train' and loss is not None:
                        loss.backward()
                        optimizer.step()

                    if loss is not None and math.isnan(loss):
                        sys.exit('found nan at epoch ' + str(epoch))
                    if loss is not None:
                        epoch_loss += loss.data.cpu().numpy()

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
        myutils.io.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, filename, log_interval)

    return model, loss_list

def test_hqsnet(trained_model, xdata, strategy, device):
    recons = []
    for i in range(len(xdata)):
        y = torch.as_tensor(xdata[i:i+1]).to(device).float()
        zf = utils.ifft(y)
        if strategy == 'unsup':
            y, zf = utils.scale(y, zf)

        pred = trained_model(zf, y)
        recons.append(pred.cpu().detach().numpy())

    preds = np.array(recons).squeeze()
     
    return preds
