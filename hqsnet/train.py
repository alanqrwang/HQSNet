from . import loss as losslayer
from . import utils
from . import model
import myutils
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def end_to_end_minimize(inp, y, resblocks, optimizer, w_coeff, tv_coeff, lmbda, device, phase, mask, gt, unsupervised):
    print('parameters', w_coeff, tv_coeff, lmbda)
    x = inp.clone()
    
    for i in range(len(resblocks)):
        # z-minimization
        res = resblocks[i]
        if phase == 'train':
            optimizer.zero_grad()
        
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        
        z = res(x)
        
        y = y.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        
        # x-minimization
        z_ksp = utils.fft(z)
        x_ksp = losslayer.data_consistency(z_ksp, y, mask, lmbda=lmbda)
        x = utils.ifft(x_ksp)
        
    if unsupervised:
        loss, _, _ = losslayer.final_loss(x, y, mask, w_coeff, tv_coeff, device)
    else:
        crit = nn.MSELoss()
        loss = crit(x, gt)
    if phase == 'train':
        loss.backward()
        optimizer.step()
    return x.detach(), resblocks, optimizer, loss.detach()

def alternate_minimize(inp, y, resblocks, optimizers, w_coeff, tv_coeff, lmbda, device, phase, mask):
    print('parameters', w_coeff, tv_coeff, lmbda)
    x = inp.clone()
    
    for i in range(len(resblocks)):
        # z-minimization
        res = resblocks[i]
        if phase == 'train':
            optimizer = optimizers[i]
            optimizer.zero_grad()
        
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        
        z = res(x)
        loss, _, _ = losslayer.loss_with_reg(z, x, w_coeff, tv_coeff, lmbda, device)
        if phase == 'train':
            loss.backward()
            optimizer.step()
        z = z.detach()
        
        y = y.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        
        # x-minimization
        z_ksp = utils.fft(z)
        x_ksp = losslayer.data_consistency(z_ksp, y, mask, lmbda=lmbda)
        x = utils.ifft(x_ksp)
        
    return x, resblocks, optimizers

def alternate_minimize_separate(inp, y, resblocks, optimizers, w_coeff, tv_coeff, lmbda, device, phase, mask, batch_idx, num_batches, epoch):
    print('parameters', w_coeff, tv_coeff, lmbda)
    x = inp.clone()

    # block_to_train = epoch % len(resblocks) # which block to start training
    block_to_train = int(batch_idx / (int(num_batches) / len(resblocks))) # index of block to train
    print('training', block_to_train)
    
    for i in range(len(resblocks)):
        # z-minimization
        res = resblocks[i]
        if phase == 'train' and i == block_to_train:
            optimizer = optimizers[i]
            optimizer.zero_grad()
        
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        
        z = res(x)
        if phase == 'train' and i == block_to_train:
            loss, _, _ = losslayer.loss_with_reg(z, x, w_coeff, tv_coeff, lmbda, device)
            loss.backward()
            optimizer.step()
        z = z.detach()
        
        y = y.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        
        # x-minimization
        z_ksp = utils.fft(z)
        x_ksp = losslayer.data_consistency(z_ksp, y, mask, lmbda=lmbda)
        x = utils.ifft(x_ksp)
        
    return x, resblocks, optimizers

def train_hqsplitamortized(dataloaders, num_epochs, device, load_checkpoint, w_coeff, tv_coeff, lmbda, K, lr, mask, filename, log_interval=50, test_data=None, unsupervised=True):
    loss_list = []
    val_loss_list = []

    best_val_loss = 1e10
    best_epoch = 0
    
    # Model
    resblocks = nn.ModuleList()
    optimizers = []
    for i in range(K):
        resblock = model.ResBlock().to(device)
        resblocks.append(resblock)
        optimizers.append(torch.optim.Adam(resblock.parameters(), lr=lr))
    big_optimizer = torch.optim.Adam(resblocks.parameters(), lr=lr)

    singles = []
    per_img_losses = []
    for epoch in range(load_checkpoint+1, num_epochs+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train %d/%d, unsupervised=%s' % (epoch, num_epochs, unsupervised))
                for r in resblocks:
                    r.train()
            elif phase == 'val':
                print('Validate %d/%d, unsupervised=%s' % (epoch, num_epochs, unsupervised))
                for r in resblocks:
                    r.eval()

            epoch_loss = 0
            epoch_samples = 0

            for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                y = y.float().to(device)
                gt = gt.float().to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    zf = utils.ifft(y)
                    if unsupervised:
                        y, zf = utils.scale(y, zf)
 
                    x_hat, resblocks, optimizers, loss = end_to_end_minimize(zf, y, resblocks, big_optimizer, w_coeff, tv_coeff, lmbda, device, phase, mask, gt, unsupervised)

                    # loss,_,_ = losslayer.final_loss(x_hat, y, mask, w_coeff, tv_coeff, device)
                    # per_img_losses.append(loss / 8)
                    # if test_data is not None:
                    #     test = test_data.clone()
                    #     test, test_zf = utils.scale(test, utils.ifft(test))

                    #     # pred,_,_ = alternate_minimize(test_zf, test, resblocks, None, w_coeff, tv_coeff, lmbda, device, 'val', mask)
                    #     pred,_,_ = end_to_end_minimize(test_zf, test, resblocks, None, w_coeff, tv_coeff, lmbda, device, 'val', mask, gt, unsupervised)
                    #     nn_loss, _,_ = losslayer.final_loss(pred, test, mask, w_coeff, tv_coeff, device)
                    #     singles.append(nn_loss)
                    #     plt.plot(singles, label='test')
                    #     plt.plot(per_img_losses, label='train')
                    #     # plt.ylim([17, 30])
                    #     plt.legend()
                    #     plt.show()
                    epoch_loss += loss.data.cpu().numpy() * len(y)

                epoch_samples += len(y)

            epoch_loss /= epoch_samples
            if phase == 'train':
                train_epoch_loss = epoch_loss
                # loss_list.append(epoch_loss)
                # print('epoch loss', epoch,  epoch_loss)
                # plt.plot(loss_list, label='nn')
    #             plt.plot(gd_loss, label='gd')
    #             plt.plot(split_loss, label='split')
                # plt.legend()
                # plt.grid()
                # # plt.ylim([17, 30])
                # plt.show()
            if phase == 'val':
                val_epoch_loss = epoch_loss
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_epoch = epoch
            
        print('Best loss: %s, Epoch: %s' % (best_val_loss, best_epoch))
        # myutils.io.save_checkpoint_multiple_optimizers(epoch, resblocks.state_dict(), optimizers, train_epoch_loss, val_epoch_loss, filename, log_interval)
        myutils.io.save_checkpoint(epoch, resblocks.state_dict(), big_optimizer.state_dict(), train_epoch_loss, val_epoch_loss, filename, log_interval)

    return resblocks, loss_list
