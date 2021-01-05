"""
Driver code for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
import torch
import torch.nn as nn
from hqsnet import utils, train, dataset, model
import numpy as np
import argparse
import os

if __name__ == "__main__":

    ############### Argument Parsing #################
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lmbda', type=float, default=0, help='gpu id to train on')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
    parser.add_argument('--tv_coeff', type=float, required=True)
    parser.add_argument('--w_coeff', type=float, required=True)

    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--K', type=int, default=5)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    print(args.device)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ##################################################

    ############### Undersampling Mask ###############
    mask = utils.get_mask()
    mask = torch.tensor(mask, requires_grad=False).float().to(args.device)
    ##################################################

    ############### Dataset ##########################
    xdata = utils.get_train_data()
    gt_data = utils.get_train_gt()
    if gt_data.shape[-1] == 1:
        print('Appending complex dimension into gt...')
        gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)

    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    params = {'batch_size': args.batch_size,
         'shuffle': False,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }
    ##################################################

    ############### Hyper-parameters #################
    lmbda = args.lmbda
    K = args.K
    ##################################################

    ############### Model and Optimizer ##############
    network = model.HQSNet(K, mask, lmbda, args.device, n_hidden=args.num_hidden).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    ##################################################

    train.train_hqsnet(network, optimizer, dataloaders, args.num_epochs, args.device, mask, args.w_coeff, args.tv_coeff)
