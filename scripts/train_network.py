import torch
import torch.nn as nn
from hqsnet import utils, train, dataset, model
import numpy as np
import argparse
import os

if __name__ == "__main__":

    ############### Argument Parsing #################
    parser = argparse.ArgumentParser(description='Half-Quadratic Minimization for CS-MRI in Pytorch')
    parser.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
    parser.add_argument('--models_dir', default='/nfs02/users/aw847/models/HQSplitting/', type=str, help='directory to save models')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lmbda', type=float, default=0, help='gpu id to train on')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
    parser.add_argument('--undersample_rate', choices=['4p1', '4p2', '8p25', '8p3'], type=str, help='undersample rate', required=True)
    parser.add_argument('--dataset', choices=['t1', 't2', 'knee', 'brats'], type=str, help='dataset', required=True)
    parser.add_argument('--strategy', choices=['sup', 'unsup', 'refine'], type=str, help='training strategy', required=True)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(args.device)
    ##################################################

    ############### Undersampling Mask ###############
    maskname = args.undersample_rate
    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(args.device)
    ##################################################

    ############### Dataset ##########################
    if args.dataset == 't1':
        data_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized.npy'
    elif args.dataset == 't2':
        data_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_test_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_test_normalized.npy'
    elif args.dataset == 'knee':
        data_path = '/nfs02/users/aw847/data/knee/knee_test_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/knee/knee_test_normalized.npy'

    xdata = utils.get_data(data_path.format(maskname=maskname))
    gt_data = utils.get_data(gt_path)
    if gt_data.shape[-1] == 1:
        print('Appending complex dimension into gt...')
        gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)

    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    params = {'batch_size': 8,
         'shuffle': False,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }
    ##################################################

    ############### Hyper-parameters #################
    w_coeff, tv_coeff = utils.get_reg_coeff()
    lmbda = args.lmbda
    K = utils.get_K()
    ##################################################

    ############### Model and Optimizer ##############
    network = model.HQSNet(K, mask, lmbda).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    ##################################################

    ################### Filename #####################
    local_name = '{prefix}_{lr}_{w_coeff}_{tv_coeff}_{lmbda}_{K}/{dataset}_{undersample_rate}_{sup}'.format(
        prefix=args.filename_prefix,
        dataset=args.dataset,
        undersample_rate=maskname,
        lr=args.lr,
        w_coeff=w_coeff,
        tv_coeff=tv_coeff,
        lmbda=lmbda,
        K=K,
        sup=args.strategy)
    model_folder = os.path.join(args.models_dir, local_name)
    if not os.path.isdir(model_folder):   
        os.makedirs(model_folder)
    filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')

    train.train_hqsnet(network, optimizer, dataloaders, args.num_epochs, args.device, w_coeff, tv_coeff, mask, filename, args.strategy)
