import os
import torch
import torch.nn
from hqsplitamortized import solver, utils, dataset, train, model
import argparse
import pprint
import numpy as np

parser = argparse.ArgumentParser(description='Amortized Half-Quadratic Minimization for CS-MRI in Pytorch')

parser.add_argument('--K', default=5, type=int, metavar='N',
                    help='number of iterations')
parser.add_argument('--lmbda', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--gpu_id', default=0, type=int,
                    metavar='N', help='gpu id to train on')
parser.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
parser.add_argument('--models_dir', default='/nfs02/users/aw847/models/HQSplitting/', type=str, help='directory to save models')

parser.add_argument('--undersample_rate', choices=['4p1', '4p2', '8p25', '8p3'], type=str, help='undersample rate', required=True)
parser.add_argument('--dataset', choices=['t1', 't2', 'knee', 'brats'], type=str, help='undersample rate', required=True)

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
else:
    args.device = torch.device('cpu')
device = args.device
pprint.pprint(vars(args))

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

maskname = args.undersample_rate
mask = utils.get_mask(maskname)
mask = torch.tensor(mask, requires_grad=False).float().to(device)

if args.dataset == 't1':
    data_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_{maskname}.npy' 
elif args.dataset == 't2':
    data_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_test_normalized_{maskname}.npy'
elif args.dataset == 'knee':
    data_path = '/nfs02/users/aw847/data/knee/knee_test_normalized_{maskname}.npy'
elif args.dataset == 'brats':
    data_path = '/nfs02/users/aw847/data/brain/brats/brats_t1_tesj_normalized_{maskname}.npy'

xdata = utils.get_data(data_path.format(maskname=maskname))



w_coeff, tv_coeff = utils.get_reg_coeff()
num_epochs = 500
load_checkpoint = 0

for slice_number in range(len(xdata)):

    local_name = '{prefix}_{dataset}_{undersample_rate}_{lr}_{lmbda}_{K}/{index:04d}/'.format(
                    prefix=args.filename_prefix,
                    dataset=args.dataset,
                    undersample_rate=maskname,
                    lr=args.lr,
                    lmbda=args.lmbda,
                    K=args.K,
                    index=slice_number)
    filename = os.path.join(args.models_dir, local_name)
    if not os.path.isdir(filename):   
        os.makedirs(filename)

    y = torch.tensor(xdata[slice_number:slice_number+1]).float().to(device)

    trainset = dataset.Dataset(y)
    valset = dataset.Dataset(y)

    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 0}

    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }
    learned_resblocks = train.train_hqsplitamortized(dataloaders, num_epochs, device, load_checkpoint, w_coeff, tv_coeff, args.lmbda, args.lr, mask, log_interval=50)
    pred = model.forward(y, learned_resblocks, mask, args.lmbda)

    pred_save_path = os.path.join(filename,'pred.npy')
    model_save_path = os.path.join(filename,'model.h5')
    print('saving pred to', pred_save_path)
    print('saving model to', model_save_path)
    torch.save(learned_resblocks.state_dict(), model_save_path)
    np.save(pred_save_path, pred.cpu().detach().numpy())
