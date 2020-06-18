import torch
import torch.nn as nn
from hqsplitamortized import utils, train, dataset
from hqsplitamortized.loss import final_loss as hqloss
import numpy as np
import argparse
import pprint
import os

def train_amortized(xdata, w_coeff, tv_coeff, lmbda, lr, mask, filename):
    print(w_coeff, tv_coeff)
    losses = []
    
    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):])

    params = {'batch_size': 8,
         'shuffle': True,
         'num_workers': 4}

    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }

    num_epochs = 100
    load_checkpoint = 0
    y = torch.tensor(xdata[0:1]).to(device).float()
    learned_resblocks = train.train_hqsplitamortized(dataloaders, num_epochs, device, load_checkpoint, w_coeff, tv_coeff, lmbda, 
                                lr, mask, filename, log_interval=1, test_data=y)
    y, y_zf = utils.scale(y, utils.ifft(y))

    pred,_,_ = train.alternate_minimize(y_zf, y, learned_resblocks, None, w_coeff, tv_coeff, lmbda, device, 'val', mask)
    nn_loss, _,_ = hqloss(pred, y, mask, w_coeff, tv_coeff, device)
#         print('x', pred)
#         print('y', y)
    print('appending', nn_loss)
#         print('in code function ', final_losses)
    losses.append(nn_loss)
#     save_path = os.path.join(filename,'{index:04d}.npy'.format(index=slice_number)) 
#     print('saving to', save_path) 
#     np.save(save_path, x_split.cpu().detach().numpy()) 
    
    return losses


parser = argparse.ArgumentParser(description='Half-Quadratic Minimization for CS-MRI in Pytorch')

parser.add_argument('--K', default=5, type=int, metavar='N',
                    help='number of iterations')
parser.add_argument('--lmbda', type=float, default=0.002)
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
    data_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_{maskname}.npy' 
elif args.dataset == 't2':
    data_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_train_normalized_{maskname}.npy'
elif args.dataset == 'knee':
    data_path = '/nfs02/users/aw847/data/knee/knee_train_normalized_{maskname}.npy'
elif args.dataset == 'brats':
    data_path = '/nfs02/users/aw847/data/brain/brats/brats_t1_train_normalized_{maskname}.npy'

xdata = utils.get_data(data_path.format(maskname=maskname))

local_name = '{prefix}_{dataset}_{undersample_rate}_{lr}_{lmbda}_{K}'.format(
                prefix=args.filename_prefix,
                dataset=args.dataset,
                undersample_rate=maskname,
                lr=args.lr,
                lmbda=args.lmbda,
                K=args.K)
model_folder = os.path.join(args.models_dir, local_name)
if not os.path.isdir(model_folder):   
    os.makedirs(model_folder)
filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')

w_coeff, tv_coeff = utils.get_reg_coeff()

train_loss = train_amortized(xdata, w_coeff, tv_coeff, args.lmbda, args.lr, mask, filename)
plt.plot(train_loss)
