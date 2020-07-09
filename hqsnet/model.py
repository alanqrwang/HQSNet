from . import utils
from . import loss as losslayer
import torch
import torch.nn as nn

'''
5-layer CNN with residual output
'''
class ResBlock(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, nf, ks, padding = ks//2)
        self.conv2 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv5 = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
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

        x_res = x + conv5_out
        return x_res

class HQSNet(nn.Module):
    def __init__(self, K, mask, lmbda):
        super(HQSNet, self).__init__()

        self.mask = mask
        self.lmbda = lmbda
        self.resblocks = nn.ModuleList()
        for i in range(K):
            resblock = ResBlock()
            self.resblocks.append(resblock)

    def forward(self, x, y):
        for i in range(len(self.resblocks)):
            # z-minimization
            x = x.permute(0, 3, 1, 2)
            
            z = self.resblocks[i](x)
            
            z = z.permute(0, 2, 3, 1)
            
            # x-minimization
            z_ksp = utils.fft(z)
            x_ksp = losslayer.data_consistency(z_ksp, y, self.mask, self.lmbda)
            x = utils.ifft(x_ksp)
        return x
