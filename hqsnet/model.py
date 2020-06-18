from . import utils
from . import loss as losslayer
import torch
import torch.nn as nn

'''
4-layer CNN with residual output
'''
class ResBlock(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, linear=False):
        super(ResBlock, self).__init__()
        self.linear = linear
        self.conv1 = nn.Conv2d(n_ch, nf, ks, padding = ks//2)
        self.conv2 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4 = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.conv1(x)
        if not self.linear:
            conv1_out = self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        if not self.linear:
            conv2_out = self.relu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        if not self.linear:
             conv3_out = self.relu(conv3_out)

        conv4_out = self.conv4(conv3_out)

        x_res = x + conv4_out
        return x_res
