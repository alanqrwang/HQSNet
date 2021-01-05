"""
Dataset wrapper for HQSNet
For more details, please read:
    
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Neural Network-based Reconstruction in Compressed Sensing MRI Without Fully-sampled Training Data" 
    MLMIR 2020. https://arxiv.org/abs/2007.14979
"""
import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        x = self.data[index]
        y = self.labels[index]
        return x, y
