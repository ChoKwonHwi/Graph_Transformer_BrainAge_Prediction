import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 4'
import torch
from torch.utils.data import DataLoader
import Data
import utils
import ConvNet
import logging
import numpy as np
import shutil

data_path = './data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data preparation
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
dataLoader = DataLoader(Data.Brain_image(data_path, modality), batch_size=1, shuffle=False, **kwargs)

for batch_idx, (image, label, name) in enumerate(dataLoader):
    image = image.float().to(device)
    print(image.shape, label.shape)
    if image.shape
