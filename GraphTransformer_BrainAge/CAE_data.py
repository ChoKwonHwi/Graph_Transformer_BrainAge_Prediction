import torch
from torch.utils.data import Dataset as image_dataset
from torch_geometric.data import Dataset as graph_dataset
from torch_geometric.data import Data
import os
import nibabel as nib
import numpy as np
import utils
import natsort

def zscore_normalize(volume):
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std

class Brain_image(image_dataset):
    def __init__(self, data_path, modality):
        self.data_path = data_path
        self.folder_list = [folder for folder in natsort.natsorted(os.listdir(self.data_path))]
        self.modality = modality

    def __getitem__(self, idx):
        # get item by index
        if self.modality == 'MRI':
            MRI_path = os.path.join(self.data_path, self.folder_list[idx], self.folder_list[idx]+'_linreg_Warped.nii.gz')
            MRI = nib.load(MRI_path).get_fdata()
            MRI = torch.from_numpy(MRI)

            #MRI = zscore_normalize(MRI)

            MRI = torch.unsqueeze(MRI, dim=0)
            #print(self.folder_list[idx])
            return MRI
            
        else :
            DTI_path = os.path.join(self.data_path, self.folder_list[idx], self.folder_list[idx]+'_eddy_dti_FA_reorient_Warped.nii.gz')
            DTI = nib.load(DTI_path).get_fdata()
            DTI = torch.from_numpy(DTI)

            #DTI = zscore_normalize(DTI)

            DTI = torch.unsqueeze(DTI, dim=0)

            return DTI

    def __len__(self):
        return len(self.folder_list)