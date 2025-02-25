import os
import torch
from CAE_data import Brain_image
from torch.utils.data import DataLoader
import Data
import GraphNet
import logging
import ConvNet
import numpy as np
import torch.optim as optim
import argparse
import csv
from torch.optim import AdamW
import warnings
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F
warnings.filterwarnings(action='ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='MRI')
parser.add_argument('--model_path', type=str, default='./trained/CAE')
parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/data/cohort')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--loss_function', type=str, default='BCE_loss')
parser.add_argument('--output_csv_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/loss.csv')
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = Brain_image(data_path=args.data_path, modality = args.modality)

    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNet.ConvAutoEncoder(nChannels=16).to(device)
    #model = ConvNet.Feature_Extraction(nChannels=16)

    # move the network to GPU/CPU
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    torch.cuda.empty_cache()

    # get trained model
    save_model = torch.load("/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/trained/CAE/model_MRI_CAE_more22.pth")
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    learning_rate = args.lr
    criterion = torch.nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    num_train, num_val = len(train_loader) / args.batch_size, len(val_loader) / args.batch_size
    all_loss_train, all_loss_valid = [], []
    min_valid_loss = 10000
    target_shape = (args.batch_size, 1, 160, 192, 160)
    output_dir = "/nasdata4/kwonhwi/GraphTransformer/data/CAE"
    cnt = 0
    model.eval()
    with torch.no_grad():
        for idx, data1 in enumerate(tqdm(val_loader)) :
            data1 = data1.to(device).float()

            if data1.shape != torch.Size(target_shape) :
                data = F.interpolate(data1, size=target_shape[2:], mode='trilinear', align_corners=False)
            
            output_path = os.path.join(output_dir, f'prediction_mri_{idx}_no_norm_org.nii.gz')
            datas = nib.Nifti1Image(data.cpu().squeeze().numpy(), affine=np.eye(4))
            nib.save(datas, output_path)

            print('data:', data.shape)
            pred = model(data)
            print('pred:', pred.shape)
            nifti_pred = nib.Nifti1Image(pred.cpu().squeeze().numpy(), affine=np.eye(4))  # Set identity affine or a suitable one
            
            # Save as NIfTI file
            output_path = os.path.join(output_dir, f'prediction_mri_{idx}_no_norm_feat.nii.gz')
            nib.save(nifti_pred, output_path)

            cnt +=1 
            if cnt == 1:
                break


if __name__ == '__main__':
    main()