import os
import torch
from CAE_data import Brain_image
from torch.utils.data import DataLoader
import Data
import GraphNet
from ConvNet import ConvAutoEncoder
import logging
import numpy as np
import torch.optim as optim
import argparse
import csv
from torch.optim import AdamW
import warnings
from tqdm import tqdm
import torch.nn.functional as F
warnings.filterwarnings(action='ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='MRI')
parser.add_argument('--model_path', type=str, default='./trained/CAE')
parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/data/cohort')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--loss_function', type=str, default='BCE_loss')
parser.add_argument('--output_csv_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/loss_no_norm_MRI_23.csv')
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = Brain_image(data_path=args.data_path, modality = args.modality)

    total_size = len(full_dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvAutoEncoder(nChannels=16).to(device)

    # move the network to GPU/CPU
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # get trained model
    save_model = torch.load("/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/trained/CAE/model_MRI_CAE_more.pth")
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


    weights_best_name_loss    = "model_MRI_CAE_more22.pth"   
    weights_best_loss       = os.path.join(args.model_path, weights_best_name_loss)

    learning_rate = args.lr
    criterion = torch.nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    num_train, num_val = len(train_loader) / args.batch_size, len(val_loader) / args.batch_size
    all_loss_train, all_loss_valid = [], []
    min_valid_loss = 100000000000
    target_shape = (args.batch_size, 1, 160, 192, 160)

    for epoch in range(args.epochs) :
        if epoch == 50 :
            learning_rate = 1e-4
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
        model.train()
        total_train_loss, total_val_loss = 0, 0
        for idx, data in enumerate(tqdm(train_loader)) :
            data = data.to(device).float()
            if data.shape != torch.Size(target_shape) :
                data = F.interpolate(data, size=target_shape[2:], mode='trilinear', align_corners=False)

            optimizer.zero_grad()

            pred = model(data)
            #print(data.shape, pred.shape)
            #print(f'Output range: {pred.min().item()} - {pred.max().item()}')
            #print(f'Target range: {data.min().item()} - {data.max().item()}')
            loss = criterion(pred, data)
            loss.backward()
        
            optimizer.step()
        
            total_train_loss += loss.item()
        
        all_loss_train.append(total_train_loss/num_train)
        print("epoch {} Train loss : {:.6f}".format(epoch, total_train_loss/num_train))    

        model.eval()
        with torch.no_grad():
            for idx, data in enumerate(tqdm(val_loader)) :
                data = data.to(device).float()

                if data.shape != torch.Size(target_shape) :
                    data = F.interpolate(data, size=target_shape[2:], mode='trilinear', align_corners=False)

                pred = model(data)
                loss = criterion(pred, data)
                total_val_loss += loss.item()

            all_loss_valid.append(total_val_loss/num_val)

        print("epoch {} Valid loss : {:.6f}".format(epoch, total_val_loss/num_val))
        
        if min_valid_loss > total_val_loss/num_val :
            torch.save(model.state_dict(), weights_best_loss)
            print("model save!!")

        with open(args.output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in zip(all_loss_train, all_loss_valid):
                writer.writerow(row)


if __name__ == '__main__':
    main()