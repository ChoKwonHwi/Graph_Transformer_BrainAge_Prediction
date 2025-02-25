import os
import torch
from torch_geometric.data import DataLoader
import Data2
import GraphNet
import logging
import numpy as np
import torch.optim as optim
import argparse
import csv
import warnings


print("PyTorch version:", torch.__version__)           # PyTorch version: 2.6.0.dev20241028
print("CUDA available:", torch.cuda.is_available())    # CUDA available: True
print("SparseTensor and scatter imported successfully!")

print(torch.cuda.device_count())                       # 1
print(torch.cuda.get_device_name(0))                   # NVIDIA RTX A5000

warnings.filterwarnings(action='ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../trained/GCN_cohort')
#parser.add_argument('--data_path', type=str, default='/mnt/GraphTransformer/data/cohort')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--loss_function', type=str, default='BCE_loss')
parser.add_argument('--output_csv_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/ugugugugugugug/MRI_GCN.csv')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main():
    # experiment settings
    model_path = '../trained/GCN_cohort'
    data_path = '../brain_network/project'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    logging.info('Prepare data...')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    #full_dataset = Data.Brain_network(data_path)
    dataLoader = DataLoader(Data2.Brain_network(data_path), batch_size=1, shuffle=False, **kwargs)

    
    # network construction
    logging.info('Initialize network...')
    net = GraphNet.GCNFeatureUpdater(nfeat = 128, nhid=64, dropout=0.5)
    
    #net = GraphNet.BrainAgePredictor(feature_updater, nfeat=128, nhid=64, dropout=0.5)
    logging.info('  + Number of Model params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    # move the network to GPU/CPU
    net = net.to(device)

    save_model = torch.load(os.path.join(model_path, 'best_model_GCN_MRI.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    logging.info("Model restored from file: {}".format(model_path))

    net.eval()
    for batch_idx, data in enumerate(dataLoader):
        data = data.to(device)
        data.x = data.x.float()

        output = net(data)
        print(output.x.shape, data.x.shape)
        print(output.edge_index.shape, data.edge_index.shape)



            




if __name__ == '__main__':
    main()