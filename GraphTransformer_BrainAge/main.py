import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch_geometric.data import DataLoader
import Data
import GraphNet
import logging
import numpy as np
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main():
    # experiment settings
    model_path = './trained/GCN'
    data_path = './brain_network/project'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    logging.info('Prepare data...')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    full_dataset = Data.Brain_network(data_path)
    
    #dataLoader = DataLoader(Data.Brain_network(data_path), batch_size=1, shuffle=False, **kwargs)

    # network construction
    logging.info('Initialize network...')
    net = GraphNet.GraphNet(input_dim=128)
    logging.info('  + Number of Model params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    
    total_size = len(full_dataset)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size - test_size

    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
    # move the network to GPU/CPU
    net = net.to(device)

    # get trained model
    save_model = torch.load(os.path.join(model_path, 'model_GraphTransformer.pth'))
    save_model = torch.load(os.path.join(model_path, 'best_model.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    logging.info("Model restored from file: {}".format(model_path))

    # samples testing
    net.eval()
    pred_age = list()
    true_age = list()

    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        data.x = data.x.float()
        # get output from network
        with torch.no_grad():
            output = net(data)
        #print(output)
        pred_age.append(output.item())
        true_age.append(data.y.item())

    pred_age = np.array(pred_age)
    true_age = np.array(true_age)

    # print the prediction results
    print('The estimated age of testing samples are:')
    print(pred_age)
    print('The true age of testing samples are:')
    print(true_age)

    # calculate performance indicators
    MAE = np.mean(np.abs(true_age - pred_age))
    RMSE = np.sqrt(np.mean(np.square(true_age - pred_age)))
    logging.info('MAE: {:.8f} \t RMSE: {:.8f}'.format(MAE, RMSE))


if __name__ == '__main__':
    main()