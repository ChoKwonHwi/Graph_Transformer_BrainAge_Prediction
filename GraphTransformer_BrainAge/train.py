import os
import torch
from torch_geometric.data import DataLoader
import Data
import GraphNet
import logging
import numpy as np
import torch.optim as optim
import argparse
import csv
import warnings
warnings.filterwarnings(action='ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./trained/GCN_cohort')
parser.add_argument('--data_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/data/cohort')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--loss_function', type=str, default='BCE_loss')
parser.add_argument('--output_csv_path', type=str, default='/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/gt_loss.csv')
args = parser.parse_args()

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

    # 데이터 비율 설정 (train: 80%, valid: 10%, test: 10%)
    total_size = len(full_dataset)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size - test_size
    
    #if args.augmentation == True :
    #    transform = transforms.RandomApply([
    #        transforms.RandomHorizontalFlip(p=0.5),
    #        transforms.RandomVerticalFlip(p=0.5),
    #        transforms.RandomRotation((-10, 10)),
    #        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    #    ], p=0.3)

    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # network construction
    logging.info('Initialize network...')
    net = GraphNet.GraphNet(input_dim=args.input_dim)
    logging.info('  + Number of Model params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))

    # move the network to GPU/CPU
    net = net.to(device)

    # hyper parameter
    num_epochs = args.epoch  # 설정한 에폭 수
    learning_rate = args.lr
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # get trained model - scrath train시 주석
    save_model = torch.load(os.path.join(model_path, 'model_GraphTransformer.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    logging.info("Model restored from file: {}".format(model_path))

    net.train()
    pred_age = list()
    true_age = list()
    best_val_loss = float('inf')
    all_loss_train, all_loss_valid = [], []
    # Training loop
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            data.x = data.x.float()
            optimizer.zero_grad()

            output = net(data)

            loss = criterion(output, data.y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.8f}')
        all_loss_train.append(avg_train_loss)
        # Validation loop
        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device)
                data.x = data.x.float()
                output = net(data)
                loss = criterion(output, data.y.float())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        all_loss_valid.append(avg_val_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.8f}')

        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(model_path, 'best_model.pth'))
            logging.info(f'Best model saved at epoch {epoch+1}')
        
        with open(args.output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in zip(all_loss_train, all_loss_valid):
                writer.writerow(row)

'''
# Test code
    for batch_idx, data in enumerate(dataLoader):
        data = data.to(device)

        # get output from network
        with torch.no_grad():
            output = net(data)

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
'''

if __name__ == '__main__':
    main()
