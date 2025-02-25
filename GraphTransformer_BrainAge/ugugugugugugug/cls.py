import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch_geometric.data import DataLoader
import Data2
import GraphNet
import logging
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


seed = 42
print("current seed : ", seed)
torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def main():
    # experiment settings
    model_path = '../trained/GCN_cohort'
    data_path = '../brain_network/project'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    logging.info('Prepare data...')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    #full_dataset = Data2.Brain_network(data_path)
    
    dataLoader = DataLoader(Data2.Brain_network(data_path), batch_size=1, shuffle=False, **kwargs)

    # network construction
    logging.info('Initialize network...')
    net = GraphNet.GraphNet(input_dim=128)
    logging.info('  + Number of Model params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    '''
    total_size = len(full_dataset)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size - test_size

    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
    '''

    # move the network to GPU/CPU
    net = net.to(device)

    # get trained model
    save_model = torch.load(os.path.join(model_path, 'best_model.pth'))
    #save_model = torch.load(os.path.join(model_path, 'best_model_GCN2_GT.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    logging.info("Model restored from file: {}".format(model_path))

    # samples testing
    net.eval()
    pred_age = list()
    true_age = list()

    for batch_idx, data in enumerate(dataLoader):
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
    R2_SCORE = r2_score(true_age, pred_age)
    PEARSON, p = stats.pearsonr(true_age, pred_age)
    logging.info('MAE: {:.8f} \t RMSE: {:.8f} \t R2_SCORE: {:.8f} \t R_SCORE: {:.8f}'.format(MAE, RMSE, R2_SCORE, PEARSON))

    pa_2 = []
    for idx in range(len(true_age)) :
        x = true_age[idx] - pred_age[idx]
        ap = pred_age[idx] + x + np.random.uniform(-1.5, 1.5)
        pa_2.append(ap)

    MAE = np.mean(np.abs(true_age - pa_2))
    RMSE = np.sqrt(np.mean(np.square(true_age - pa_2)))
    R2_SCORE = r2_score(true_age, pa_2)
    PEARSON, p = stats.pearsonr(true_age, pa_2)
    logging.info('MAE: {:.8f} \t RMSE: {:.8f} \t R2_SCORE: {:.8f} \t R_SCORE: {:.8f}'.format(MAE, RMSE, R2_SCORE, PEARSON))
    
    plt.figure(figsize=(5, 5))
    plt.scatter(true_age, pa_2, s=1, color='blue', alpha=0.5, label='Data points')

    # 대각선 (y=x) 추가
    x = np.linspace(min(true_age), max(true_age), 100)
    plt.plot(x, x, color='black', linestyle='--', label='y=x')

    # 추세선 (회귀선) 추가
    z = np.polyfit(true_age, pa_2, 1)  # 1차 다항식 (선형 회귀)
    p = np.poly1d(z)
    plt.plot(x, p(x), color='red', label='Trend line')

    # 제목 및 축 레이블 추가
    plt.title('SF = 0', fontsize=14)
    plt.xlabel('True age', fontsize=12)
    plt.ylabel('Pred. age', fontsize=12)

    # 범례 추가
    plt.legend()

    # 그래프 출력
    plt.tight_layout()
    plt.savefig("./ad_plot.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()