import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import Data2
import utils
import ConvNet
import GraphNet
import logging
import numpy as np
import shutil
from torch_geometric.data import Data

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def BrainNetwork_single_modal(modality):
    # experiment settings
    model_path = '../trained/CAE'
    data_path = '../data/ADNI/NC'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data preparation
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    dataLoader = DataLoader(Data2.Brain_image(data_path, modality), batch_size=1, shuffle=False, **kwargs)

    # network construction
    net = ConvNet.Feature_Extraction(nChannels=16)
    net2 = GraphNet.GCNFeatureUpdater(nfeat = 128, nhid=64)

    # move the network to GPU/CPU
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    torch.cuda.empty_cache()

    net2 = torch.nn.DataParallel(net2)
    net2 = net2.to(device)

    # get trained model
    if modality == 'DTI' :
        save_model = torch.load(os.path.join(model_path, 'model_MRI_' + modality + '_more.pth'))
        save_model2 = torch.load('/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/trained/GCN_cohort/best_model_GCN_DTI.pth')
    else :
        save_model = torch.load(os.path.join(model_path, 'model_' + modality + '_CAE_more.pth'))
        save_model2 = torch.load('/nasdata4/kwonhwi/GraphTransformer/GraphTransformer_BrainAge/trained/GCN_cohort/best_model_GCN_MRI.pth')


    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

    model_dict2 = net2.state_dict()
    state_dict2 = {k: v for k, v in save_model2.items() if k in model_dict2.keys()}
    model_dict2.update(state_dict2)
    net2.load_state_dict(model_dict2)    

    # load ROI template
    template_s = np.load('../template/aal90_template_20x24x20.npy')
    template_m = np.load('../template/aal90_template_40x48x40.npy')

    # define file paths
    node_path = '../brain_network/ADNI/NC/' + modality + '/node_feature'
    adjacency_path = node_path.replace('node_feature', 'adjacency_matrix')
    age_path = node_path.replace('node_feature', 'age')
    '''
    node_path = '../brain_network/project/' + modality + '/node_feature'
    adjacency_path = node_path.replace('node_feature', 'adjacency_matrix')
    age_path = node_path.replace('node_feature', 'age')

    node_path2 = '../brain_network/project2/' + modality + '/node_feature'
    adjacency_path2 = node_path2.replace('node_feature', 'adjacency_matrix')
    age_path2 = node_path2.replace('node_feature', 'age')
    '''
    # reset folders
    shutil.rmtree(node_path)
    os.mkdir(node_path)
    shutil.rmtree(adjacency_path)
    os.mkdir(adjacency_path)
    shutil.rmtree(age_path)
    os.mkdir(age_path)
    '''
    # reset folders
    shutil.rmtree(node_path2)
    os.mkdir(node_path2)
    shutil.rmtree(adjacency_path2)
    os.mkdir(adjacency_path2)
    shutil.rmtree(age_path2)
    os.mkdir(age_path2)
    '''
    target_shape = (1, 1, 160, 192, 160)

    # brain network construction
    net.eval()
    net2.eval()

    for batch_idx, (image, label, name) in enumerate(dataLoader):
        image = image.float().to(device)
        #print(name, image.shape, label.shape)
        if image.shape != torch.Size(target_shape) :
            image = F.interpolate(image, size=target_shape[2:], mode='trilinear', align_corners=False)
        #print(image.shape)
        # obtain feature maps from network
        with torch.no_grad():
            feature_map_s, feature_map_m = net(image)
        torch.cuda.empty_cache()

        feature_map_s = feature_map_s.cpu().detach().numpy().squeeze()
        feature_map_m = feature_map_m.cpu().detach().numpy().squeeze()

        # get ROI feature as node features
        roi_feature = utils.get_roi_feature(feature_map_s, feature_map_m, template_s, template_m)
        #np.save(os.path.join(node_path, name[0]), roi_feature)

        # get adjacency matrix
        distance_matrix = np.load('../template/aal90_distance_matrix.npy')
        adjacency_matrix = utils.get_adjacency_matrix(roi_feature, distance_matrix, k_num=8)
        #np.save(os.path.join(adjacency_path, name[0]), adjacency_matrix)
        
        # get subject age
        #np.save(os.path.join(age_path, name[0]), label[0])
        
        e_ind = utils.get_edge(adjacency_matrix)
        edge_index2 = torch.from_numpy(e_ind)
        node_feature2 = torch.from_numpy(roi_feature)
        
        graph_data = Data(x=node_feature2, y=label[0], edge_index=edge_index2)
        graph_data = graph_data.to(device)
        graph_data.x = graph_data.x.float()

        #print(roi_feature.shape, adjacency_matrix.shape, label[0])
        gcn_output = net2(graph_data)

        #print(gcn_output.x.shape, gcn_output.edge_index.shape, gcn_output.y)
        
        edge_indx2 = utils.get_adjacency_matrix(gcn_output.x.cpu().detach(), distance_matrix, k_num=8)
        #print(edge_indx2.shape)

        np.save(os.path.join(node_path, name[0]), gcn_output.x.cpu().detach())
        np.save(os.path.join(adjacency_path, name[0]), edge_indx2)
        np.save(os.path.join(age_path, name[0]), gcn_output.y.cpu().detach())

    logging.info('Brain network construction of {} modality is completed.'.format(modality))


def BrainNetwork_multi_modal():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define file paths
    mri_path = '../brain_network/ADNI/NC/MRI'
    dti_path = '../brain_network/ADNI/NC/DTI'
    fusion_path = '../brain_network/ADNI/NC/Fusion'

    mri_node_path = os.path.join(mri_path, 'node_feature')
    dti_node_path = os.path.join(dti_path, 'node_feature')
    fusion_node_path = os.path.join(fusion_path, 'node_feature')

    mri_adjacency_path = os.path.join(mri_path, 'adjacency_matrix')
    dti_adjacency_path = os.path.join(dti_path, 'adjacency_matrix')
    fusion_adjacency_path = os.path.join(fusion_path, 'adjacency_matrix')

    mri_age_path = os.path.join(mri_path, 'age')
    dti_age_path = os.path.join(dti_path, 'age')
    fusion_age_path = os.path.join(fusion_path, 'age')

    # reset folders
    shutil.rmtree(fusion_node_path)
    os.mkdir(fusion_node_path)
    shutil.rmtree(fusion_adjacency_path)
    os.mkdir(fusion_adjacency_path)
    shutil.rmtree(fusion_age_path)
    os.mkdir(fusion_age_path)

    # combine node features
    sub_dir = os.listdir(mri_node_path)
    for name in sub_dir:
        mri_node_feature = np.load(os.path.join(mri_node_path, name))
        dti_node_feature = np.load(os.path.join(dti_node_path, name))
        fusion_node_feature = np.concatenate((mri_node_feature, dti_node_feature), axis=0)
        np.save(os.path.join(fusion_node_path, name), fusion_node_feature)

    # combine adjacency matrix
    sub_dir = os.listdir(mri_adjacency_path)
    for name in sub_dir:
        mri_adj_matrix = np.load(os.path.join(mri_adjacency_path, name))
        dti_adj_matrix = np.load(os.path.join(dti_adjacency_path, name))
        fusion_adj_matrix = utils.combine_modality_matrix(mri_adj_matrix, dti_adj_matrix, k_num=8)
        np.save(os.path.join(fusion_adjacency_path, name), fusion_adj_matrix)

    # get subject age
    sub_dir = os.listdir(mri_age_path)
    for name in sub_dir:
        shutil.copy(os.path.join(dti_age_path, name), os.path.join(fusion_age_path, name))

    logging.info('Multimodal brain network construction is completed.')


if __name__ == '__main__':
    # brain network construction for MRI 
    BrainNetwork_single_modal(modality='MRI')
    print("MRI clear")
    torch.cuda.empty_cache()
    # brain network construction for DTI
    BrainNetwork_single_modal(modality='DTI')
    print("DTI clear")
    torch.cuda.empty_cache()
    # multimodal brain network construction
    BrainNetwork_multi_modal()
    print("Multi clear")
