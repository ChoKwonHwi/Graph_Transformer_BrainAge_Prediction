import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from pygcn.layers import GraphConvolution
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__() 

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNFeatureUpdater(nn.Module):
    def __init__(self, nfeat=128, nhid=64, dropout=0.5):
        """
        GCN for updating node features. No classification performed here.
        """
        super(GCNFeatureUpdater, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)  # First GCN layer
        self.gc2 = GCNConv(nhid, nfeat)  # Second GCN layer to match input feature size
        self.dropout = dropout

    def forward(self, graph_data):
        """
        Updates node features based on graph structure.

        Args:
            graph_data: PyTorch Geometric data object containing:
                        - graph_data.x:ㅈ Node features (num_nodes, nfeat)
                        - graph_data.edge_index: Edge index (2, num_edges)
        
        Returns:
            graph_data: Updated PyTorch Geometric data object with updated node features.
        """
        x, edge_index = graph_data.x, graph_data.edge_index
        
        # Update features using GCN layers
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        
        # Assign updated features back to the graph_data object
        graph_data.x = x
        
        return graph_data

class BrainAgePredictor(nn.Module):
    def __init__(self, feature_updater, nfeat=128, nhid=64, dropout=0.5):
        """
        Combines GCN-based feature updater and downstream task (e.g., brain age prediction).
        """
        super(BrainAgePredictor, self).__init__()
        
        self.feature_updater = feature_updater
        self.fc1 = nn.Linear(nfeat, nhid)  # Fully connected layer for downstream task
        self.fc2 = nn.Linear(nhid, 1)  # Predict brain age as a single output
        self.dropout = dropout

    def forward(self, graph_data):
        """
        Args:
            graph_data: Input graph data containing node features and edge_index.
        
        Returns:
            age_pred: Predicted brain age for the entire graph.
        """
        # Update node features using the GCN feature updater
        updated_graph_data = self.feature_updater(graph_data)
        x = updated_graph_data.x  # Updated node features
        
        # Perform downstream task (e.g., brain age prediction)
        # Global mean pooling (example) to aggregate node features
        graph_embedding = torch.mean(x, dim=0)  # (nfeat,)
        
        # Predict brain age
        x = F.relu(self.fc1(graph_embedding))
        x = F.dropout(x, self.dropout, training=self.training)
        age_pred = self.fc2(x)
        return age_pred


class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, head_num, hidden_dim):
        super(Graph_Transformer, self).__init__()
        #  multi-head self-attention
        self.graph_conv = TransformerConv(input_dim, input_dim//head_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        # feed forward network
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        #  multi-head self-attention
        out1 = self.lin_out(self.graph_conv(x, edge_index))

        # feed forward network
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4


class GraphNet(nn.Module):
    def __init__(self, input_dim, head_num=4, hidden_dim=64, ratio=0.8, gcn_hidden=64):
        super(GraphNet, self).__init__()
        # GCN Layer
        self.gcn = GCN(input_dim, gcn_hidden, input_dim, dropout=0.5)

        self.conv1 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.conv2 = Graph_Transformer(input_dim, head_num, hidden_dim)
        self.conv3 = Graph_Transformer(input_dim, head_num, hidden_dim)

        self.pool1 = SAGPooling(input_dim, ratio)
        self.pool2 = SAGPooling(input_dim, ratio)
        self.pool3 = SAGPooling(input_dim, ratio)

        self.lin1 = nn.Linear(input_dim*2, input_dim)
        self.lin2 = nn.Linear(input_dim, input_dim//2)
        self.lin3 = nn.Linear(input_dim//2, 1)

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim//2)
        self.act = nn.ReLU()

    def forward(self, graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        # print(x.shape)
        # print(edge_index.shape)
        x = self.gcn(x, edge_index)

        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, perm, score = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x, edge_index, _, batch, perm, score = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # multi-level features from read out layers
        x_feature = x1 + x2 + x3

        x_out = self.act(self.bn1(self.lin1(x_feature)))
        x_out = self.act(self.bn2(self.lin2(x_out)))
        x_out = self.lin3(x_out).squeeze(1)

        return x_out