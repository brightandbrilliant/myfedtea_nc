import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(GraphSAGE, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

