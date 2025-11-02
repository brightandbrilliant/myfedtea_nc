# 假设 Model/GNN_Classifier.py
import torch.nn as nn
from graphsage import GraphSAGE
from resmlp import ResMLP

class GNNClassifier(nn.Module):
    def __init__(self, encoder_params, decoder_params, num_classes):
        super().__init__()
        self.feature_extractor = GraphSAGE(**encoder_params)
        feature_dim = encoder_params["output_dim"]
        self.classifier = ResMLP(input_dim=feature_dim, output_dim=num_classes, **decoder_params)

    def forward(self, x, edge_index):
        z = self.feature_extractor(x, edge_index)
        return self.classifier(z)