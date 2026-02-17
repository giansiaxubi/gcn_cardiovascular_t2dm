import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNAnomaly(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCNAnomaly, self).__init__()
        # 2-Layer GCN as per Eq (2)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Center c (initialized later during training)
        self.center = None

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        # Output is the embedding H
        return x

    def get_anomaly_score(self, x, edge_index):
        """Eq (3): Squared Euclidean distance to center c."""
        embeddings = self.forward(x, edge_index)
        if self.center is None:
            raise ValueError("Center 'c' not initialized.")
            
        # Broadcast center subtraction
        dist = torch.sum((embeddings - self.center) ** 2, dim=1)
        return dist, embeddings
