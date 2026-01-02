import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from src.models.kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, use_gating=True):
        super().__init__()
        self.use_gating = use_gating
        
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = KANLayer(8, hidden_channels)
        self.sender_proj = KANLayer(1, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: GATConv(hidden_channels, hidden_channels, heads=2, concat=False, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        self.bn_combined = nn.BatchNorm1d(hidden_channels * 2)
        self.gate = nn.Parameter(torch.tensor([0.0])) 
        self.classifier = KANLayer(hidden_channels * 2, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Hybrid Projection
        x_dict = {
            'email': self.email_proj(x_dict['email']),
            'url': self.url_proj(x_dict['url']),
            'sender': self.sender_proj(x_dict['sender'])
        }
        x_dict = {k: torch.tanh(v) for k, v in x_dict.items()}
        bert_info = x_dict['email']

        # 2. Message Passing
        try:
            gnn_out = self.conv1(x_dict, edge_index_dict)
            graph_info = gnn_out['email']
        except:
            graph_info = torch.zeros_like(bert_info)

        graph_info = F.leaky_relu(graph_info, 0.2)

        # 3. Gated or Direct Fusion
        if self.use_gating:
            alpha = torch.sigmoid(self.gate)
            combined = torch.cat([alpha * graph_info, (1 - alpha) * bert_info], dim=-1)
        else:
            # Direct concatenation (No gating)
            combined = torch.cat([graph_info, bert_info], dim=-1)

        combined = self.bn_combined(combined)
        out = self.classifier(combined)
        return out
