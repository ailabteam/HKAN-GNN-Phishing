import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from src.models.kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: GATConv(hidden_channels, hidden_channels, heads=2, concat=False, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')
        
        # KAN bây giờ nhận đầu vào là (hidden_channels * 2) do phép Concatenation
        self.classifier = KANLayer(hidden_channels * 2, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Projection
        x_dict = {
            node_type: self.email_proj(x) if node_type == 'email' else
                       self.url_proj(x) if node_type == 'url' else
                       self.sender_proj(x)
            for node_type, x in x_dict.items()
        }

        # Giữ lại bản sao BERT
        bert_info = x_dict['email']

        # 2. Message Passing
        try:
            gnn_out = self.conv1(x_dict, edge_index_dict)
            graph_info = gnn_out['email']
        except:
            graph_info = torch.zeros_like(bert_info)

        graph_info = F.leaky_relu(graph_info, 0.2)

        # 3. CONCATENATION (Quyết định thắng bại)
        # Nối BERT và Graph info lại: 64 + 64 = 128 chiều
        combined_info = torch.cat([bert_info, graph_info], dim=-1)

        # 4. KAN Classification
        out = self.classifier(combined_info)
        return out
