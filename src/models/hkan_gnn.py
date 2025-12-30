import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from src.models.kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')
        
        self.gate = nn.Parameter(torch.tensor([0.5])) 
        self.classifier = KANLayer(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Feature Projection
        x_dict = {
            node_type: self.email_proj(x) if node_type == 'email' else
                       self.url_proj(x) if node_type == 'url' else
                       self.sender_proj(x)
            for node_type, x in x_dict.items()
        }

        # Lưu lại đặc trưng Email gốc cho Residual
        res_email = x_dict['email']

        # 2. Message Passing
        # Gộp output của conv1 vào x_dict hiện tại để tránh KeyError
        gnn_out = self.conv1(x_dict, edge_index_dict)
        for node_type, x in gnn_out.items():
            x_dict[node_type] = x

        # 3. Activation
        x_dict = {key: F.leaky_relu(x, 0.2) for key, x in x_dict.items()}

        # 4. Gated Residual Connection
        alpha = torch.sigmoid(self.gate)
        # Bây giờ 'email' chắc chắn tồn tại trong x_dict
        x_dict['email'] = alpha * x_dict['email'] + (1 - alpha) * res_email

        # 5. KAN Classification
        out = self.classifier(x_dict['email'])
        return out
