import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv # Dùng SAGEConv thay cho GraphConv vì nó mạnh hơn
from src.models.kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        # Sử dụng SAGEConv - thuật toán mạnh mẽ hơn để tổng hợp tin nhắn
        self.conv1 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.classifier = KANLayer(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Chiếu đặc trưng
        x_dict = {
            node_type: self.email_proj(x) if node_type == 'email' else
                       self.url_proj(x) if node_type == 'url' else
                       self.sender_proj(x)
            for node_type, x in x_dict.items()
        }

        # Lưu lại đặc trưng Email trước khi qua GNN (Residual)
        res_email = x_dict['email']

        # Message Passing
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(x, 0.2) for key, x in x_dict.items()}

        # CỘNG KẾT NỐI TẮT (Residual Connection)
        # Email mới = Email sau GNN + Email BERT gốc
        x_dict['email'] = x_dict['email'] + res_email

        out = self.classifier(x_dict['email'])
        return out
