import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GraphConv
from .kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        # 1. Chiếu đặc trưng từ không gian BERT (768) và URL (8) về hidden_channels
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        # 2. Lớp KAN-GNN Layer
        # Chúng ta dùng HeteroConv để xử lý các loại cạnh khác nhau
        self.conv1 = HeteroConv({
            ('email', 'sent_by', 'sender'): GraphConv(hidden_channels, hidden_channels),
            ('email', 'contains', 'url'): GraphConv(hidden_channels, hidden_channels),
            # Cạnh ngược để Email nhận thông tin từ hạ tầng
            ('sender', 'rev_sent_by', 'email'): GraphConv(hidden_channels, hidden_channels),
            ('url', 'rev_contains', 'email'): GraphConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        # 3. Bộ phân loại cuối cùng dùng KAN
        self.classifier = KANLayer(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Chiếu đặc trưng
        x_dict['email'] = self.email_proj(x_dict['email'])
        x_dict['url'] = self.url_proj(x_dict['url'])
        x_dict['sender'] = self.sender_proj(x_dict['sender'])

        # Message Passing qua đồ thị
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Phân loại Node Email
        out = self.classifier(x_dict['email'])
        return out
