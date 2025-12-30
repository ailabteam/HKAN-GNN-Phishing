import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from src.models.kan_layer import KANLayer

class HKANGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        
        # BERT vẫn dùng Linear vì chiều quá lớn (768)
        self.email_proj = nn.Linear(768, hidden_channels)
        
        # URL và Sender dùng KAN vì chiều nhỏ (8 và 1), KAN sẽ học tốt hơn MLP ở đây
        self.url_proj = KANLayer(8, hidden_channels)
        self.sender_proj = KANLayer(1, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')
        
        # BN là "chìa khóa" để KAN hoạt động
        self.bn_email = nn.BatchNorm1d(hidden_channels)
        self.bn_combined = nn.BatchNorm1d(hidden_channels * 2)

        self.classifier = KANLayer(hidden_channels * 2, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Projection với KAN cho các node số
        x_dict['email'] = self.email_proj(x_dict['email'])
        x_dict['url'] = self.url_proj(x_dict['url'])
        x_dict['sender'] = self.sender_proj(x_dict['sender'])
        
        # Chuẩn hóa ngay sau khi proj
        x_dict = {k: torch.tanh(v) for k, v in x_dict.items()} 

        res_email = x_dict['email']

        # 2. Message Passing
        gnn_out = self.conv1(x_dict, edge_index_dict)
        for node_type, x in gnn_out.items():
            x_dict[node_type] = x

        x_dict = {key: F.leaky_relu(x, 0.2) for key, x in x_dict.items()}

        # 3. Concatenation & BatchNorm (Ép dữ liệu về dải của KAN)
        combined = torch.cat([x_dict['email'], res_email], dim=-1)
        combined = self.bn_combined(combined) 

        # 4. KAN Classification
        out = self.classifier(combined)
        return out
