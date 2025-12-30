# Model này y hệt HKANGNN nhưng thay KANLayer bằng nn.Linear (MLP)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv

class HGNNMLP(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: GraphConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        # Thay KAN bằng Linear chuẩn
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict['email'] = self.email_proj(x_dict['email'])
        x_dict['url'] = self.url_proj(x_dict['url'])
        x_dict['sender'] = self.sender_proj(x_dict['sender'])
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.classifier(x_dict['email'])
