import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv

class HANModel(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # HAN yêu cầu projection ban đầu giống chúng ta
        self.email_proj = nn.Linear(768, hidden_channels)
        self.url_proj = nn.Linear(8, hidden_channels)
        self.sender_proj = nn.Linear(1, hidden_channels)

        self.han_conv = HANConv(hidden_channels, hidden_channels, metadata, heads=2)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'email': self.email_proj(x_dict['email']),
            'url': self.url_proj(x_dict['url']),
            'sender': self.sender_proj(x_dict['sender'])
        }
        # HAN message passing
        out_dict = self.han_conv(x_dict, edge_index_dict)
        return self.classifier(out_dict['email'])
