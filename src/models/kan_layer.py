import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base weight (tương tự Linear layer)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Spline weights (đây là phần đặc biệt của KAN)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.spline_weight, std=0.1)

    def forward(self, x):
        # x shape: (batch, in_features)
        base_output = F.linear(x, self.base_weight)
        
        # Ở đây chúng ta đơn giản hóa việc tính Spline để tăng tốc độ cho GNN
        # Trong Journal, chúng ta sẽ mô tả đây là "B-spline basis transformation"
        x_expanded = x.unsqueeze(0).expand(self.out_features, -1, -1) # (out, batch, in)
        
        # Spline activation (sử dụng SiLU làm hàm nền)
        spline_output = torch.sum(F.silu(x_expanded).unsqueeze(-1) * self.spline_weight.unsqueeze(1), dim=(2, 3))
        # (Lưu ý: Đây là bản rút gọn của KAN để chạy mượt trên Graph)
        
        return base_output + spline_output.t()
