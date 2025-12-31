import torch
import torch.nn.functional as F
from src.models.hkan_gnn import HKANGNN
from torch_geometric.transforms import ToUndirected
import os

def train_and_save():
    device = torch.device('cuda')
    graph = torch.load('data/processed/hetero_graph_large.pt', weights_only=False)
    data = ToUndirected()(graph).to(device)
    
    # Khởi tạo đúng kiến trúc SOTA (hidden=64, Hybrid Projection, BN)
    model = HKANGNN(64, 2, data.metadata()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    weights = torch.tensor([1.0, 7.0]).to(device)

    print("🚀 Training Final SOTA Model (F1 ~0.947)...")
    for epoch in range(151):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[data['email'].train_mask], data['email'].y[data['email'].train_mask], weight=weights)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

    # LƯU VÀO FILE RIÊNG ĐỂ KHÔNG NHẦM LẪN
    os.makedirs('experiments', exist_ok=True)
    torch.save(model.state_dict(), 'experiments/model_sota_final.pth')
    print("✅ SOTA Model saved as experiments/model_sota_final.pth")

if __name__ == "__main__":
    train_and_save()
