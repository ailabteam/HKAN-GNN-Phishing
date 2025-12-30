import torch
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load đồ thị
    graph = torch.load('data/processed/hetero_graph_large.pt')
    
    # Thêm các cạnh ngược để thông tin chảy hai chiều
    graph = ToUndirected()(graph)
    graph = graph.to(device)

    # Khởi tạo model
    model = HKANGNN(hidden_channels=64, out_channels=2, metadata=graph.metadata()).to(device)
    
    # Tính trọng số cho Loss (vì lớp 1 ít hơn lớp 0)
    # Tỉ lệ 33k:3k -> Trọng số lớp 1 nên gấp ~9-10 lần
    weights = torch.tensor([1.0, 9.0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(101):
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        
        # Chỉ tính loss trên các node Email
        loss = F.cross_entropy(out, graph['email'].y, weight=weights)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == graph['email'].y).sum().item() / graph['email'].num_nodes
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    # Lưu model
    torch.save(model.state_dict(), 'experiments/model_final.pth')

if __name__ == "__main__":
    train()
