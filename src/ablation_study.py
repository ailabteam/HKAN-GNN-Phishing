import torch
import torch.nn.functional as F
import pandas as pd
from src.models.hkan_gnn import HKANGNN
import torch_geometric.transforms as T
import copy
from sklearn.metrics import f1_score
import os

def run_experiment(graph, edge_types_to_keep, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Tạo bản sao đồ thị
    data = copy.deepcopy(graph)
    
    # 2. Xử lý các cạnh
    all_edge_types = list(data.edge_index_dict.keys())
    for et in all_edge_types:
        if et not in edge_types_to_keep:
            # Thay vì xóa hoàn toàn, ta để cạnh rỗng (2, 0) để không lỗi model
            data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 3. Thêm cạnh ngược và chuyển lên GPU
    data = T.ToUndirected()(data).to(device)
    
    # 4. Khởi tạo model
    model = HKANGNN(64, 2, data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    weights = torch.tensor([1.0, 9.0]).to(device)

    # 5. Training Loop
    model.train()
    for epoch in range(101):
        optimizer.zero_grad()
        # forward pass với edge_index_dict (có thể chứa các tensor rỗng)
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[data['email'].train_mask], 
                               data['email'].y[data['email'].train_mask], 
                               weight=weights)
        loss.backward()
        optimizer.step()

    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out[data['email'].test_mask].argmax(dim=1).cpu()
        y = data['email'].y[data['email'].test_mask].cpu()
        f1 = f1_score(y, pred)
        return f1

if __name__ == "__main__":
    graph_path = 'data/processed/hetero_graph_large.pt'
    graph = torch.load(graph_path, weights_only=False)
    results = []

    print("🚀 Starting Ablation Study (Fixed)...")

    # Định nghĩa các loại cạnh gốc để so sánh
    base_edges = [('email', 'sent_by', 'sender'), ('email', 'contains', 'url')]

    # Kịch bản 1: Full
    print("Running: Full HKAN-GNN...")
    f1_full = run_experiment(graph, base_edges, "Full")
    results.append({"Setting": "Full HKAN-GNN", "Phish F1-Score": f1_full})

    # Kịch bản 2: No URL
    print("Running: No-URL Entity...")
    f1_no_url = run_experiment(graph, [('email', 'sent_by', 'sender')], "No-URL")
    results.append({"Setting": "No-URL Entity", "Phish F1-Score": f1_no_url})

    # Kịch bản 3: No Sender
    print("Running: No-Sender Entity...")
    f1_no_sender = run_experiment(graph, [('email', 'contains', 'url')], "No-Sender")
    results.append({"Setting": "No-Sender Entity", "Phish F1-Score": f1_no_sender})

    # Kịch bản 4: Text-only (Giao thức truyền tin bị vô hiệu hóa)
    print("Running: Text-only (No GNN info)...")
    f1_text = run_experiment(graph, [], "Text-only")
    results.append({"Setting": "Text-only (No GNN)", "Phish F1-Score": f1_text})

    df = pd.DataFrame(results)
    print("\n" + "="*35)
    print("      ABLATION STUDY RESULTS")
    print("="*35)
    print(df.to_string(index=False))
    
    df.to_csv('experiments/ablation_results.csv', index=False)
