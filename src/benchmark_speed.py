import time
import torch
import torch.nn as nn
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP
from torch_geometric.transforms import ToUndirected
import warnings

warnings.filterwarnings("ignore", category=UserWarning) # Tắt warning cho sạch log

def benchmark():
    device = torch.device('cuda')
    graph = torch.load('data/processed/hetero_graph_large.pt', weights_only=False)
    graph = ToUndirected()(graph).to(device)
    
    metadata = graph.metadata()
    kan_model = HKANGNN(64, 2, metadata).to(device)
    mlp_model = HGNNMLP(64, 2, metadata).to(device)

    # Đo số lượng tham số (Parameter Count) - Rất quan trọng cho Paper
    kan_params = sum(p.numel() for p in kan_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())

    # Warm up GPU
    for _ in range(20):
        _ = kan_model(graph.x_dict, graph.edge_index_dict)
        _ = mlp_model(graph.x_dict, graph.edge_index_dict)

    # Measure KAN Latency
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = kan_model(graph.x_dict, graph.edge_index_dict)
    torch.cuda.synchronize()
    kan_time = (time.time() - start) / 100

    # Measure MLP Latency
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = mlp_model(graph.x_dict, graph.edge_index_dict)
    torch.cuda.synchronize()
    mlp_time = (time.time() - start) / 100

    print("\n" + "="*40)
    print(f"{'Metric':<20} | {'HKAN-GNN':<10} | {'HMLP-GNN':<10}")
    print("-" * 40)
    print(f"{'Total Parameters':<20} | {kan_params:<10} | {mlp_params:<10}")
    print(f"{'Inference Time (ms)':<20} | {kan_time*1000: <10.2f} | {mlp_time*1000:<10.2f}")
    print("="*40)

if __name__ == "__main__":
    benchmark()
