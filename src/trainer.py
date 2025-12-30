import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP
from torch_geometric.transforms import ToUndirected

def evaluate(model, graph, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)
        pred = out[mask].argmax(dim=1)
        y = graph['email'].y[mask]
        acc = (pred == y).sum().item() / mask.sum().item()
        # Tính Recall cho lớp Phishing (label 1)
        phish_mask = (y == 1)
        recall = (pred[phish_mask] == 1).sum().item() / phish_mask.sum().item() if phish_mask.sum() > 0 else 0
        return acc, recall

def train_model(model_type='KAN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = torch.load('data/processed/hetero_graph_large.pt', weights_only=False)
    graph = ToUndirected()(graph).to(device)
    
    if model_type == 'KAN':
        model = HKANGNN(64, 2, graph.metadata()).to(device)
    else:
        model = HGNNMLP(64, 2, graph.metadata()).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    weights = torch.tensor([1.0, 9.0]).to(device)
    
    history = {'loss': [], 'val_acc': [], 'val_recall': []}
    
    for epoch in range(101):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = F.cross_entropy(out[graph['email'].train_mask], graph['email'].y[graph['email'].train_mask], weight=weights)
        loss.backward()
        optimizer.step()
        
        acc, recall = evaluate(model, graph, graph['email'].val_mask)
        history['loss'].append(loss.item())
        history['val_acc'].append(acc)
        history['val_recall'].append(recall)
        
        if epoch % 20 == 0:
            print(f"{model_type} - Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")
            
    return history, model

if __name__ == "__main__":
    print("Training KAN-based GNN...")
    kan_hist, kan_model = train_model('KAN')
    print("\nTraining MLP-based GNN (Baseline)...")
    mlp_hist, mlp_model = train_model('MLP')
    
    # Vẽ biểu đồ so sánh lưu vào PDF
    with PdfPages('figures/model_comparison.pdf') as pdf:
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(kan_hist['loss'], label='HKAN-GNN')
        plt.plot(mlp_hist['loss'], label='HMLP-GNN')
        plt.title('Training Loss')
        plt.legend()
        
        # Plot Recall
        plt.subplot(1, 2, 2)
        plt.plot(kan_hist['val_recall'], label='HKAN-GNN')
        plt.plot(mlp_hist['val_recall'], label='HMLP-GNN')
        plt.title('Phishing Recall (Validation)')
        plt.legend()
        
        pdf.savefig()
        plt.close()
    print("\n✅ Comparison complete. Check figures/model_comparison.pdf")
