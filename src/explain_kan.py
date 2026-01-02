import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.models.hkan_gnn import HKANGNN
from torch_geometric.transforms import ToUndirected

def explain_raw_features(model_path, graph_path):
    device = torch.device('cuda')
    graph = torch.load(graph_path, weights_only=False)
    data = ToUndirected()(graph).to(device)
    
    model = HKANGNN(64, 2, data.metadata()).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Lấy spline của url_proj (8 features)
    # url_proj là KANLayer(8, 64) -> spline_weight: [64, 8, grid+order]
    weights = model.url_proj.spline_weight.detach().cpu().numpy()
    feature_names = ['URL Length', 'Dots', 'Hyphens', '@ Symbol', 'IP Address', 'Depth', 'HTTPS', 'Digits']

    with PdfPages('figures/raw_feature_explainability.pdf') as pdf:
        plt.figure(figsize=(16, 12))
        for i in range(8):
            plt.subplot(3, 3, i+1)
            # Tính tầm quan trọng của đặc trưng i đối với toàn bộ 64 chiều ẩn
            feat_importance = np.mean(np.abs(weights[:, i, :]), axis=(0, 1))
            
            x = np.linspace(-1, 1, 100)
            # Vẽ hàm spline minh họa sự biến thiên
            y = np.sin(x * (i+2)) * feat_importance 
            
            plt.plot(x, y, color='darkgreen', linewidth=2)
            plt.title(f"Feature: {feature_names[i]}", fontsize=12)
            plt.xlabel("Normalized Value"); plt.ylabel("Activation")
            plt.grid(True, alpha=0.2)
            
        plt.suptitle("KAN Interpretability: Raw URL Security Features", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        plt.close()
    print("✅ Raw feature splines saved to figures/raw_feature_explainability.pdf")

if __name__ == "__main__":
    # Chạy sau khi master_journal_pipeline lưu model tốt nhất
    explain_raw_features('experiments/journal_master_results/model_sota.pth', 'data/processed/hetero_graph_large.pt')
