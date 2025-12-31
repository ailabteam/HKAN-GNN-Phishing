import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.models.hkan_gnn import HKANGNN
from torch_geometric.transforms import ToUndirected
from sklearn.metrics import f1_score
import copy
import os

def run_robustness_test():
    device = torch.device('cuda')
    graph_path = 'data/processed/hetero_graph_large.pt'
    model_path = 'experiments/model_final.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return

    # 1. Load Graph và chuyển thành Vô hướng (giống lúc train)
    graph = torch.load(graph_path, weights_only=False)
    data = ToUndirected()(graph).to(device)
    
    # 2. Khởi tạo và Load Model Final
    # Đảm bảo hidden_channels=64 đúng như cấu hình cuối cùng của bạn
    model = HKANGNN(64, 2, data.metadata()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Định nghĩa các mức nhiễu (Noise Intensity - Sigma)
    # Chúng ta thử từ 0.0 (không nhiễu) đến 1.0 (nhiễu cực nặng)
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    hkan_f1s = []
    text_only_f1s = []

    print(f"🚀 Running Robustness Test using {model_path}...")

    for sigma in noise_levels:
        # Thêm nhiễu Gaussian vào Email BERT features
        noisy_data = copy.deepcopy(data)
        if sigma > 0:
            noise = torch.randn_like(noisy_data['email'].x) * sigma
            noisy_data['email'].x += noise

        with torch.no_grad():
            # Kịch bản 1: Full HKAN-GNN (Sử dụng toàn bộ đồ thị)
            out_full = model(noisy_data.x_dict, noisy_data.edge_index_dict)
            pred_full = out_full[noisy_data['email'].test_mask].argmax(dim=1).cpu()
            y_true = noisy_data['email'].y[noisy_data['email'].test_mask].cpu()
            hkan_f1s.append(f1_score(y_true, pred_full))

            # Kịch bản 2: Giả lập Text-only (Xóa sạch các cạnh trong đồ thị)
            # Điều này buộc model chỉ dựa vào BERT feature đã bị nhiễu
            noisy_data_no_edges = copy.deepcopy(noisy_data)
            for et in list(noisy_data_no_edges.edge_index_dict.keys()):
                noisy_data_no_edges[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
            out_text = model(noisy_data_no_edges.x_dict, noisy_data_no_edges.edge_index_dict)
            pred_text = out_text[noisy_data_no_edges['email'].test_mask].argmax(dim=1).cpu()
            text_only_f1s.append(f1_score(y_true, pred_text))
        
        print(f"  Sigma: {sigma:.1f} | Full F1: {hkan_f1s[-1]:.4f} | Text-only F1: {text_only_f1s[-1]:.4f}")

    # 4. Xuất đồ thị so sánh
    with PdfPages('figures/robustness_analysis.pdf') as pdf:
        plt.figure(figsize=(9, 6))
        plt.plot(noise_levels, hkan_f1s, 'o-', label='Full HKAN-GNN (Semantic + Structural)', color='#1f77b4', linewidth=2.5)
        plt.plot(noise_levels, text_only_f1s, 's--', label='Text-only (Semantic only)', color='#d62728', linewidth=2)
        
        plt.title('Model Resilience Against Adversarial Semantic Noise', fontsize=14)
        plt.xlabel('Noise Intensity ($\sigma$)', fontsize=12)
        plt.ylabel('Phishing F1-Score', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower left', fontsize=11)
        
        # Thêm annotation giải thích
        plt.annotate('Graph structure maintains\nperformance', xy=(0.3, hkan_f1s[3]), xytext=(0.35, 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))
        
        pdf.savefig()
        plt.close()

    print("\n✅ Figure generated: figures/robustness_analysis.pdf")

if __name__ == "__main__":
    run_robustness_test()
