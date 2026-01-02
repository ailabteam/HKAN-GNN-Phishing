import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from torch_geometric.transforms import ToUndirected
import os
import random
import copy

# Imports models
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP
from src.models.han_baseline import HANModel # Giả sử bạn đặt tên class là HANModel

# ==========================================
# SYNCED CONFIGURATIONS
# ==========================================
SEEDS = [42, 123, 2024, 88, 777]
EPOCHS = 150
HIDDEN = 64
LR = 0.0005
GRAPH_PATH = 'data/processed/hetero_graph_large.pt'
OUTPUT_DIR = 'experiments/journal_master_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_model(model_type, graph, seed, is_ablation=False, edge_types_to_keep=None, use_gating=True):
    set_seed(seed)
    device = torch.device('cuda')
    data = copy.deepcopy(graph)

    # Xử lý Ablation: Xóa cạnh nếu cần
    if is_ablation and edge_types_to_keep is not None:
        all_et = list(data.edge_index_dict.keys())
        for et in all_et:
            if et not in edge_types_to_keep:
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)

    data = ToUndirected()(data).to(device)

    # Khởi tạo mô hình dựa trên loại
    if model_type == 'HKAN-GNN':
        model = HKANGNN(HIDDEN, 2, data.metadata(), use_gating=use_gating).to(device)
    elif model_type == 'HMLP-GNN':
        model = HGNNMLP(HIDDEN, 2, data.metadata()).to(device)
    elif model_type == 'HAN':
        model = HANModel(HIDDEN, 2, data.metadata()).to(device)
    elif model_type == 'BERT-only':
        # BERT-only là HKAN-GNN nhưng không có cạnh
        for et in list(data.edge_index_dict.keys()):
            data[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        model = HKANGNN(HIDDEN, 2, data.metadata(), use_gating=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weights = torch.tensor([1.0, 7.0]).to(device)

    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[data['email'].train_mask], data['email'].y[data['email'].train_mask], weight=weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out[data['email'].test_mask], dim=1)[:, 1].cpu().numpy()
        pred = out[data['email'].test_mask].argmax(dim=1).cpu().numpy()
        y_true = data['email'].y[data['email'].test_mask].cpu().numpy()

    return {
        'f1': f1_score(y_true, pred),
        'rec': recall_score(y_true, pred),
        'acc': (pred == y_true).mean(),
        'probs': probs,
        'y_true': y_true,
        'cm': confusion_matrix(y_true, pred),
        'model': model
    }

def run_everything():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    
    # 1. THÊM HAN VÀ BERT-ONLY VÀO SO SÁNH CHÍNH (5 SEEDS)
    print("Step 1: Running Comparison (HKAN, HMLP, HAN, BERT-only)...")
    model_types = ['HKAN-GNN', 'HMLP-GNN', 'HAN', 'BERT-only']
    all_main_results = {m: [] for m in model_types}
    
    for m_type in model_types:
        for s in SEEDS:
            print(f"  Training {m_type} Seed {s}...")
            all_main_results[m_type].append(train_model(m_type, graph, s))

    # 2. ABLATION STUDY (Seed 42) - Thêm No-Gating
    print("Step 2: Running Ablation Study (including Gating)...")
    ab_edges = [('email', 'sent_by', 'sender'), ('email', 'contains', 'url')]
    ab_configs = [
        ("Full HKAN-GNN", ab_edges, True),
        ("No-Gating Ablation", ab_edges, False),
        ("No-URL Entity", [('email', 'sent_by', 'sender')], True),
        ("Text-only (No GNN)", [], False)
    ]
    ab_table = []
    for name, edges, gate_status in ab_configs:
        print(f"  Ablation: {name}...")
        res = train_model('HKAN-GNN', graph, 42, is_ablation=True, edge_types_to_keep=edges, use_gating=gate_status)
        ab_table.append({'Setting': name, 'F1': res['f1']})

    # 3. EXPORT TABLES
    print("Step 3: Exporting Sync Tables...")
    perf_data = []
    for m in model_types:
        f1s = [r['f1'] for r in all_main_results[m]]
        accs = [r['acc'] for r in all_main_results[m]]
        recs = [r['rec'] for r in all_main_results[m]]
        perf_data.append({
            'Model': m,
            'Acc': f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
            'F1': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            'Recall': f"{np.mean(recs):.4f} ± {np.std(recs):.4f}"
        })
    pd.DataFrame(perf_data).to_csv(f"{OUTPUT_DIR}/Table_III_Performance_Extended.csv", index=False)
    pd.DataFrame(ab_table).to_csv(f"{OUTPUT_DIR}/Table_V_Ablation_Gated.csv", index=False)

    # 4. EXPORT FIGURES (PR-Curve so sánh cả 4)
    with PdfPages('figures/final_master_plots_extended.pdf') as pdf:
        plt.figure(figsize=(10, 7))
        for m in model_types:
            y_t, prb = all_main_results[m][0]['y_true'], all_main_results[m][0]['probs']
            p, r, _ = precision_recall_curve(y_t, prb)
            plt.plot(r, p, label=f"{m} (AUC={auc(r, p):.4f})")
        plt.title('Figure 3: PR-Curves (Extended Baselines)'); plt.legend(); pdf.savefig(); plt.close()

    print(f"\n✅ Pipeline Sync Complete! Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_everything()
