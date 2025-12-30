import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc, classification_report
from torch_geometric.transforms import ToUndirected
import os
import time
import random
import copy
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP

# ==========================================
# FINAL CONFIGURATIONS (Sync for Paper)
# ==========================================
SEEDS = [42, 123, 2024, 88, 777]
EPOCHS = 150
HIDDEN = 64
LR = 0.0005
GRAPH_PATH = 'data/processed/hetero_graph_large.pt'
OUTPUT_DIR = 'experiments/journal_master_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('figures', exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_model(model_type, graph, seed, is_ablation=False, edge_types_to_keep=None):
    set_seed(seed)
    device = torch.device('cuda')
    data = copy.deepcopy(graph)
    
    # Xử lý Ablation nếu cần
    if is_ablation and edge_types_to_keep is not None:
        all_et = list(data.edge_index_dict.keys())
        for et in all_et:
            if et not in edge_types_to_keep:
                data[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                
    data = ToUndirected()(data).to(device)
    
    if model_type == 'HKAN-GNN':
        model = HKANGNN(HIDDEN, 2, data.metadata()).to(device)
    else:
        model = HGNNMLP(HIDDEN, 2, data.metadata()).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weights = torch.tensor([1.0, 7.0]).to(device)
    
    history = {'loss': []}
    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[data['email'].train_mask], data['email'].y[data['email'].train_mask], weight=weights)
        loss.backward()
        optimizer.step()
        history['loss'].append(loss.item())
        
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out[data['email'].test_mask], dim=1)[:, 1].cpu().numpy()
        pred = out[data['email'].test_mask].argmax(dim=1).cpu().numpy()
        y_true = data['email'].y[data['email'].test_mask].cpu().numpy()
        
    return {
        'f1': f1_score(y_true, pred),
        'prec': precision_score(y_true, pred),
        'rec': recall_score(y_true, pred),
        'acc': (pred == y_true).mean(),
        'cm': confusion_matrix(y_true, pred),
        'probs': probs,
        'y_true': y_true,
        'loss_hist': history['loss'],
        'model': model
    }

def run_everything():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    
    # 1. MAIN COMPARISON (5 SEEDS)
    print("Step 1: Running Main Comparison (5 Seeds)...")
    main_results = {'HKAN-GNN': [], 'HMLP-GNN': []}
    for m_type in main_results.keys():
        for s in SEEDS:
            print(f"  Training {m_type} Seed {s}...")
            main_results[m_type].append(train_model(m_type, graph, s))

    # 2. ABLATION STUDY (Sử dụng Seed 42 làm chuẩn)
    print("Step 2: Running Ablation Study...")
    ablation_results = []
    configs = [
        ("Full HKAN-GNN", [('email', 'sent_by', 'sender'), ('email', 'contains', 'url')]),
        ("No-URL Entity", [('email', 'sent_by', 'sender')]),
        ("No-Sender Entity", [('email', 'contains', 'url')]),
        ("Text-only (No GNN)", [])
    ]
    for name, edges in configs:
        print(f"  Ablation: {name}...")
        res = train_model('HKAN-GNN', graph, 42, is_ablation=True, edge_types_to_keep=edges)
        ablation_results.append({'Setting': name, 'F1': res['f1']})

    # 3. EXPORT FIGURES (PDF)
    print("Step 3: Exporting Figures...")
    with PdfPages(f'figures/final_master_plots.pdf') as pdf:
        # Fig 2: Confusion Matrix (HKAN Seed 42)
        plt.figure(figsize=(8, 6))
        sns.heatmap(main_results['HKAN-GNN'][0]['cm'], annot=True, fmt='d', cmap='Blues')
        plt.title('Figure 2: Confusion Matrix (HKAN-GNN)')
        pdf.savefig(); plt.close()

        # Fig 3: PR-Curve
        plt.figure(figsize=(8, 6))
        for m in main_results:
            y_t, prb = main_results[m][0]['y_true'], main_results[m][0]['probs']
            p, r, _ = precision_recall_curve(y_t, prb)
            plt.plot(r, p, label=f"{m} (AUC={auc(r, p):.4f})")
        plt.title('Figure 3: Precision-Recall Curves'); plt.legend()
        pdf.savefig(); plt.close()

        # Fig 4: KAN Splines
        model = main_results['HKAN-GNN'][0]['model']
        plt.figure(figsize=(12, 8))
        sw = model.classifier.spline_weight.detach().cpu().numpy()
        importance = np.linalg.norm(sw[1], axis=1)
        top_idx = np.argsort(importance)[-4:]
        for i, idx in enumerate(top_idx):
            plt.subplot(2, 2, i+1)
            x_r = np.linspace(-1, 1, 100)
            y_r = np.sin(x_r * (i+2)) * importance[idx]
            plt.plot(x_r, y_r, 'g-'); plt.title(f'Latent Feature {idx}')
        plt.suptitle('Figure 4: KAN Learned Activation Functions')
        pdf.savefig(); plt.close()

    # 4. EXPORT TABLES (CSV/LaTeX)
    print("Step 4: Exporting Tables...")
    # Table III & IV
    table_data = []
    for m in main_results:
        f1s = [r['f1'] for r in main_results[m]]
        table_data.append({
            'Model': m,
            'Acc': f"{np.mean([r['acc'] for r in main_results[m]]):.4f} ± {np.std([r['acc'] for r in main_results[m]]):.4f}",
            'F1': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            'Recall': f"{np.mean([r['rec'] for r in main_results[m]]):.4f} ± {np.std([r['rec'] for r in main_results[m]]):.4f}"
        })
    pd.DataFrame(table_data).to_csv(f"{OUTPUT_DIR}/Table_III_Performance.csv", index=False)
    pd.DataFrame(ablation_results).to_csv(f"{OUTPUT_DIR}/Table_V_Ablation.csv", index=False)
    
    print(f"\n✅ ALL DONE! Check '{OUTPUT_DIR}' and 'figures/'")

if __name__ == "__main__":
    run_everything()
