import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.manifold import TSNE
from torch_geometric.transforms import ToUndirected
import os
import time
import random
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP

# SETTINGS
SEEDS = [42, 123, 2024, 88, 777] # 5 Seeds cho chuẩn Journal
EPOCHS = 100
HIDDEN = 64
GRAPH_PATH = 'data/processed/hetero_graph_large.pt'
RES_DIR = 'experiments/journal_results'
os.makedirs(RES_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_and_eval(model_type, graph, seed):
    set_seed(seed)
    device = torch.device('cuda')
    data = ToUndirected()(graph).to(device)
    
    if model_type == 'HKAN-GNN':
        model = HKANGNN(HIDDEN, 2, data.metadata()).to(device)
    else:
        model = HGNNMLP(HIDDEN, 2, data.metadata()).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    weights = torch.tensor([1.0, 9.0]).to(device)
    
    history = {'loss': [], 'acc': []}
    
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
        
        # Lấy embeddings để vẽ TSNE
        embeddings = out[data['email'].test_mask].cpu().numpy()
        
    return {
        'metrics': {
            'f1': f1_score(y_true, pred),
            'prec': precision_score(y_true, pred),
            'rec': recall_score(y_true, pred),
            'acc': (pred == y_true).mean()
        },
        'curve': (y_true, probs),
        'cm': confusion_matrix(y_true, pred),
        'history': history,
        'emb': embeddings,
        'y_true': y_true
    }

def run_pipeline():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    results = {}

    for m_type in ['HKAN-GNN', 'HMLP-GNN']:
        print(f"--- Evaluating {m_type} ---")
        seed_runs = []
        for s in SEEDS:
            seed_runs.append(train_and_eval(m_type, graph, s))
        results[m_type] = seed_runs

    # XUẤT PDF MULTI-PAGE
    with PdfPages(f"{RES_DIR}/full_research_plots.pdf") as pdf:
        
        # Figure 1: Robustness (Boxplots)
        plt.figure(figsize=(10, 6))
        f1_data = [[r['metrics']['f1'] for r in results[m]] for m in results]
        plt.boxplot(f1_data, labels=results.keys())
        plt.title('F1-Score Stability (5 Seeds)')
        pdf.savefig(); plt.close()

        # Figure 2: Convergence (Loss Curve của Seed 0)
        plt.figure(figsize=(10, 6))
        for m in results:
            plt.plot(results[m][0]['history']['loss'], label=m)
        plt.title('Training Convergence')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        pdf.savefig(); plt.close()

        # Figure 3: Precision-Recall Curve (Đặc biệt quan trọng cho Phishing)
        plt.figure(figsize=(10, 6))
        for m in results:
            y_true, probs = results[m][0]['curve']
            p, r, _ = precision_recall_curve(y_true, probs)
            plt.plot(r, p, label=f"{m} (AUC={auc(r, p):.4f})")
        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
        pdf.savefig(); plt.close()

        # Figure 4: T-SNE Visualization
        plt.figure(figsize=(10, 8))
        tsne = TSNE(n_components=2)
        emb_2d = tsne.fit_transform(results['HKAN-GNN'][0]['emb'][:1000])
        y_subset = results['HKAN-GNN'][0]['y_true'][:1000]
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_subset, cmap='coolwarm', alpha=0.5)
        plt.title('T-SNE Projection of HKAN-GNN Latent Space')
        pdf.savefig(); plt.close()

    # XUẤT TABLE RA CSV
    final_table = []
    for m in results:
        f1s = [r['metrics']['f1'] for r in results[m]]
        recs = [r['metrics']['rec'] for r in results[m]]
        final_table.append({
            'Model': m,
            'F1 Mean': np.mean(f1s), 'F1 Std': np.std(f1s),
            'Recall Mean': np.mean(recs), 'Recall Std': np.std(recs)
        })
    pd.DataFrame(final_table).to_csv(f"{RES_DIR}/main_performance.csv")
    print(f"✅ Pipeline complete. Figures & Tables saved in {RES_DIR}")

if __name__ == "__main__":
    run_pipeline()
