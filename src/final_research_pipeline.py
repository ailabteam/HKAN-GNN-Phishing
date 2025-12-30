import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch_geometric.transforms import ToUndirected
import os
import random
from src.models.hkan_gnn import HKANGNN
from src.models.hgnn_mlp import HGNNMLP

# SETTINGS
SEEDS = [42, 123, 2024, 88, 777] 
EPOCHS = 200
HIDDEN = 64
LR_KAN = 0.0005 # Giảm LR để KAN học mịn hơn

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
        lr = LR_KAN # KAN cần LR thấp hơn để ổn định splines
    else:
        model = HGNNMLP(HIDDEN, 2, data.metadata()).to(device)
        lr = 0.005
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Thêm Scheduler để hội tụ mịn hơn
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    weights = torch.tensor([1.0, 7.0]).to(device)
    
    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[data['email'].train_mask], data['email'].y[data['email'].train_mask], weight=weights)
        loss.backward()
        # Gradient Clipping giúp KAN không bị sụp đổ
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out[data['email'].test_mask].argmax(dim=1).cpu().numpy()
        y_true = data['email'].y[data['email'].test_mask].cpu().numpy()
        
    metrics = {
        'f1': f1_score(y_true, pred),
        'rec': recall_score(y_true, pred),
        'acc': (pred == y_true).mean()
    }
    return metrics

def run_pipeline():
    graph = torch.load(GRAPH_PATH, weights_only=False)
    final_results = []

    print(f"{'Model':<12} | {'Seed':<5} | {'F1':<8} | {'Recall':<8} | {'Acc':<8}")
    print("-" * 50)

    for m_type in ['HKAN-GNN', 'HMLP-GNN']:
        m_f1s, m_recs, m_accs = [], [], []
        
        for s in SEEDS:
            res = train_and_eval(m_type, graph, s)
            print(f"{m_type:<12} | {s:<5} | {res['f1']:.4f} | {res['rec']:.4f} | {res['acc']:.4f}")
            
            m_f1s.append(res['f1'])
            m_recs.append(res['rec'])
            m_accs.append(res['acc'])
        
        final_results.append({
            'Model': m_type,
            'F1 Mean': np.mean(m_f1s), 'F1 Std': np.std(m_f1s),
            'Recall Mean': np.mean(m_recs), 'Recall Std': np.std(m_recs),
            'Acc Mean': np.mean(m_accs), 'Acc Std': np.std(m_accs)
        })
        print("-" * 50)

    # Lưu Table
    df = pd.DataFrame(final_results)
    df.to_csv(f"{RES_DIR}/main_performance.csv", index=False)
    print(f"\n✅ All seeds finished. Results saved in {RES_DIR}")
    print(df[['Model', 'F1 Mean', 'F1 Std']])

if __name__ == "__main__":
    run_pipeline()
