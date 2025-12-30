# HKAN-GNN: Heterogeneous Kolmogorov-Arnold Graph Neural Networks for Explainable Phishing Email Detection

Official implementation of the paper: **"HKAN-GNN: Heterogeneous Kolmogorov-Arnold Graph Neural Networks for Explainable Phishing Email Detection"**.

## 🚀 Overview
HKAN-GNN is a novel framework that combines the high-order non-linear approximation power of **Kolmogorov-Arnold Networks (KAN)** with the structural learning capabilities of **Heterogeneous Graph Neural Networks (GNN)** to detect phishing emails. 

By modeling emails, senders, and URLs as a heterogeneous graph and employing a **Gated Concatenation** mechanism with BERT embeddings, HKAN-GNN achieves state-of-the-art performance while providing intrinsic interpretability through learned B-spline functions.

## ✨ Key Features
- **Heterogeneous Graph Construction:** Captures relationships between Email content, Sender behavior, and URL infrastructure.
- **KAN-based Classification:** Replaces traditional MLPs with KAN layers for superior non-linear fitting and explainability.
- **Semantic-Structural Fusion:** Uses a Gated Residual connection to adaptively fuse BERT embeddings with graph signals.
- **High Performance:** Achieves **0.9471 F1-score** and **96.99% Accuracy** on a large-scale dataset (36,911 samples).

## 🛠️ Installation

### Prerequisites
- OS: Ubuntu 22.04 LTS
- GPU: 2x NVIDIA GeForce RTX 4090 (24GB)
- CUDA: 12.1+

### Environment Setup
```bash
conda create -n hkan_gnn python=3.11 -y
conda activate hkan_gnn

# Install PyTorch and PyG
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install additional dependencies
pip install scikit-learn pandas matplotlib networkx tqdm transformers seaborn
```

## 📂 Project Structure
- `src/`: Core source code (models, preprocessing, trainers).
- `data/`: Raw and processed datasets (Enron, Nazario, SpamAssassin).
- `figures/`: Generated plots for research analysis.
- `experiments/`: Logs, saved models, and master results.

## 📈 Experimental Results

### Performance Comparison (Mean ± Std over 5 Seeds)
| Model | Accuracy | Phishing F1-Score | Phish Recall |
| :--- | :---: | :---: | :---: |
| HMLP-GNN (Baseline) | 0.9687 ± 0.0028 | 0.8591 ± 0.0107 | 0.9843 ± 0.0016 |
| **HKAN-GNN (Proposed)** | **0.9895 ± 0.0014** | **0.9471 ± 0.0068** | **0.9739 ± 0.0023** |

### Ablation Study
| Setting | Phishing F1-Score |
| :--- | :---: |
| **Full HKAN-GNN** | **0.9475** |
| No-URL Entity | 0.9427 |
| No-Sender Entity | 0.9470 |
| Text-only (No GNN) | 0.9388 |

## 📊 Visualizations
Detailed visualizations including **Confusion Matrices**, **PR-Curves**, and **KAN Spline Activations** can be found in the `figures/` directory.

