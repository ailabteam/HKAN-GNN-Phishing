# HKAN-GNN: Heterogeneous Kolmogorov-Arnold Graph Neural Networks for Explainable Phishing Email Detection

Official implementation of the paper: **"HKAN-GNN: Heterogeneous Kolmogorov-Arnold Graph Neural Networks for Explainable Phishing Email Detection"**, submitted to **IEEE Transactions on Information Forensics and Security (T-IFS)**.

## 🚀 Overview
HKAN-GNN is a pioneering framework that integrates the high-order non-linear approximation power of **Kolmogorov-Arnold Networks (KAN)** with **Heterogeneous Graph Neural Networks (GNN)** to detect phishing emails. 

By modeling the email ecosystem as a heterogeneous graph (Emails, Senders, URLs) and employing a **Hybrid Projection** mechanism with **Gated Concatenation**, HKAN-GNN effectively captures complex interactions between semantic BERT embeddings and sparse infrastructural metadata.

## ✨ Key Contributions
- **Novel Architecture:** First-time integration of KAN layers within a Heterogeneous GNN for cybersecurity.
- **Hybrid Projection:** Utilizes KAN for low-dimensional entities (URL/Sender) and Linear layers for high-dimensional BERT embeddings to optimize efficiency.
- **Explainable Forensics:** Provides intrinsic transparency via learned B-spline activation functions on raw security features.
- **Security Resilience:** Maintains high performance under heavy adversarial semantic noise where content-only models collapse.

## 🛠️ Installation

### Prerequisites
- **OS:** Ubuntu 22.04 LTS
- **GPU:** NVIDIA GeForce RTX 4090 (24GB) recommended
- **CUDA:** 12.1+
- **Python:** 3.11

### Environment Setup
```bash
conda create -n hkan_gnn python=3.11 -y
conda activate hkan_gnn

# Install PyTorch and PyG (Optimized for CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Install additional dependencies
pip install scikit-learn pandas matplotlib networkx tqdm transformers seaborn
```

## 📂 Project Structure
- `src/models/`: Implementation of `HKANGNN`, `KANLayer`, and baselines (`HAN`, `HMLP`).
- `src/master_journal_pipeline.py`: Unified pipeline for multi-seed training and ablation studies.
- `src/robustness_test.py`: Adversarial noise stress-test.
- `src/explain_kan.py`: Visualization of learned spline functions.
- `data/`: Processed heterogeneous graph (36,911 nodes).

## 📈 Experimental Results

### 1. Main Performance Comparison (Mean ± Std over 5 Seeds)
| Model | Accuracy | Phishing F1-Score | Phish Recall |
| :--- | :---: | :---: | :---: |
| BERT-only (Baseline) | 0.4471 ± 0.1859 | 0.2768 ± 0.0741 | **0.9978 ± 0.0023** |
| HAN (SOTA Hetero GNN) | 0.8299 ± 0.0242 | 0.5070 ± 0.0256 | 0.8969 ± 0.0286 |
| HMLP-GNN (Baseline) | 0.9688 ± 0.0028 | 0.8592 ± 0.0108 | 0.9843 ± 0.0016 |
| **HKAN-GNN (Ours)** | **0.9872 ± 0.0013** | **0.9369 ± 0.0057** | 0.9793 ± 0.0022 |

### 2. Gated Ablation Study
| Setting | Phishing F1-Score | Improvement ($\Delta$) |
| :--- | :---: | :---: |
| **Full HKAN-GNN** | **0.9350** | - |
| No-Gating Ablation | 0.9344 | -0.0006 |
| No-URL Entity | 0.9319 | -0.0031 |
| Text-only (No GNN) | 0.1974 | -0.7376 |

### 3. Security Robustness (Adversarial Evasion)
Under heavy semantic noise ($\sigma=0.5$), HKAN-GNN maintains an F1-score of **0.7386**, outperforming the Text-only model (**0.2673**) by a margin of **47.1%**.

## 📊 Visualizations
The system automatically generates high-quality forensic plots in the `figures/` directory:
- `figures/confusion_matrix.pdf`: Classification error patterns.
- `figures/final_master_plots_extended.pdf`: PR-Curves across all baselines.
- `figures/raw_feature_explainability.pdf`: Learned B-splines for 8 URL security features.
- `figures/robustness_analysis.pdf`: Model resilience curve.

## 🚀 How to Run
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. Run full master pipeline (Comparison + Ablation)
python src/master_journal_pipeline.py

# 2. Generate forensic interpretability plots
python src/explain_kan.py

# 3. Run security robustness stress-test
python src/train_final_sota.py
python src/robustness_test.py
```

## ✉️ Contact
**Phuc Hao Do**  
Bonch-Bruevich Saint Petersburg State University of Telecommunications, Russia  
Danang Architecture University, Vietnam  
Email: `do.hf@sut.ru`; `haodp@dau.edu.vn`
Bạn có thể dán nội dung này vào file `README.md` trên GitHub. Đây là một bộ mặt cực kỳ uy tín cho dự án của bạn! Chúc bạn submit thành công rực rỡ!
