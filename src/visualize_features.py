import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def plot_tsne():
    with open('data/processed/features_bert.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X = data['embeddings'].numpy()
    y = data['labels'].numpy()

    print("Running TSNE (this might take a while)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X[:2000]) # Lấy 2000 mẫu để vẽ cho nhanh
    y_subset = y[:2000]

    with PdfPages('figures/feature_analysis.pdf') as pdf:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_subset, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, ticks=[0, 1], label='Class')
        plt.title('TSNE Visualization of BERT Email Embeddings')
        plt.xlabel('TSNE-1')
        plt.ylabel('TSNE-2')
        pdf.savefig()
        plt.close()
    
    print("✅ Visualization saved to figures/feature_analysis.pdf")

if __name__ == "__main__":
    plot_tsne()
