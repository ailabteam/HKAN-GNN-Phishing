import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.models.hkan_gnn import HKANGNN
from torch_geometric.transforms import ToUndirected

def final_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = torch.load('data/processed/hetero_graph_large.pt', weights_only=False)
    graph = ToUndirected()(graph).to(device)
    
    model = HKANGNN(64, 2, graph.metadata()).to(device)
    model.load_state_dict(torch.load('experiments/model_final.pth'))
    model.eval()

    with torch.no_grad():
        out = model(graph.x_dict, graph.edge_index_dict)
        y_pred = out[graph['email'].test_mask].argmax(dim=1).cpu().numpy()
        y_true = graph['email'].y[graph['email'].test_mask].cpu().numpy()

    # Tính toán các chỉ số
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Phishing'])
    print(report)

    # Vẽ Confusion Matrix vào PDF
    with PdfPages('figures/confusion_matrix.pdf') as pdf:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
        plt.title('Final Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    final_eval()
