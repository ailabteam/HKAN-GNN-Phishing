import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def inspect():
    graph = torch.load('data/processed/hetero_graph.pt')
    
    with PdfPages('figures/graph_statistics.pdf') as pdf:
        plt.figure(figsize=(8, 6))
        node_types = ['Email', 'Sender', 'URL']
        node_counts = [graph['email'].num_nodes, graph['sender'].num_nodes, graph['url'].num_nodes]
        
        plt.bar(node_types, node_counts, color=['blue', 'green', 'red'])
        plt.title('Node Distribution in Hetero-Graph')
        plt.ylabel('Count')
        pdf.savefig()
        plt.close()

        # Thống kê về bậc của node (degree)
        plt.figure(figsize=(8, 6))
        # Email connected to how many URLs
        url_per_email = torch.zeros(graph['email'].num_nodes)
        edges = graph['email', 'contains', 'url'].edge_index[0]
        for e in edges: url_per_email[e] += 1
        
        plt.hist(url_per_email.numpy(), bins=20, color='orange', edgecolor='black')
        plt.title('Distribution of URLs per Email')
        plt.xlabel('Number of URLs')
        plt.ylabel('Email Count')
        pdf.savefig()
        plt.close()

    print("✅ Graph inspection report saved to figures/graph_statistics.pdf")

if __name__ == "__main__":
    inspect()
