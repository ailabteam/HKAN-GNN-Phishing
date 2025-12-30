import pickle
import torch
from torch_geometric.data import HeteroData
from url_features import get_url_features
from collections import Counter

def build():
    with open('data/processed/features_bert.pkl', 'rb') as f:
        data = pickle.load(f)
    
    email_embeddings = data['embeddings']
    labels = data['labels']
    senders_list = data['senders']
    urls_list = data['urls'] # List of lists

    # 1. Map Senders to IDs
    unique_senders = list(set(senders_list))
    sender_to_id = {s: i for i, s in enumerate(unique_senders)}
    
    # 2. Map unique URLs to IDs
    all_urls = [url for sublist in urls_list for url in sublist]
    unique_urls = list(set(all_urls))
    url_to_id = {u: i for i, u in enumerate(unique_urls)}

    # 3. Khởi tạo HeteroData
    graph = HeteroData()

    # --- Thêm Node Features ---
    graph['email'].x = email_embeddings
    graph['email'].y = labels

    # Đặc trưng cho Sender (đơn giản: số email đã gửi)
    sender_counts = Counter(senders_list)
    sender_feats = [[sender_counts[s]] for s in unique_senders]
    graph['sender'].x = torch.tensor(sender_feats, dtype=torch.float)

    # Đặc trưng cho URL (sử dụng hàm từ url_features.py)
    if unique_urls:
        url_feats = torch.stack([get_url_features(u) for u in unique_urls])
        graph['url'].x = url_feats
    else:
        # Trường hợp không có URL nào (hiếm)
        graph['url'].x = torch.empty((0, 8))

    # --- Thêm Edges (Cạnh) ---
    # Email -> Sender
    email_sender_edge = []
    for i, s in enumerate(senders_list):
        email_sender_edge.append([i, sender_to_id[s]])
    graph['email', 'sent_by', 'sender'].edge_index = torch.tensor(email_sender_edge).t().contiguous()

    # Email -> URL
    email_url_edge = []
    for i, urls in enumerate(urls_list):
        for u in urls:
            email_url_edge.append([i, url_to_id[u]])
    
    if email_url_edge:
        graph['email', 'contains', 'url'].edge_index = torch.tensor(email_url_edge).t().contiguous()
    else:
        graph['email', 'contains', 'url'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4. Lưu đồ thị
    torch.save(graph, 'data/processed/hetero_graph.pt')
    
    print("✅ Heterogeneous Graph Built!")
    print(f"Nodes: Email({graph['email'].num_nodes}), Sender({graph['sender'].num_nodes}), URL({graph['url'].num_nodes})")
    print(f"Edges: Email-Sender({graph['email', 'sent_by', 'sender'].num_edges}), Email-URL({graph['email', 'contains', 'url'].num_edges})")

if __name__ == "__main__":
    build()
