import re
import torch
import numpy as np

def get_url_features(url):
    # Trích xuất 8 đặc trưng số cơ bản của URL
    features = []
    features.append(len(url)) # Độ dài
    features.append(url.count('.')) # Số dấu chấm
    features.append(url.count('-')) # Số dấu gạch ngang
    features.append(url.count('@')) # Có ký hiệu @ (thường thấy trong phishing)
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0) # Có chứa IP không
    features.append(url.count('/')) # Độ sâu thư mục
    features.append(1 if "https" in url else 0) # Có HTTPS không
    features.append(sum(c.isdigit() for c in url)) # Số lượng chữ số
    
    return torch.tensor(features, dtype=torch.float)

# Hàm này sẽ được gọi trong bước build graph
