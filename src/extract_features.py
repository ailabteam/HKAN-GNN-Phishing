import torch
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

def generate_bert_embeddings(texts, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    embeddings = []
    # Xử lý theo batch để tối ưu VRAM
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        # Padding và Truncation cực kỳ quan trọng cho email dài ngắn khác nhau
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy vector [CLS] (đại diện ngữ nghĩa toàn bộ email)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeddings)
            
    return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    # Đọc dữ liệu 36,911 mẫu
    input_file = 'data/processed/email_data_large.pkl'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit()
        
    df = pd.read_pickle(input_file)
    print(f"Loaded {len(df)} emails for embedding...")

    # Chạy trích xuất
    embeddings = generate_bert_embeddings(df['body'].tolist(), batch_size=128)
    
    # Đóng gói dữ liệu để làm Node cho Graph
    feature_data = {
        'embeddings': embeddings,
        'labels': torch.tensor(df['label'].values),
        'senders': df['sender'].tolist(),
        'urls': df['urls'].tolist()
    }
    
    # Lưu file version "large"
    output_file = 'data/processed/features_bert_large.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(feature_data, f)
        
    print(f"✅ Feature extraction complete!")
    print(f"Final tensor shape: {embeddings.shape}")
    print(f"Saved to: {output_file}")
