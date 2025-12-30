import torch
import pandas as pd
import pickle
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

def generate_bert_embeddings(texts, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy vector [CLS] đại diện cho cả câu/đoạn văn
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeddings)
            
    return torch.cat(embeddings, dim=0)

if __name__ == "__main__":
    df = pd.read_pickle('data/processed/email_data.pkl')
    
    print(f"Generating BERT embeddings for {len(df)} emails on {torch.cuda.get_device_name(0)}...")
    embeddings = generate_bert_embeddings(df['body'].tolist())
    
    # Lưu embeddings và metadata
    feature_data = {
        'embeddings': embeddings,
        'labels': torch.tensor(df['label'].values),
        'senders': df['sender'].tolist(),
        'urls': df['urls'].tolist()
    }
    
    with open('data/processed/features_bert.pkl', 'wb') as f:
        pickle.dump(feature_data, f)
        
    print("✅ Feature extraction complete. Saved to data/processed/features_bert.pkl")
