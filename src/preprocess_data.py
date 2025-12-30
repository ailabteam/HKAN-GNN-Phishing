import os
import mailbox
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from email.utils import parseaddr
import pickle

def extract_urls(text):
    if not text: return []
    # Regex chuẩn hơn để bắt URL và loại bỏ nhiễu
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text.lower())
    return list(set(urls)) # Lấy unique URLs trong 1 email

def clean_text(text):
    if not text: return ""
    # Loại bỏ các khoảng trắng thừa và ký tự đặc biệt gây nhiễu BERT
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_enron_folder(base_path, limit=10000):
    data = []
    print("Parsing Enron Dataset...")
    files_list = []
    for root, _, files in os.walk(base_path):
        for f in files:
            files_list.append(os.path.join(root, f))
    
    # Chỉ lấy các file trong các thư mục quan trọng để tránh thư mục rác
    for file_path in tqdm(files_list[:limit]):
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                # Tách header và body đơn giản
                parts = content.split('\n\n', 1)
                header = parts[0]
                body = clean_text(parts[1]) if len(parts) > 1 else ""
                
                # Trích xuất Sender từ header
                sender = re.search(r'From: (.*)', header)
                sender = sender.group(1) if sender else "unknown"
                sender = parseaddr(sender)[1] # Lấy email address sạch
                
                data.append({
                    'sender': sender,
                    'body': body,
                    'urls': extract_urls(body),
                    'label': 0 # Benign
                })
        except: continue
    return data

def parse_nazario_mbox(mbox_paths):
    data = []
    print("Parsing Nazario Phishing Mboxes...")
    for path in mbox_paths:
        if not os.path.exists(path): continue
        mbox = mailbox.mbox(path)
        for message in tqdm(mbox):
            try:
                sender = message.get('From', 'unknown')
                sender = parseaddr(sender)[1]
                
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode('latin-1')
                            break
                else:
                    body = message.get_payload(decode=True).decode('latin-1')
                
                body = clean_text(body)
                data.append({
                    'sender': sender,
                    'body': body,
                    'urls': extract_urls(body),
                    'label': 1 # Phishing
                })
            except: continue
    return data

if __name__ == "__main__":
    # Chạy xử lý
    enron_data = parse_enron_folder('data/raw/maildir', limit=10000)
    nazario_data = parse_nazario_mbox([f'data/raw/nazario_phishing_{i}.mbox' for i in range(4)])
    
    df = pd.DataFrame(enron_data + nazario_data)
    
    # Loại bỏ email không có body hoặc quá ngắn (nhiễu)
    df = df[df['body'].str.len() > 10]
    df = df.drop_duplicates(subset=['body'])
    
    # Lưu ra file processed
    os.makedirs('data/processed', exist_ok=True)
    df.to_pickle('data/processed/email_data.pkl')
    print(f"✅ Preprocessing done. Total emails: {len(df)}")
    print(f"Distribution: \n{df['label'].value_counts()}")
