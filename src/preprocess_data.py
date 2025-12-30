import os
import mailbox
import re
import pandas as pd
from tqdm import tqdm
from email.utils import parseaddr

def extract_urls(text):
    if not text: return []
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return list(set(re.findall(url_pattern, text.lower())))

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_raw_file(file_path):
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
            parts = content.split('\n\n', 1)
            header = parts[0]
            body = clean_text(parts[1]) if len(parts) > 1 else ""
            sender = re.search(r'From: (.*)', header)
            sender = parseaddr(sender.group(1))[1] if sender else "unknown"
            return sender, body
    except: return None, None

def run():
    data = []
    raw_path = 'data/raw'
    sa_path = os.path.join(raw_path, 'sa_data')
    
    # 1. ENRON (Benign) - Lấy 50,000 mẫu để tạo quy mô lớn
    print("Processing Enron (Target 50k)...")
    enron_base = os.path.join(raw_path, 'maildir')
    count = 0
    for root, _, files in os.walk(enron_base):
        for f in files:
            if count >= 50000: break
            s, b = parse_raw_file(os.path.join(root, f))
            if b and len(b) > 30:
                data.append({'sender': s, 'body': b, 'urls': extract_urls(b), 'label': 0})
                count += 1
        if count >= 50000: break

    # 2. SPAMASSASSIN (Phân loại Ham/Spam)
    print("Processing SpamAssassin...")
    if os.path.exists(sa_path):
        for folder in os.listdir(sa_path):
            folder_p = os.path.join(sa_path, folder)
            if not os.path.isdir(folder_p): continue
            
            # Gán nhãn: Thư mục chứa 'spam' -> nhãn 1, còn lại -> nhãn 0
            label = 1 if 'spam' in folder.lower() else 0
            print(f"  - Folder '{folder}' -> Label {label}")
            
            for f in tqdm(os.listdir(folder_p)):
                s, b = parse_raw_file(os.path.join(folder_p, f))
                if b: data.append({'sender': s, 'body': b, 'urls': extract_urls(b), 'label': label})

    # 3. NAZARIO (Phishing - Nhãn 1)
    print("Processing Nazario...")
    for i in range(4):
        mbox_p = os.path.join(raw_path, f'nazario_phishing_{i}.mbox')
        if os.path.exists(mbox_p):
            mbox = mailbox.mbox(mbox_p)
            for msg in tqdm(mbox):
                try:
                    s = parseaddr(msg.get('From', 'unknown'))[1]
                    b = clean_text(msg.get_payload(decode=True).decode('latin-1')) if not msg.is_multipart() else ""
                    if b: data.append({'sender': s, 'body': b, 'urls': extract_urls(b), 'label': 1})
                except: continue

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['body']).dropna()
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_pickle('data/processed/email_data_large.pkl')
    print(f"\n✅ FINAL DATASET SIZE: {len(df)}")
    print(f"Distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    run()
