import os
import mailbox
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from email.utils import parseaddr

def extract_urls(text):
    if not text: return []
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text.lower())

def parse_enron(base_path, limit=5000):
    emails = []
    print("Parsing Enron...")
    count = 0
    for root, _, files in os.walk(base_path):
        for file in files:
            if count >= limit: break
            try:
                with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                    content = f.read()
                    emails.append({'body': content, 'label': 0})
                    count += 1
            except: continue
        if count >= limit: break
    return emails

def parse_nazario(mbox_paths):
    emails = []
    print("Parsing Nazario...")
    for path in mbox_paths:
        if not os.path.exists(path): continue
        mbox = mailbox.mbox(path)
        for message in tqdm(mbox):
            try:
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode('latin-1')
                            break
                else:
                    body = message.get_payload(decode=True).decode('latin-1')
                emails.append({'body': body, 'label': 1})
            except: continue
    return emails

def run_analysis():
    # Load data
    enron_emails = parse_enron('data/raw/maildir', limit=5000)
    nazario_emails = parse_nazario([f'data/raw/nazario_phishing_{i}.mbox' for i in range(4)])
    
    df = pd.DataFrame(enron_emails + nazario_emails)
    
    # Cleaning: Remove duplicates & empty bodies
    df = df.dropna(subset=['body'])
    df = df.drop_duplicates(subset=['body'])
    
    # Feature extraction for analysis
    df['url_count'] = df['body'].apply(lambda x: len(extract_urls(x)))
    df['body_len'] = df['body'].apply(len)
    
    # Save statistics to PDF
    with PdfPages('figures/data_exploration.pdf') as pdf:
        # 1. Class Distribution
        plt.figure(figsize=(8, 6))
        df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Class Distribution (0: Benign, 1: Phishing)')
        plt.xticks([0, 1], ['Benign', 'Phishing'], rotation=0)
        pdf.savefig()
        plt.close()
        
        # 2. URL Count Distribution
        plt.figure(figsize=(8, 6))
        df.groupby('label')['url_count'].mean().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Average URL Count per Email')
        plt.xticks([0, 1], ['Benign', 'Phishing'], rotation=0)
        pdf.savefig()
        plt.close()

        # 3. Body Length (Log scale)
        plt.figure(figsize=(8, 6))
        df.boxplot(column='body_len', by='label')
        plt.yscale('log')
        plt.title('Email Body Length Distribution')
        pdf.savefig()
        plt.close()

    # Lưu dữ liệu đã làm sạch sơ bộ để bước sau tạo đồ thị
    df.to_pickle('data/processed/cleaned_emails.pkl')
    print(f"✅ Analysis done. Saved PDF to figures/data_exploration.pdf")
    print(f"Total cleaned emails: {len(df)}")

if __name__ == "__main__":
    run_analysis()
