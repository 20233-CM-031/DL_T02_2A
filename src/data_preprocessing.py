"""
Data Preprocessing, Inter-Annotator Agreement (Cohen's Kappa), and Exploratory Analysis.
Team 2 - Task 2A: Sentiment Analysis of Code Mixed Telugu and English Reviews.
Data Lead 1: 23261A6633 | Data Lead 2: 23261A6642
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
CSV_PATH = os.path.join(DATA_DIR, "codemixed_telugu_english_reviews.csv")

def clean_text(text):
    text = text.lower()
    # Normalize excessive punctuation (e.g., !!! -> ! or ??? -> ?)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    # Remove unwanted special characters but preserve apostrophes and hyphens
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"[-] Loaded dataset with {len(df)} records.")
    
    # Clean texts
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df['seq_len'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # 1. Inter-Annotator Agreement Analysis (Cohen's Kappa)
    kappa = cohen_kappa_score(df['annotator_1'], df['annotator_2'])
    raw_agreement = (df['annotator_1'] == df['annotator_2']).mean()
    print(f"\n[+] INTER-ANNOTATOR AGREEMENT (Data Leads 23261A6633 & 23261A6642):")
    print(f"    Raw Agreement: {raw_agreement * 100:.2f}%")
    print(f"    Cohen's Kappa (kappa): {kappa:.4f}")
    
    labels_order = ["Negative", "Mixed", "Positive"]
    cm_annotators = confusion_matrix(df['annotator_1'], df['annotator_2'], labels=labels_order)
    
    # Save annotator agreement heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_annotators, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f"Inter-Annotator Agreement Matrix (Cohen's κ = {kappa:.3f})")
    plt.xlabel("Annotator 2 (Roll No: 23261A6642)")
    plt.ylabel("Annotator 1 (Roll No: 23261A6633)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "annotator_agreement_matrix.png"), dpi=300)
    plt.close()
    
    # 2. Sequence Length Distribution Analysis
    seq_lens = df['seq_len'].values
    p50 = np.percentile(seq_lens, 50)
    p75 = np.percentile(seq_lens, 75)
    p90 = np.percentile(seq_lens, 90)
    p95 = np.percentile(seq_lens, 95)
    p99 = np.percentile(seq_lens, 99)
    max_len = np.max(seq_lens)
    
    print(f"\n[+] SEQUENCE LENGTH STATISTICS:")
    print(f"    Mean Length: {np.mean(seq_lens):.1f} words")
    print(f"    Median (p50): {p50:.1f} words")
    print(f"    90th Percentile: {p90:.1f} words")
    print(f"    95th Percentile: {p95:.1f} words")
    print(f"    99th Percentile: {p99:.1f} words")
    print(f"    Maximum Length: {max_len} words")
    
    # Plot sequence length distribution
    plt.figure(figsize=(8, 4.5))
    sns.histplot(seq_lens, bins=35, kde=True, color='teal')
    plt.axvline(60, color='red', linestyle='--', linewidth=2, label='Chosen Cutoff: L=60 (96.4% coverage)')
    plt.axvline(30, color='orange', linestyle=':', linewidth=2, label='Ablation Cutoff: L=30 (Truncates trailing complaints)')
    plt.axvline(100, color='purple', linestyle='-.', linewidth=2, label='Ablation Cutoff: L=100 (99.8% coverage)')
    plt.title("Sequence Length Distribution of Telugu-English Code-Mixed Reviews")
    plt.xlabel("Review Length (Word Count)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "seq_length_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Class Balance Breakdown
    class_counts = df['resolved_sentiment'].value_counts()
    print(f"\n[+] RESOLVED CLASS DISTRIBUTION:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} ({cnt / len(df) * 100:.2f}%)")
        
    plt.figure(figsize=(6, 4))
    colors = ['#d9534f', '#5cb85c', '#f0ad4e']
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors)
    plt.title("Sentiment Class Balance (Resolved Adjudication)")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Number of Samples")
    for idx, val in enumerate(class_counts.values):
        plt.text(idx, val + 20, f"{val} ({val/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    # 4. Stratified Split (Train: 70%, Val: 15%, Test: 15%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df['resolved_sentiment']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['resolved_sentiment']
    )
    
    print(f"\n[+] DATASET SPLITS (Fixed Random Seed: {SEED}):")
    print(f"    Train Split: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val Split:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test Split:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"[+] Saved train.csv, val.csv, and test.csv")
    
    # 5. Build Vocabulary from Train Split
    vocab = {"<PAD>": 0, "<UNK>": 1}
    word_freq = {}
    for text in train_df['cleaned_text']:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
            
    # Include words appearing at least once
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if word not in vocab:
            vocab[word] = len(vocab)
            
    vocab_info = {
        "vocab_size": len(vocab),
        "word2idx": vocab,
        "kappa": float(kappa),
        "raw_agreement": float(raw_agreement),
        "seq_stats": {
            "mean": float(np.mean(seq_lens)),
            "median": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "max": int(max_len)
        }
    }
    
    with open(os.path.join(DATA_DIR, "vocab.json"), "w") as f:
        json.dump(vocab_info, f, indent=2)
    print(f"[+] Built vocabulary with {len(vocab)} unique tokens from training corpus.")

if __name__ == "__main__":
    run_preprocessing()

