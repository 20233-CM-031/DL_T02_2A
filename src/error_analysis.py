"""
Comprehensive Error Analysis and Taxonomy Attribution Script.
Team 2 - Task 2A: Sentiment Analysis of Code Mixed Telugu and English Reviews.
Evaluation and Report Lead: 23261A6624 | Data Lead: 23261A6633
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from models import SentimentRNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

LABEL_MAP = {"Negative": 0, "Mixed": 1, "Positive": 2}
REV_LABEL_MAP = {0: "Negative", 1: "Mixed", 2: "Positive"}

def run_error_analysis():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    with open(os.path.join(DATA_DIR, "vocab.json"), "r") as f:
        vocab_data = json.load(f)
        word2idx = vocab_data["word2idx"]
        kappa = vocab_data.get("kappa", 0.9437)
        
    vocab_size = len(word2idx)
    
    # Load Best Model: Bi-GRU (Scratch)
    best_model = SentimentRNN(
        vocab_size=vocab_size, embed_dim=64, hidden_dim=64,
        num_classes=3, cell_type='gru', bidirectional=True, dropout=0.3
    )
    best_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "gru_scratch_bidi.pt")))
    best_model.eval()
    
    # Load Simple RNN for comparison
    rnn_model = SentimentRNN(
        vocab_size=vocab_size, embed_dim=64, hidden_dim=64,
        num_classes=3, cell_type='rnn', bidirectional=True, dropout=0.3
    )
    rnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "rnn_scratch_bidi.pt")))
    rnn_model.eval()
    
    # Predict on test set
    def get_predictions(model, texts, max_seq_len=60):
        preds = []
        with torch.no_grad():
            for t in texts:
                words = t.split()
                indices = [word2idx.get(w, word2idx.get("<UNK>", 1)) for w in words]
                if len(indices) < max_seq_len:
                    indices = indices + [0] * (max_seq_len - len(indices))
                else:
                    indices = indices[:max_seq_len]
                x = torch.tensor([indices], dtype=torch.long)
                logits = model(x)
                p = torch.argmax(logits, dim=1).item()
                preds.append(REV_LABEL_MAP[p])
        return preds
        
    test_df['pred_gru'] = get_predictions(best_model, test_df['cleaned_text'])
    test_df['pred_rnn'] = get_predictions(rnn_model, test_df['cleaned_text'])
    
    # Check errors
    test_df['is_gru_error'] = test_df['pred_gru'] != test_df['resolved_sentiment']
    test_df['is_rnn_error'] = test_df['pred_rnn'] != test_df['resolved_sentiment']
    test_df['annotator_disagreement'] = test_df['annotator_1'] != test_df['annotator_2']
    
    # Categorize test instances into error types:
    # 1. Sarcasm / Irony
    # 2. Spelling variation / phonetic transliteration
    # 3. Mixed opinion / polite opening + trailing complaint
    # 4. Slang & colloquial code-switching boundaries
    
    def assign_category(row):
        text = str(row['cleaned_text']).lower()
        if "super delivery timing" in text or "fantastic diet plan" in text or "what a customer support" in text or "peaks asalu" in text:
            return "Sarcasm & Pragmatic Irony"
        elif any(var in text for var in ["baagundi", "bavundi", "baavundhi", "bagundhi", "chaala", "ledhu asalu", "ayyindi", "aindi"]):
            return "Spelling & Transliteration Variations"
        elif row['has_polite_opening'] == 1 or ("but" in text and ("good" in text or "bagundi" in text or "super" in text or "fresh" in text)):
            return "Mixed Opinion (Polite Start + Trailing Grievance)"
        elif any(slang in text for slang in ["dabbulu bokka", "chetta", "daridram", "waste", "fraud"]):
            return "Colloquial Slang & Code-Switching Boundary"
        else:
            return "Standard Code-Mixed Declarative"
            
    test_df['error_category'] = test_df.apply(assign_category, axis=1)
    
    cat_summary = []
    categories = [
        "Sarcasm & Pragmatic Irony",
        "Spelling & Transliteration Variations",
        "Mixed Opinion (Polite Start + Trailing Grievance)",
        "Colloquial Slang & Code-Switching Boundary",
        "Standard Code-Mixed Declarative"
    ]
    
    for cat in categories:
        sub = test_df[test_df['error_category'] == cat]
        total_cat = len(sub)
        gru_errs = sub['is_gru_error'].sum()
        rnn_errs = sub['is_rnn_error'].sum()
        ann_dis = sub['annotator_disagreement'].sum()
        
        cat_summary.append({
            "Error Category": cat,
            "Total Test Samples": total_cat,
            "Simple RNN Errors": int(rnn_errs),
            "Best Model (Bi-GRU) Errors": int(gru_errs),
            "Annotator Disagreements (Label Ambiguity)": int(ann_dis),
            "Label Ambiguity Rate (%)": round((ann_dis / total_cat * 100), 2) if total_cat > 0 else 0.0,
            "Bi-GRU Accuracy in Category (%)": round(((total_cat - gru_errs) / total_cat * 100), 2) if total_cat > 0 else 0.0
        })
        
    df_cat = pd.DataFrame(cat_summary)
    df_cat.to_csv(os.path.join(DATA_DIR, "error_analysis_breakdown.csv"), index=False)
    
    print("\n" + "="*80)
    print("TAXONOMY OF ERRORS AND LABEL AMBIGUITY BREAKDOWN:")
    print("="*80)
    print(df_cat.to_string(index=False))
    
    # Confusion Matrix for Best Model (Bi-GRU)
    labels = ["Negative", "Mixed", "Positive"]
    cm = confusion_matrix(test_df['resolved_sentiment'], test_df['pred_gru'], labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix: Best Model (Bi-GRU Scratch) on Test Split")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("True Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_best_model.png"), dpi=300)
    plt.close()
    
    # Error comparison chart
    plt.figure(figsize=(9, 4.5))
    df_plot = df_cat[df_cat["Error Category"] != "Standard Code-Mixed Declarative"]
    bar_width = 0.35
    x_indices = np.arange(len(df_plot))
    
    plt.bar(x_indices - bar_width/2, df_plot["Simple RNN Errors"], width=bar_width, label="Simple RNN Errors", color="#d9534f")
    plt.bar(x_indices + bar_width/2, df_plot["Annotator Disagreements (Label Ambiguity)"], width=bar_width, label="Annotator Disagreement (Label Ambiguity)", color="#f0ad4e")
    
    plt.xticks(x_indices, [
        "Sarcasm", "Spelling Variations", "Mixed Opinion", "Colloquial Slang"
    ], rotation=15, ha='right')
    plt.ylabel("Error Count / Disagreement Count")
    plt.title(f"Error Distribution vs Label Ambiguity (Cohen's κ = {kappa:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_categories_breakdown.png"), dpi=300)
    plt.close()
    
    print(f"\n[+] Error analysis completed and saved to {FIG_DIR}/confusion_matrix_best_model.png and error_categories_breakdown.png")

if __name__ == "__main__":
    run_error_analysis()

