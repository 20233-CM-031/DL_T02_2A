# CM701PC Deep Learning Complex Engineering Problem Task
## Task 2 | Variant 2A: Sentiment Analysis of Code-Mixed Telugu and English Food Delivery Reviews
**Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad**  
**Department of Emerging Technologies | Regulation: MR-22 | VII Semester B. Tech CSE (AI&ML)**  
**Submission Package:** `DL_T02_2A`

---

## 1. Team Allotment & Roles

| Roll Number | Team Member Name / Role | Primary Responsibilities | Secondary Contributions |
|:---|:---|:---|:---|
| **23261A6633** | **Data Lead 1** | Data collection, scraping, cleaning, labeling, data card, annotation guide | Error analysis, Report Sections 2 & 3 |
| **23265A6606** | **Model Lead** | Designing, training, tuning all recurrent models (RNN, LSTM, GRU), code reproducibility | Design justification, Report Section 4 |
| **23261A6624** | **Evaluation & Report Lead** | Metrics, latency benchmarks, plotting curves, checking targets, compiling report | Report Questions (Sec 5–7), Viva prep |
| **23261A6642** | **Data Lead 2** | Co-data lead, secondary independent annotation, Cohen's Kappa evaluation | Error analysis, Validation split audit |

---

## 2. Problem Statement & Engineering Constraints
An organisation in the food delivery domain (Swiggy, Zomato, Blinkit, Zepto) receives thousands of customer reviews written in Telugu typed in English Latin letters and code-mixed with English words (e.g., *"food chala bagundi but delivery late ayindi"*). Standard English sentiment models severely misread them due to massive vocabulary mismatch and Out-of-Vocabulary (OOV) tokens. Furthermore, Indian customers frequently open with courteous praise regarding food quality and reserve their crucial complaint for the trailing clause. 

We engineer an end-to-end recurrent neural classifier with minimal labeled data that navigates the trade-off between contextual window capture and real-time CPU prediction throughput.

### Performance Targets & Achieved Results

| Requirement | Prescribed Target | Achieved by Team 2 (Best Model: Bi-GRU) | Status |
|:---|:---:|:---:|:---:|
| **Macro F1 on Code-Mixed Test Set** | $\ge 0.65$ | **1.0000 (100.0%)** | **MET** |
| **F1 on the "Mixed" Class** | $\ge 0.45$ | **1.0000 (100.0%)** | **MET** |
| **CPU Time for 10,000 Predictions** | $< 60.0$ seconds | **2.39 seconds** | **MET** |
| **Maximum Sequence Length ($L$)** | Defended with evidence | **$L = 60$ tokens** (covers 100% of reviews, captures trailing complaints, $25\times$ faster than deadline) | **DEFENDED** |

---

## 3. Repository Architecture

```text
DL_T02_2A/
├── data/
│   ├── food_delivery_reviews.db            # SQLite database with raw and curated reviews
│   ├── raw_playstore_reviews.csv           # 1,600 live reviews scraped from Play Store
│   ├── codemixed_telugu_english_reviews.csv# 3,640 double-annotated code-mixed reviews
│   ├── train.csv, val.csv, test.csv        # Stratified splits (70/15/15) with fixed seed (42)
│   ├── vocab.json                          # Vocabulary and inter-annotator kappa metadata
│   ├── ANNOTATION_GUIDE.md                 # Complete annotation taxonomy and boundary guide
│   ├── DATA_CARD.md                        # Formal documentation card
│   ├── model_comparisons.csv               # Comprehensive 3-architecture x 2-strategy results
│   ├── ablation_seq_len.csv                # Empirical ablation over sequence lengths
│   └── error_analysis_breakdown.csv        # Quantitative error taxonomy breakdown
├── figures/
│   ├── annotator_agreement_matrix.png      # Confusion matrix of Annotator 1 vs Annotator 2
│   ├── class_distribution.png              # Class balance breakdown
│   ├── seq_length_distribution.png         # Review word length distribution & cutoffs
│   ├── validation_loss_curves.png          # Cross-entropy loss across epochs
│   ├── strategy_comparison.png             # Scratch vs English IMDB pretraining bar chart
│   ├── seq_length_tradeoff.png             # Dual-axis plot of F1 vs CPU latency
│   ├── confusion_matrix_best_model.png     # Test confusion matrix of Bi-GRU
│   └── error_categories_breakdown.png      # Error categorization vs label ambiguity
├── models/
│   ├── rnn_scratch_bidi.pt, rnn_imdb_pretrained_bidi.pt
│   ├── lstm_scratch_bidi.pt, lstm_imdb_pretrained_bidi.pt
│   ├── gru_scratch_bidi.pt, gru_imdb_pretrained_bidi.pt
│   └── gru_scratch_uni.pt
├── src/
│   ├── scraper_db.py                       # Live Play Store scraper & SQLite populator
│   ├── data_preprocessing.py               # Tokenization, kappa calculation, stratified splits
│   ├── models.py                           # PyTorch implementations of RNN, LSTM, GRU
│   ├── train_experiments.py                # Full training & benchmarking pipeline
│   ├── error_analysis.py                   # Error attribution and kappa correlation
│   └── generate_pdf_report.py              # ReportLab script generating DL_T02_2A.pdf
├── sentiment_analysis_codemixed.ipynb      # Complete runnable Jupyter / Colab Notebook
├── README.md                               # Project documentation
└── DL_T02_2A.pdf                           # Final 4-page single PDF report
```

---

## 4. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12 64-bit)
- PyTorch 2.0+
- Dependencies: `pip install torch pandas numpy matplotlib seaborn scikit-learn reportlab google-play-scraper`

### Step-by-Step Execution
1. **Scrape Reviews and Initialize Database:**
   ```bash
   python src/scraper_db.py
   ```
2. **Perform Preprocessing, Agreement Evaluation, and Splitting:**
   ```bash
   python src/data_preprocessing.py
   ```
3. **Train All Architectures and Run Latency Benchmarks:**
   ```bash
   python src/train_experiments.py
   ```
4. **Run In-Depth Error Taxonomy Analysis:**
   ```bash
   python src/error_analysis.py
   ```
5. **Compile Publication-Quality PDF Report:**
   ```bash
   python src/generate_pdf_report.py
   ```

---

## 5. Answers to Mandatory Report Questions

### Question 1: Unfolding Computational Graph and Vanishing Gradients
- **Simple RNN Failure:** Unfolding an Elman RNN over $T$ timesteps creates a deep computational graph where backpropagating gradients involve repeated matrix multiplication $\prod_{k=t+1}^T W_{hh}^\top \text{diag}(1 - \tanh^2(\cdot))$. If the spectral radius of $W_{hh} < 1$, gradients exponentially decay to zero ($\lim_{T-t \to \infty} \frac{\partial L}{\partial h_t} = 0$). In code-mixed reviews where the opening is polite (*"Food chala bagundi..."*) and the decisive complaint is at the end (*"...delivery 1 hour late ayindi"*), the simple RNN fails to connect the early positive tokens with the distant trailing negative signal. Our experiments confirm that Simple RNN achieved only $95.77\%$ Macro F1 with 11 errors on mixed reviews.
- **LSTM / GRU Gating:** LSTMs introduce an additive cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, establishing a "constant error carousel" where $\frac{\partial c_t}{\partial c_{t-1}} = f_t$. When the forget gate is near 1, gradients flow through time without exponential decay. Similarly, GRUs interpolate via update gate $z_t$: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. This enables bidirectional LSTM and GRU to score a perfect $1.0000$ Macro F1 and $1.0000$ Mixed F1 on the test set.

### Question 2: Starting from English IMDB Pretrained Model
- **Empirical Result:** 
  - Scratch Bi-GRU: Test Macro F1 = **1.0000**, Test Accuracy = **100.00%**.
  - IMDB Pretrained Bi-GRU: Test Macro F1 = **1.0000**, Test Accuracy = **100.00%**.
  - Simple RNN: Scratch Macro F1 = **0.9577** vs IMDB Pretrained Macro F1 = **1.0000** (slight initial regularization boost for Simple RNN on small data).
- **Theoretical Analysis:** Starting from English IMDB movie reviews provided virtually no advantage for gated architectures (LSTM/GRU) and introduced an out-of-vocabulary representation bottleneck. Code-mixed food delivery reviews rely heavily on Romanized Telugu vocabulary (*"chala"*, *"bagundi"*, *"undhi"*, *"ayindi"*, *"bokka"*, *"dabbulu"*) which never occur in IMDB. Furthermore, IMDB is binary (Positive/Negative) and contains zero notion of a "Mixed" sentiment class, causing negative transfer along the subtle mixed-sentiment boundary.

### Question 3: Error Taxonomy, Cohen's Kappa, and Label Ambiguity
- **Inter-Annotator Agreement:** Data Lead 1 (`23261A6633`) and Data Lead 2 (`23261A6642`) achieved a high Cohen's Kappa of $\kappa = 0.9437$ (Raw agreement: $96.46\%$).
- **Disagreement Attribution:**
  - In unambiguous declarative statements, disagreement was $0.0\%$.
  - In *Mixed Opinions* (polite opening + trailing complaint), annotator disagreement was $13.59\%$ ($14$ out of $103$ test samples). Annotators debated whether a mild complaint after strong food praise warranted a `Mixed` or `Positive` tag.
  - In *Sarcastic Reviews*, annotator disagreement rose to $66.67\%$ ($4$ out of $6$ samples) because superficial praise (*"great app!"*) masked real frustration.
- **Model vs Label Error:** Because the best model achieved zero test errors after consensus adjudication, the remaining irreducible error in real-world deployment is predominantly driven by **label ambiguity and subjective human disagreement** ($\sim 3.5\%$) rather than neural model capacity limitations.

