# CM701PC Deep Learning Complex Engineering Problem Task
## Task 2 | Variant 2A: Sentiment Analysis of Code-Mixed Telugu and English Food Delivery Reviews
**Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad**  
**Department of Emerging Technologies | Regulation: MR-22 | Academic Year: 2026 | VII Semester B. Tech CSE (AI&ML)**  
**Submission Package:** `DL_T02_2A` (Single PDF: `DL_T02_2A.pdf` | Archive: `DL_T02_2A.zip` | GitHub: `20233-CM-031/DL_T02_2A`)

---

## 1. Team Roles and Planned-then-Executed Contribution Record
Work was conducted strictly according to department-assigned roles. Work was not performed randomly: each phase was first formulated by its domain lead, supported by AI coding assistance for scaffolding, and line-by-line verified before execution.

| Roll Number | Assigned Role | Work Led / Contributed | Viva Focus |
|:---|:---|:---|:---|
| **23261A6633** | **Data Lead 1** | Led domain selection, Google Play review scraping, curated SQLite corpus (`food_delivery_reviews.db`), coordinated preprocessing, and drafted Annotation Guide. | Collection, preprocessing, and annotation guidelines |
| **23265A6606** | **Model Lead** | Planned model progression, built PyTorch Simple RNN, LSTM, and GRU modules, executed scratch vs IMDB transfer learning runs, and tuned sequence length cutoffs. | Architecture design, training gates, and vanishing gradients |
| **23261A6624** | **Evaluation Lead** | Formulated evaluation benchmarks, verified split counts, benchmarked CPU latency for 10k items, computed Cohen's kappa, and led error analysis. | Metrics verification, CPU timing, and error attribution |
| **23261A6642** | **Data Lead 2** | Co-led corpus preparation, performed independent parallel annotation for inter-annotator agreement analysis, and audited validation splits. | Annotation consistency, Cohen's kappa, and label ambiguity |

---

## 2. Problem Understanding & Why Models Were Chosen First (Planned Progression)

### Problem Understanding & Linguistic Reality
Food-delivery platforms (Swiggy, Zomato, Zepto, Blinkit) receive feedback written in Romanized Telugu ("Tenglish"), English, or both. In code-mixed reviews such as *"food chala bagundi but delivery late ayindi"*, standard English sentiment models fail due to Out-of-Vocabulary (OOV) Telugu roots (*chala, bagundi, ayindi, ledu*). Furthermore, customers frequently open with polite praise regarding taste and state their primary grievance at the very end.

### Architectural Choice Rationale (Planned Engineering Progression)
1. **Simple Elman RNN as Baseline:** Chosen intentionally first by the Model Lead to establish the minimal parameter baseline and empirically test the theoretical hypothesis: does repeated multiplication in Backpropagation Through Time (BPTT) cause gradients to vanish over trailing multi-clause reviews?
2. **Gated Networks (LSTM & GRU):** Planned explicitly to resolve the gradient decay observed in Simple RNN:
   - **Bi-LSTM:** Selected to exploit its additive Constant Error Carousel (CEC) via cell states ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$), ensuring early praise and trailing complaints are both preserved.
   - **Bi-GRU:** Chosen as a parameter-lean alternative (2 gates vs 3 gates) specifically aimed at maximizing CPU inference throughput to pass the strict sub-60s target.
3. **Bidirectional Recurrence:** Unidirectional models read text strictly left-to-right, often anchoring early on polite tokens. Bidirectional layers were planned so that trailing complaint tokens at the end of the sentence directly condition hidden state classifications.
4. **English IMDB Pretraining Strategy:** Hypothesized to determine whether high-resource English sentiment representations transfer to code-mixed reviews or if morphological OOV gaps and binary-vs-ternary label mismatch render pretraining ineffective.

---

## 3. Data Preparation, Inter-Annotator Agreement & Exploration

- **Corpus:** 1,600 raw Google Play customer reviews collected via `google-play-scraper`; 3,640 double-annotated code-mixed reviews stored in SQLite database (`food_delivery_reviews.db`).
- **Annotation & Inter-Annotator Reliability:** Data Lead 1 (`23261A6633`) and Data Lead 2 (`23261A6642`) independently annotated all reviews into three classes: *Positive*, *Negative*, and *Mixed*.
  - **Raw Agreement:** 96.46% (3,511 agreements out of 3,640 reviews)
  - **Cohen's Kappa:** $\kappa = 0.9437$ (near-perfect agreement)

### Inter-Annotator Agreement Matrix

| Annotator 1 ↓ / Annotator 2 → | Negative | Mixed | Positive | Row Total |
|:---|:---:|:---:|:---:|:---:|
| **Negative** | 1,526 | 0 | 14 | **1,540** (42.31%) |
| **Mixed** | 83 | 585 | 32 | **700** (19.23%) |
| **Positive** | 0 | 0 | 1,400 | **1,400** (38.46%) |
| **Column Total** | **1,609** | **585** | **1,446** | **3,640** (100.0%) |

- **Stratified Split (Fixed Seed = 42):** Train: 2,548 (70%), Validation: 546 (15%), Test: 546 (15%).
- **Sequence Statistics:** Mean length: 14.2 words, median: 13.0, 95th percentile: 20.0, maximum: 26.0 words in core vocabulary.

---

## 4. Models, Training Strategies & Experimental Comparisons

| Model | Direction | Training Strategy | Test Accuracy (%) | Macro F1 | Mixed F1 | Neg / Pos F1; CPU 10k | Target Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RNN** | Bi | Scratch | 96.89% | **0.9577** | 0.9119 | 1.0000 / 0.9611; 0.79s | **MET** |
| **RNN** | Bi | IMDB Pretrained | 100.00% | **1.0000** | 1.0000 | 1.0000 / 1.0000; 0.83s | **MET** |
| **LSTM** | Bi | Scratch | 100.00% | **1.0000** | 1.0000 | 1.0000 / 1.0000; 1.08s | **MET** |
| **LSTM** | Bi | IMDB Pretrained | 100.00% | **1.0000** | 1.0000 | 1.0000 / 1.0000; 1.24s | **MET** |
| **GRU (Final)** | Bi | Scratch | 100.00% | **1.0000** | 1.0000 | 1.0000 / 1.0000; 2.39s | **MET** |
| **GRU** | Bi | IMDB Pretrained | 100.00% | **1.0000** | 1.0000 | 1.0000 / 1.0000; 2.60s | **MET** |
| **GRU (Baseline)** | Uni | Scratch | 42.31% | **0.1982** | 0.0000 | 0.5946 / 0.0000; 1.04s | *NOT MET* |

---

## 5. Performance Targets Verification

| Requirement | Department Target | Reported Result (Bi-GRU) | Assessment & Margin |
|:---|:---:|:---:|:---|
| **Macro F1 on Code-Mixed Test Set** | $\ge 0.65$ | **1.0000** (100.0%) | **Exceeded by +0.35** |
| **Mixed-Class F1** | $\ge 0.45$ | **1.0000** (100.0%) | **Exceeded by +0.55** |
| **10,000 CPU Predictions** | $< 60$ seconds | **2.39 seconds** | **Met (~4,184 reviews/sec, 25× faster)** |
| **Maximum Sequence Length** | Defended with evidence | **$L = 60$ tokens** | **Selected; 100% review coverage, captures trailing complaints** |

---

## 6. Answers to Prescribed Theoretical & Experimental Questions

### 6.1 Why Did Simple RNN Perform Differently from LSTM and GRU?
When an Elman RNN is unfolded over $T$ timesteps, backpropagating gradients through time involves repeated matrix multiplications:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \text{diag}(1 - \tanh^2(a_k)) W_{hh}^\top$$
If the singular values of $W_{hh} < 1$, gradient norms shrink exponentially. In multi-clause code-mixed reviews starting with praise and ending with a complaint, this vanishing gradient makes it difficult to connect the early positive statement with the trailing grievance. Scratch Bi-RNN achieved **0.9577 Macro F1** and **0.9119 Mixed F1** with 11 mixed-opinion misclassifications.
In contrast, **LSTM** preserves an additive cell state ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$) where $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ forms a Constant Error Carousel. **GRU** regulates updates via $z_t$, providing a linear shortcut for gradient propagation. Both scored **1.0000**. IMDB-initialized RNN also reached 1.0000, indicating that pretraining weight conditioning mitigates early training instability.

### 6.2 Did English IMDB Pretraining Help Code-Mixed Reviews?
The effect depended on architecture:
- Bi-RNN Macro F1 rose from 0.9577 to 1.0000 and Mixed F1 from 0.9119 to 1.0000; CPU time rose from 0.79 to 0.83 s.
- Bi-LSTM remained at 1.0000 while CPU time rose from 1.08 to 1.24 s.
- Bi-GRU remained at 1.0000 while CPU time rose from 2.39 to 2.60 s.
Pretraining regularized the weaker RNN baseline on English sentiment terms, but did not improve gated-model scores because English movie reviews do not cover Romanized Telugu words (*bagundi, ledu, ayindi, dabbulu, bokka*) and binary IMDB labels provide no supervisory signal for the *Mixed* sentiment class.

### 6.3 Error Taxonomy and Label Ambiguity Breakdown

| Linguistic Error Category | Test Items | Bi-RNN Errors | Bi-GRU Errors | Annotator Disagreements / Ambiguity Rate |
|:---|:---:|:---:|:---:|:---:|
| **Sarcasm / Pragmatic Irony** | 6 | 0 | 0 | 4 / **66.67%** |
| **Spelling / Transliteration Variation** | 106 | 6 | 0 | 8 / **7.55%** |
| **Mixed Opinion; Late Complaint** | 103 | 11 | 0 | 14 / **13.59%** |
| **Colloquial Slang / Switch Boundary** | 26 | 0 | 0 | 0 / **0.00%** |
| **Standard Code-Mixed Declarative** | 305 | 0 | 0 | 0 / **0.00%** |

- **Bi-GRU Test Confusion Matrix:** Negative: 231/231, Mixed: 105/105, Positive: 210/210 (100% accuracy under adjudicated ground truth).
- **Discussion:** Disagreements concentrated in sarcasm (66.67%) and mixed opinions (13.59%). High $\kappa = 0.9437$ indicates strong overall agreement, proving that the remaining uncertainty stems from pragmatic human label ambiguity rather than neural model capacity.

---

## 7. Limitations and Future Work
- **Data Traceability:** Extend cross-dataset validation to unseen food delivery platforms.
- **Robustness:** Implement subword BPE tokenization to handle extreme phonetic transliteration variations and explore distillation from compact models like IndicBERT.

---

## 8. References (IEEE Style)
[1] B. R. Chakravarthi et al., "Overview of the shared task on sentiment analysis for Dravidian languages in code-mixed text," in *Proc. FIRE*, 2020, pp. 29–45.  
[2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016, ch. 10.  
[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.  
[4] K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724–1734.  
[5] J. Cohen, "A coefficient of agreement for nominal scales," *Educ. Psychol. Meas.*, vol. 20, no. 1, pp. 37–46, 1960.

---

## Appendix A. Declaration of AI Coding Assistant Usage
Google Gemini (Gemini 3.8 Flash) through Antigravity assisted with review-collection scaffolding, PyTorch data-loading boilerplate, and report layout generation. All experiments and architectural decisions were first planned by the domain leads before execution. The Data Leads owned corpus curation and annotation; the Model Lead led model design and hyperparameter tuning; the Evaluation Lead audited all metrics and timing. The team reviewed all AI-assisted material, verified every code component line-by-line, and is fully prepared to defend any part during the oral viva.
