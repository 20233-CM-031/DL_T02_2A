# Dataset Card: Code-Mixed Telugu-English Food Delivery Reviews (CM-TE-FoodRev)

## 1. Dataset Description
- **Curated By:** Team 2 (Roll Nos: 23261A6633, 23265A6606, 23261A6624, 23261A6642)
- **Task Variant:** 2A (Food Delivery App Reviews)
- **Course:** CM701PC Deep Learning & CM751PC Deep Learning Lab, MGIT (Autonomous), Hyderabad
- **Language:** Code-Mixed Telugu and English (Tenglish written in Latin Script)
- **Domain:** Mobile Food & Grocery Delivery Customer Reviews (Swiggy, Zomato, Blinkit, Zepto)
- **Dataset Size:** 3,640 curated and double-annotated reviews; 1,600 raw Google Play Store live reviews.

---

## 2. Dataset Structure

### 2.1 Schema of `codemixed_telugu_english_reviews.csv`
| Column Name | Data Type | Description |
|:---|:---|:---|
| `id` | Integer | Unique identifier for each review entry |
| `review_text` | String | Raw text containing code-mixed Telugu-English review |
| `app_name` | String | Source mobile application (`Swiggy`, `Zomato`, `Zepto`, `Blinkit`) |
| `annotator_1` | String | Sentiment label assigned by Data Lead 1 (`23261A6633`) |
| `annotator_2` | String | Sentiment label assigned by Data Lead 2 (`23261A6642`) |
| `resolved_sentiment` | String | Adjudicated ground truth sentiment (`Positive`, `Negative`, `Mixed`) |
| `has_polite_opening` | Integer (0/1) | Flag indicating whether the text starts with polite/praise tokens |
| `length_words` | Integer | Token count in whitespace-delimited words |
| `review_date` | String | Timestamp of collection/review |

### 2.2 Class Distribution
| Sentiment Class | Sample Count | Percentage (%) | Role & Challenge |
|:---|:---:|:---:|:---|
| **Negative** | 1,540 | 42.31% | Critical service complaints, delays, cold food, monetary loss |
| **Positive** | 1,400 | 38.46% | High satisfaction, fast delivery, authentic taste |
| **Mixed** | 700 | 19.23% | Dual valence (e.g. polite start + ending complaint); target metric $\ge 0.45$ F1 |
| **Total** | **3,640** | **100.0%** | Comprehensive benchmark for Dravidian code-mixing |

---

## 3. Linguistic Characteristics
1. **Script:** Non-standard Latin transliteration of Dravidian Telugu phonology.
2. **Code-Mixing Style:** Intra-sentential and inter-sentential alternation between English domain terminology (*delivery, rider, packaging, tracking, order, refund*) and Telugu grammatical roots, case markers, and auxillary verbs (*bagundi, raledu, chesaru, icharu, aindi, kani, chala*).
3. **Discourse Patterns:** High frequency of pleasantries followed by severe trailing complaints (*"Food chala bagundi but delivery late ayindi"*).
4. **Phonetic Variance:** Multiple transliterated surface forms for identical semantic roots (e.g., *bagundi* vs *baagundhi* vs *bavundi*).

---

## 4. Split Strategy
To ensure strict zero data leakage and reproducibility across all experiments:
- **Random Seed:** Fixed at `42`
- **Stratified Partition:**
  - **Train Set (70%):** 2,548 samples
  - **Validation Set (15%):** 546 samples (used for early stopping & hyperparameter tuning)
  - **Test Set (15%):** 546 samples (held-out evaluate split for final reporting)

---

## 5. Limitations & Ethical Considerations
- **Annotation Subjectivity:** Borderline mixed reviews where a customer mentions minor packaging flaw amidst overwhelming praise may lead to subtle annotator variance.
- **Dialectal Variation:** Reviews predominantly reflect Deccan (Hyderabad/Telangana) and Coastal Andhra vernacular food delivery phrasing.
- **PII Scrubbed:** User names, phone numbers, and delivery addresses have been sanitized or omitted to safeguard customer privacy.

