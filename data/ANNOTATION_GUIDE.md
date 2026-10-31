# Annotation Guide for Code-Mixed Telugu-English Food Delivery App Reviews

**Team 2 | Task 2A: Food Delivery App Reviews**  
**Course:** CM701PC Deep Learning & CM751PC Deep Learning Lab  
**Institution:** Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad  
**Department:** Emerging Technologies  
**Data Leads:** 23261A6633 & 23261A6642  
**Regulation:** MR-22 (B. Tech CSE - AIML, VII Semester)

---

## 1. Overview and Objective
This annotation guide establishes a rigorous, standardized protocol for manual sentiment labeling of code-mixed Telugu-English ("Tenglish") customer reviews collected from Indian food delivery platforms (Swiggy, Zomato, Zepto, Blinkit).

Because customers frequently employ colloquial Telugu typed in the Latin script alongside standard English, automated English-only sentiment analyzers fail significantly. This guide defines decision boundaries for three target sentiment classes:
- **Positive**
- **Negative**
- **Mixed** (crucial class reflecting dual valence, polite openings, and trailing complaints)

---

## 2. Sentiment Class Definitions

### 2.1 Positive
A review is labeled **Positive** if the customer expresses overall satisfaction, appreciation, or contentment with the food quality, taste, delivery speed, packaging, or customer service, without substantial contradictory grievances.

*Examples:*
- *"Food delivery chala fast ga vachindi, food kuda hot and fresh ga undi."* (Delivery was very fast, food was also hot and fresh.)
- *"Taste chala bagundi and delivery boy very polite ga matladaru."* (Taste was great and delivery boy spoke very politely.)
- *"Great app and fast delivery, discounts kuda bagunnayi."* (Great app and fast delivery, discounts are also good.)

### 2.2 Negative
A review is labeled **Negative** if the customer expresses frustration, disappointment, anger, or reports service failure, food deterioration, severe delays, or monetary losses.

*Examples:*
- *"Worst food delivery service, order pettina 2 hours tarvatha deliver chesaru."* (Worst service, delivered 2 hours after ordering.)
- *"Food totally cold ga vachindi, box motham damaged and spilled outside."* (Food arrived completely cold, box entirely spilled.)
- *"Item missing undi parcel lo, customer care refund ivvaledu, dabbulu bokka."* (Item missing, no refund, complete waste of money.)

### 2.3 Mixed (Ambivalent / Shifted Valence)
A review is labeled **Mixed** if it contains **both positive and negative sentiment elements** within the same text. 
**Special Focus Case: Polite Opening + Trailing Complaint**:
Customers in the Deccan/Telugu region frequently begin reviews with a polite or complimentary observation about food taste or app design, but conclude by voicing their primary complaint (e.g., severe delay, packaging leakage, rude delivery partner).

*Key Rule:* If the review acknowledges satisfactory performance in one dimension (e.g., food taste) while complaining about another dimension (e.g., delivery delay or platform fee), it **must** be categorized as **Mixed**, regardless of syntactic length.

*Examples:*
- *"Food chala bagundi but delivery late ayindi."* (Food was very good but delivery was late.)
- *"Restaurant food quality is really good but packaging covers leak aipoyi mess aindi."* (Quality good, but leakage caused a mess.)
- *"Delivery executive was very polite and soft spoken but food totally cold ga deliver ayindi."* (Polite executive, but cold food.)
- *"Biryani aroma and taste super undi but quantity mathram single person ki kuda saripodu."* (Great taste, but insufficient quantity.)

---

## 3. Linguistic & Domain Challenges

### 3.1 Sarcasm & Irony
Sarcastic reviews frequently use superficially positive words ("super", "great", "peaks", "wah") to ridicule poor service.
- *Handling:* Sarcasm directed at service failures must be annotated according to the **underlying pragmatic intent** (**Negative**).
- *Example:* *"Chala super delivery timing... lunch time order chesthe dinner time ki techaru, great app!"* $\rightarrow$ **Negative**.

### 3.2 Transliteration and Phonetic Spelling Invariance
Telugu words lack a standardized Latin orthography. Annotators must recognize equivalent phonetic spelling variants:
- *bagundi* $\equiv$ *baagundi* $\equiv$ *bavundi* $\equiv$ *baavundhi* $\equiv$ *bagundhi*
- *chala* $\equiv$ *chaala* $\equiv$ *chalaaaa*
- *ledu* $\equiv$ *ledhu* $\equiv$ *ledhu asalu*
- *ayindi* $\equiv$ *aindi* $\equiv$ *ayyindi* $\equiv$ *ayindhi*

### 3.3 Discourse Markers
Pay special attention to Telugu and English contrastive conjunctions that signal sentiment transitions:
- English: *but*, *however*, *although*, *yet*
- Telugu: *kani*, *kaani*, *ayina*, *matram*, *ayithe*

---

## 4. Annotation Workflow & Agreement
1. **Independent Double Annotation:** Every review is independently annotated by Annotator 1 (Data Lead 1: `23261A6633`) and Annotator 2 (Data Lead 2: `23261A6642`).
2. **Agreement Calculation:** Inter-annotator agreement is evaluated using **Cohen's Kappa ($\kappa$)**.
3. **Dispute Resolution:** In case of disagreement ($\approx 6-8\%$ of samples, primarily between Mixed vs Negative due to trailing complaint severity), the team convenes and resolves the label by mutual consensus adhering to Section 2.3.

