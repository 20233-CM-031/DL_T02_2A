Telugu-English Food Delivery Review Sentiment Analysis
Team 2 · Task 2 · Variant 2A
Course: CM701PC Deep Learning and CM751PC Deep Learning Lab
Institution: Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad
About the Project
This project classifies food-delivery app reviews written in a mix of Telugu (typed using English letters) and English. Each review is classified into one of three sentiment categories:
- Positive
- Negative
- Mixed
The project compares Simple RNN, LSTM and GRU models to understand which approach works best for code-mixed customer reviews.
Dataset
The reviews are from food-delivery platforms represented in the project collection: Swiggy, Zomato, Blinkit and Zepto.
- 1,600 raw customer reviews were collected from Google Play Store listings.
- The curated code-mixed dataset contains 3,640 reviews.
- The curated data was split into training (70%), validation (15%) and testing (15%) sets using random seed 42.
- The class labels are Positive, Negative and Mixed.
- Two team members independently annotated the reviews. The reported agreement was 96.46%, with Cohen's kappa 0.9437.
The raw collection count and curated dataset count refer to different stages of the project. See DATA_CARD.md for dataset details and ANNOTATION_GUIDE.md for the labelling rules.
Models and Experiments
The team compared:
- Simple RNN, LSTM and GRU.
- Unidirectional and bidirectional models.
- Training from scratch and initialisation using an English IMDB sentiment model.
- Sequence lengths of 15, 30, 60 and 100 tokens.
The selected model was a bidirectional GRU (Bi-GRU) with a sequence length of 60.
Reported Results
Measure	Reported result	Task target
Test Macro F1 (Bi-GRU)	1.0000	At least 0.65
Mixed-class F1 (Bi-GRU)	1.0000	At least 0.45
CPU time for 10,000 predictions	2.39 seconds	Under 60 seconds
Review coverage at sequence length 60	100%	Evidence-based selection


The Bi-LSTM also achieved 1.0000 Macro F1 and Mixed-class F1, with a reported CPU time of 1.08 seconds for 10,000 predictions. The team selected Bi-GRU as its final model; Bi-LSTM was faster in the reported benchmark.

These are the results reported by the project experiments on its test split. They should not be interpreted as a guarantee of perfect performance on reviews from other platforms, regions or time periods.
Project Files
The submission package includes the following main files and resources:
File	Description
sentiment_analysis_codemixed.ipynb	Jupyter Notebook with the implementation, explanations and visualisations
food_delivery_reviews.db	SQLite database containing the project's review records
train.csv	Training data split
val.csv	Validation data split
test.csv	Test data split
DATA_CARD.md	Dataset description and details
ANNOTATION_GUIDE.md	Sentiment labelling guidelines
DL_T02_2A.pdf	Final engineering report
DL_T02_2A.zip	Packaged project archive


The project archive also contains the source code, trained models, figures and supporting documentation.
Getting Started
1. Clone or download this repository.
2. Make sure Python and Jupyter Notebook are installed.
3. Open sentiment_analysis_codemixed.ipynb in Jupyter Notebook or JupyterLab.
4. Keep the notebook, database and dataset files in the locations expected by the notebook.
5. Install the Python packages required by the notebook, then run the cells in order.
Refer to the notebook and project documentation for the implementation and experiment details. The exact package versions and execution environment should be taken from the project files supplied with the repository.
Team Contributions
Roll number	Role	Main contribution
23261A6633	Data Lead 1	Review collection, SQLite storage, dataset curation, Data Card and Annotation Guide
23265A6606	Model Lead & Report Lead	RNN, LSTM and GRU implementation, IMDB transfer learning, model and sequence-length experiments, architecture documentation
23261A6624	Evaluation & Report Lead	Metrics, CPU benchmarking, plots, target verification, error analysis and report compilation
23261A6642	Data Lead 2	Independent annotation, Cohen's kappa, stratified-split support and validation audit


All team members contributed to the overall project work and preparation.
Limitations and Future Work
The project completed its main data, modelling and evaluation tasks. Some extensions remain for future work:
- Better handling of Telugu spelling and transliteration variations using character-level or subword methods.
- Improved handling of sarcasm and unclear mixed opinions.
- Testing on more platforms, regions and Telugu dialects.
- Adding image analysis for complaints involving damaged food or packaging.
- Exploring contextual models such as IndicBERT or Gemma.
- Building a production application with monitoring and continuous user feedback.
AI Assistance
Google Antigravity with Gemini 3.8 Flash was used as a coding assistant for selected boilerplate code, initial PyTorch DataLoader scaffolding and report-layout support. The team reviewed and integrated the generated material and carried out the project experiments and evaluation. Further details are provided in the final report.
Academic Context
Prepared for CM701PC Deep Learning and CM751PC Deep Learning Lab, Team 2, Variant 2A, Mahatma Gandhi Institute of Technology (Autonomous), Hyderabad.
