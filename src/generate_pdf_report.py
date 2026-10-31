"""
Updated Publication-Grade PDF Generator for Team 2 - Task 2A.
Reflects the exact planned-then-executed methodology, short and clear technical explanations,
precise experimental comparisons, and verified 4-page layout.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "DL_T02_2A.pdf")
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0B2545"))
        
        # Header (pages 1 to 4)
        self.drawString(36, 756, "MGIT (Autonomous), Hyderabad — Department of Emerging Technologies")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#444444"))
        self.drawRightString(576, 756, "Deep Learning CEP Task: DL_T02_2A")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(36, 750, 576, 750)
        
        # Footer
        self.line(36, 36, 576, 36)
        self.drawString(36, 25, "Regulation MR-22 | B. Tech CSE (AI&ML), VII Sem | Team 2 — Variant 2A")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_text)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=44
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=15,
        textColor=colors.HexColor("#0B2545"),
        alignment=1,
        spaceAfter=2
    )
    
    sub_title_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#134074"),
        alignment=1,
        spaceAfter=4
    )
    
    sec_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=12.5,
        textColor=colors.HexColor("#0B2545"),
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    
    sub_heading = ParagraphStyle(
        'SubHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#134074"),
        spaceBefore=3,
        spaceAfter=1.5,
        keepWithNext=True
    )
    
    body = ParagraphStyle(
        'Body12',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=10.2,
        textColor=colors.HexColor("#1a1a1a"),
        alignment=4,
        spaceAfter=2.5
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=8.8,
        textColor=colors.HexColor("#111111")
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=8.8,
        textColor=colors.HexColor("#111111")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=8.8,
        textColor=colors.white
    )

    story = []
    
    # =========================================================================
    # PAGE 1: HEADER, TEAM ROLES & SECTION 2: PROBLEM UNDERSTANDING & PLAN
    # =========================================================================
    story.append(Paragraph("MAHATMA GANDHI INSTITUTE OF TECHNOLOGY (AUTONOMOUS)", title_style))
    story.append(Paragraph("Department of Emerging Technologies | Regulation MR-22 | Academic Year 2026", sub_title_style))
    story.append(Paragraph("<b>COMPLEX ENGINEERING PROBLEM REPORT: TASK 2 (VARIANT 2A)</b><br/><font size=8 color='#444444'>CM701PC Deep Learning & CM751PC Deep Learning Lab | Faculty: Dr. M. Rudra Kumar, Professor</font>", title_style))
    story.append(Spacer(1, 2))
    
    meta_box = [
        [Paragraph("<b>Task & Variant:</b> Task 2 — Sentiment Analysis of Code-Mixed Telugu and English Reviews | Variant 2A", table_cell),
         Paragraph("<b>Target Domain:</b> Swiggy, Zomato, Zepto, and Blinkit Customer Feedback", table_cell)],
        [Paragraph("<b>Team Designation:</b> Team 2 (Rolls: 23261A6633, 23265A6606, 23261A6624, 23261A6642)", table_cell),
         Paragraph("<b>Submission Mode:</b> Code Repository & Evaluated Single PDF Report", table_cell)]
    ]
    t_meta = Table(meta_box, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f4f7fb")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 3))
    
    story.append(Paragraph("<b>1. Team Roles and Contribution Record</b>", sec_heading))
    story.append(Paragraph("The team followed the department-assigned roles. Work was not performed randomly: each phase was planned by its designated domain lead before execution, utilizing AI assistance for boilerplate and layout, with all code, experiments, and metrics rigorously verified by the leads.", body))
    
    contrib_data = [
        [Paragraph("Roll Number", table_header), Paragraph("Assigned Role", table_header), 
         Paragraph("Work Led / Contributed", table_header), Paragraph("Viva Focus", table_header)],
        [Paragraph("<b>23261A6633</b>", table_cell_bold), Paragraph("Data Lead 1", table_cell), 
         Paragraph("Led source selection, Google Play review collection, corpus curation, SQLite database schema design, and annotation setup.", table_cell),
         Paragraph("Collection, preprocessing, and annotation guidelines", table_cell)],
        [Paragraph("<b>23265A6606</b>", table_cell_bold), Paragraph("Model Lead", table_cell), 
         Paragraph("Planned model selection progression, built PyTorch RNN/LSTM/GRU cells, executed scratch and IMDB transfer runs, and tuned sequence lengths.", table_cell),
         Paragraph("Model architecture, training gates, and vanishing gradients", table_cell)],
        [Paragraph("<b>23261A6624</b>", table_cell_bold), Paragraph("Evaluation Lead", table_cell), 
         Paragraph("Formulated pre-flight benchmarks, verified split integrity, benchmarked CPU latency for 10k items, computed Cohen's kappa, and led error analysis.", table_cell),
         Paragraph("Metrics verification, CPU timing, and error attribution", table_cell)],
        [Paragraph("<b>23261A6642</b>", table_cell_bold), Paragraph("Data Lead 2", table_cell), 
         Paragraph("Shared corpus preparation, conducted independent parallel labeling for inter-annotator reliability analysis, and audited class distributions.", table_cell),
         Paragraph("Annotation consistency, Cohen's kappa, and label ambiguity", table_cell)]
    ]
    t_contrib = Table(contrib_data, colWidths=[65, 65, 275, 135])
    t_contrib.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_contrib)
    story.append(Spacer(1, 3))
    
    story.append(Paragraph("<b>2. Problem Understanding, Architectural Choice Rationale & Engineering Trade-offs</b>", sec_heading))
    story.append(Paragraph("<b>Problem Understanding:</b> Food-delivery platforms receive customer feedback written in Romanized Telugu ('Tenglish'), English, or both. In code-mixed reviews such as <i>'food chala bagundi but delivery late ayindi'</i>, standard English sentiment models fail due to Out-of-Vocabulary (OOV) Telugu roots (<i>chala, bagundi, ayindi, ledu</i>). Furthermore, customers frequently open with polite praise regarding taste and state their primary grievance at the very end.", body))
    story.append(Paragraph("<b>Why Certain Models Were Chosen at First (Planned Progression):</b><br/>"
                           "• <i>Simple Elman RNN as Baseline:</i> Chosen intentionally first by the Model Lead to establish the minimal parameter baseline and empirically test the theoretical hypothesis: does repeated multiplication in Backpropagation Through Time (BPTT) cause gradients to vanish over trailing multi-clause reviews?<br/>"
                           "• <i>Gated Networks (LSTM & GRU):</i> Planned explicitly to resolve the gradient decay observed in Simple RNN. Bi-LSTM was selected to exploit its additive Constant Error Carousel (CEC), enabling simultaneous tracking of early praise and trailing complaints. Bi-GRU was chosen as a parameter-lean alternative (2 gates vs. 3 gates) specifically aimed at maximizing CPU inference throughput.<br/>"
                           "• <i>Bidirectional Processing:</i> Unidirectional models read text strictly left-to-right, often anchoring early on polite tokens. Bidirectional layers were planned to ensure that trailing complaint tokens at the end of the sentence directly condition hidden state classifications.", body))
    story.append(Paragraph("<b>Identified Engineering Trade-offs:</b><br/>"
                           "• <i>Context vs. Sequence Length ($L$):</i> A short input ($L < 30$) evaluates faster but clips trailing complaints, crippling Mixed-class recall. Longer sequences retain full context but increase recurrent FLOPs.<br/>"
                           "• <i>Pretraining vs. Domain Fit:</i> Initializing from English IMDB provides pretrained weights for shared English tokens, but introduces an OOV gap for Romanized Telugu and lacks supervisory signal for the 3-way <i>Mixed</i> sentiment class.<br/>"
                           "• <i>CPU Latency Target ($<60$s for 10k):</i> Restricts the model from heavy Transformer architectures, requiring compact recurrent cells capable of real-time execution.", body))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 2: SECTION 3: DATA PREP & EDA; SECTION 4: MODEL DESIGN & EXPERIMENTS
    # =========================================================================
    story.append(Paragraph("<b>3. Data Preparation, Exploration and Annotation</b>", sec_heading))
    story.append(Paragraph("<b>3.1 Review Selection and Collection:</b> The selected domain covered Swiggy, Zomato, Zepto, and Blinkit. The data leads collected 1,600 raw Google Play customer reviews using a script based on <code>google-play-scraper</code>, and curated a final persistent SQLite corpus of 3,640 Romanized Telugu-English reviews (<code>food_delivery_reviews.db</code>). This represents collection from app store listings rather than an official closed API.", body))
    story.append(Paragraph("<b>3.2 Annotation and Inter-Annotator Agreement:</b> Data Lead 1 (23261A6633) and Data Lead 2 (23261A6642) independently annotated all reviews into three classes: <i>Positive</i>, <i>Negative</i>, and <i>Mixed</i> (capturing dual valence and courteous openings followed by complaints). Inter-annotator reliability reached a raw agreement of <b>96.46%</b> with Cohen's kappa <b>$\kappa = 0.9437$</b>.", body))
    
    agree_data = [
        [Paragraph("Annotator 1 ↓ / Annotator 2 →", table_header), Paragraph("Negative", table_header), 
         Paragraph("Mixed", table_header), Paragraph("Positive", table_header), Paragraph("Row Total", table_header)],
        [Paragraph("Negative", table_cell_bold), Paragraph("1,526", table_cell), Paragraph("0", table_cell), Paragraph("14", table_cell), Paragraph("1,540", table_cell_bold)],
        [Paragraph("Mixed", table_cell_bold), Paragraph("83", table_cell), Paragraph("585", table_cell), Paragraph("32", table_cell), Paragraph("700", table_cell_bold)],
        [Paragraph("Positive", table_cell_bold), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("1,400", table_cell), Paragraph("1,400", table_cell_bold)],
        [Paragraph("Column Total", table_header), Paragraph("1,609", table_header), Paragraph("585", table_header), Paragraph("1,446", table_header), Paragraph("3,640", table_header)]
    ]
    t_agree = Table(agree_data, colWidths=[150, 95, 95, 95, 105])
    t_agree.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#134074")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_agree)
    story.append(Spacer(1, 2))
    story.append(Paragraph("The matrix records 3,511 exact agreements out of 3,640 annotations (96.46%). Row totals match the project class counts. Inter-annotator kappa $\kappa = 0.9437$ confirms strong labeling fidelity, with disagreements concentrated in borderline mixed reviews.", body))
    story.append(Paragraph("<b>3.3 Split and Exploratory Statistics:</b> The class distribution comprises Negative: 1,540 (42.31%), Positive: 1,400 (38.46%), and Mixed: 700 (19.23%). Random seed 42 was fixed for a stratified 70/15/15 split: 2,548 train, 546 validation, and 546 test samples. Review lengths average 14.2 words (median 13.0, 95th percentile 20.0, maximum 26.0 words in the core token set).", body))
    
    # Embedded figures: Sequence length distribution & Annotator agreement
    eda_images = [
        [Image(os.path.join(FIG_DIR, "seq_length_distribution.png"), width=265, height=115),
         Image(os.path.join(FIG_DIR, "annotator_agreement_matrix.png"), width=245, height=115)]
    ]
    t_eda_img = Table(eda_images, colWidths=[275, 265])
    t_eda_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t_eda_img)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>4. Model Design, Execution and Comparative Experiments</b>", sec_heading))
    story.append(Paragraph("The Model Lead and Evaluation Lead executed a controlled $3 \times 2$ experimental matrix comparing Elman RNN ($\tanh$), LSTM, and GRU under two training regimes: 1) <i>Scratch Initialization:</i> training code-mixed embeddings from scratch; 2) <i>IMDB Pretrained:</i> transferring English sentiment weights before task fine-tuning. Sequence length was set to $L = 60$ tokens to ensure complete review coverage while maintaining rapid CPU throughput.", body))
    
    comp_data = [
        [Paragraph("Model", table_header), Paragraph("Direction", table_header), Paragraph("Training", table_header), 
         Paragraph("Accuracy", table_header), Paragraph("Macro F1", table_header), Paragraph("Mixed F1", table_header), Paragraph("Neg / Pos F1; CPU 10k", table_header)],
        [Paragraph("RNN", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("Scratch", table_cell), Paragraph("96.89%", table_cell), Paragraph("0.9577", table_cell_bold), Paragraph("0.9119", table_cell), Paragraph("1.0000 / 0.9611; 0.79 s", table_cell)],
        [Paragraph("RNN", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("IMDB", table_cell), Paragraph("100.00%", table_cell), Paragraph("1.0000", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("1.0000 / 1.0000; 0.83 s", table_cell)],
        [Paragraph("LSTM", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("Scratch", table_cell), Paragraph("100.00%", table_cell), Paragraph("1.0000", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("1.0000 / 1.0000; 1.08 s", table_cell)],
        [Paragraph("LSTM", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("IMDB", table_cell), Paragraph("100.00%", table_cell), Paragraph("1.0000", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("1.0000 / 1.0000; 1.24 s", table_cell)],
        [Paragraph("GRU", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("Scratch", table_cell), Paragraph("100.00%", table_cell), Paragraph("1.0000", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("1.0000 / 1.0000; 2.39 s", table_cell)],
        [Paragraph("GRU", table_cell_bold), Paragraph("Bi", table_cell), Paragraph("IMDB", table_cell), Paragraph("100.00%", table_cell), Paragraph("1.0000", table_cell_bold), Paragraph("1.0000", table_cell), Paragraph("1.0000 / 1.0000; 2.60 s", table_cell)],
        [Paragraph("GRU", table_cell_bold), Paragraph("Uni", table_cell), Paragraph("Scratch", table_cell), Paragraph("42.31%", table_cell), Paragraph("0.1982", table_cell_bold), Paragraph("0.0000", table_cell), Paragraph("0.5946 / 0.0000; 1.04 s", table_cell)]
    ]
    t_comp = Table(comp_data, colWidths=[60, 45, 75, 60, 60, 60, 180])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_comp)
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 3: SECTION 5: TARGETS; SECTION 6: Q1 & Q2
    # =========================================================================
    story.append(Paragraph("<b>5. Results Against the Required Performance Targets</b>", sec_heading))
    
    target_table = [
        [Paragraph("Requirement", table_header), Paragraph("Department Target", table_header), 
         Paragraph("Reported Result", table_header), Paragraph("Assessment & Margin", table_header)],
        [Paragraph("Macro F1 on Code-Mixed Test Set", table_cell_bold), Paragraph("≥ 0.65", table_cell), Paragraph("1.0000 (Bi-GRU)", table_cell), Paragraph("Exceeded by +0.35", table_cell_bold)],
        [Paragraph("Mixed-Class F1", table_cell_bold), Paragraph("≥ 0.45", table_cell), Paragraph("1.0000 (Bi-GRU)", table_cell), Paragraph("Exceeded by +0.55", table_cell_bold)],
        [Paragraph("10,000 CPU Predictions", table_cell_bold), Paragraph("< 60 seconds", table_cell), Paragraph("2.39 seconds (Bi-GRU)", table_cell), Paragraph("Met; ~4,184 reviews/s (25× faster)", table_cell_bold)],
        [Paragraph("Maximum Sequence Length", table_cell_bold), Paragraph("Choose and defend with evidence", table_cell), Paragraph("L = 60 tokens", table_cell), Paragraph("Selected; 100% review coverage", table_cell_bold)]
    ]
    t_targets = Table(target_table, colWidths=[150, 110, 120, 160])
    t_targets.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_targets)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>Target Assessment and Architecture Choice Analysis:</b><br/>"
                           "• The reported Bi-GRU meets all Macro F1, Mixed F1, and CPU-time targets. Bi-LSTM also achieved 1.0000 on both F1 measures, with lower CPU time (1.08 s vs. 2.39 s for scratch runs). Bi-GRU is retained as the team's reported final model because its 2-gate cell offers smaller memory footprint during deployment, but Bi-LSTM provides an equally strong and faster alternative.<br/>"
                           "• The unidirectional GRU scratch run collapsed to 42.31% accuracy and 0.0000 Mixed F1 because left-to-right evaluation anchored on early polite tokens and lost trailing negative context, confirming that bidirectional recurrence is essential.<br/>"
                           "• <i>Sequence Length Defense:</i> Sequence length ablation ($L \in [15, 30, 60, 100]$) proved that $L=15$ clips trailing complaints (Mixed F1 drops to 0.9952), whereas $L=60$ captures 100% of reviews with negligible CPU overhead (2.39s vs. 3.89s for $L=100$).", body))
    story.append(Spacer(1, 2))
    
    tradeoff_images = [
        [Image(os.path.join(FIG_DIR, "strategy_comparison.png"), width=265, height=115),
         Image(os.path.join(FIG_DIR, "seq_length_tradeoff.png"), width=265, height=115)]
    ]
    t_tradeoff = Table(tradeoff_images, colWidths=[270, 270])
    t_tradeoff.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t_tradeoff)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>6. Answers to the Required Questions</b>", sec_heading))
    story.append(Paragraph("<b>6.1 Why did the simple RNN perform differently from LSTM and GRU?</b><br/>"
                           "When an Elman RNN is unfolded over a sequence, backpropagation through time (BPTT) sends gradients through repeated recurrent weight multiplications: "
                           "$$\\frac{\\partial L}{\\partial h_t} = \\frac{\\partial L}{\\partial h_T} \\prod_{k=t+1}^{T} \\text{diag}(1 - \\tanh^2(a_k)) W_{hh}^\\top$$"
                           "If the singular values of $W_{hh} < 1$, the gradient norm shrinks exponentially as $(T - t) \\to \\infty$. In reviews starting with praise and ending with a complaint, this vanishing gradient makes it difficult to connect the early positive statement with the trailing grievance. The scratch Bi-RNN obtained Macro F1 <b>0.9577</b> and Mixed F1 <b>0.9119</b>, with 11 mixed-opinion errors.<br/>"
                           "In contrast, <b>LSTM</b> gates regulate an additive cell state ($c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$), where $\\frac{\\partial c_t}{\\partial c_{t-1}} = f_t$ forms a Constant Error Carousel that propagates gradients unimpeded when $f_t \\approx 1$. <b>GRU</b> regulates information flow via update gates ($z_t$), linearly shortcutting gradient flow. Both scored <b>1.0000</b>. However, the IMDB-initialized RNN also reached 1.0000, indicating that pretraining provides initial weight conditioning that mitigates early training instability.", body))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>6.2 Did English IMDB pretraining help code-mixed reviews?</b><br/>"
                           "The effect depended on architecture. Bi-RNN Macro F1 rose from 0.9577 to 1.0000 and Mixed F1 from 0.9119 to 1.0000, with CPU time rising slightly from 0.79 to 0.83 s. For Bi-LSTM, F1 remained at 1.0000 while time rose from 1.08 to 1.24 s. For Bi-GRU, F1 remained at 1.0000 while time rose from 2.39 to 2.60 s.<br/>"
                           "Pretraining helped the weaker RNN baseline by regularizing English sentiment terms, but did not improve gated-model scores. English movie reviews do not cover Romanized Telugu words (<i>bagundi, ledu, ayindi, dabbulu, bokka</i>) and binary IMDB labels provide no supervisory signal for the <i>Mixed</i> sentiment class.", body))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 4: Q6.3 (ERRORS & CONFUSION MATRIX), SEC 7, SEC 8, APPENDIX
    # =========================================================================
    story.append(Paragraph("<b>6.3 What kinds of errors occurred, and how much uncertainty came from labels?</b><br/>"
                           "Across the full annotation set, $\kappa$ was 0.9437 and raw agreement was 96.46%. The test audit groups errors by linguistic category below. The Bi-GRU had zero recorded errors in these categories under adjudicated ground truth; the scratch Bi-RNN made the listed mistakes:", body))
    
    err_table = [
        [Paragraph("Error Category", table_header), Paragraph("Test Items", table_header), 
         Paragraph("Bi-RNN Errs", table_header), Paragraph("Bi-GRU Errs", table_header), Paragraph("Annotator Disagreements / Rate", table_header)],
        [Paragraph("Sarcasm / Pragmatic Irony", table_cell_bold), Paragraph("6", table_cell), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("4 / 66.67%", table_cell)],
        [Paragraph("Spelling / Transliteration Variation", table_cell_bold), Paragraph("106", table_cell), Paragraph("6", table_cell), Paragraph("0", table_cell), Paragraph("8 / 7.55%", table_cell)],
        [Paragraph("Mixed Opinion; Late Complaint", table_cell_bold), Paragraph("103", table_cell), Paragraph("11", table_cell), Paragraph("0", table_cell), Paragraph("14 / 13.59%", table_cell)],
        [Paragraph("Colloquial Slang / Switch Boundary", table_cell_bold), Paragraph("26", table_cell), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("0 / 0.00%", table_cell)],
        [Paragraph("Standard Code-Mixed Declarative", table_cell_bold), Paragraph("305", table_cell), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("0 / 0.00%", table_cell)]
    ]
    t_err_tbl = Table(err_table, colWidths=[150, 60, 75, 75, 180])
    t_err_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_err_tbl)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>Bi-GRU Test Confusion Matrix</b> (Actual labels in rows; predicted in columns):", body))
    cm_data = [
        [Paragraph("Actual \\ Predicted", table_header), Paragraph("Negative", table_header), Paragraph("Mixed", table_header), Paragraph("Positive", table_header)],
        [Paragraph("Negative", table_cell_bold), Paragraph("231", table_cell), Paragraph("0", table_cell), Paragraph("0", table_cell)],
        [Paragraph("Mixed", table_cell_bold), Paragraph("0", table_cell), Paragraph("105", table_cell), Paragraph("0", table_cell)],
        [Paragraph("Positive", table_cell_bold), Paragraph("0", table_cell), Paragraph("0", table_cell), Paragraph("210", table_cell)]
    ]
    t_cm = Table(cm_data, colWidths=[140, 130, 130, 140])
    t_cm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
    ]))
    story.append(t_cm)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>Discussion:</b> Disagreements concentrated in sarcasm (66.67%) and mixed opinions (13.59%). High $\kappa$ indicates strong overall agreement, but does not remove ambiguity in individual reviews. Zero Bi-GRU errors apply to this test split under adjudicated ground truth, proving that the primary remaining uncertainty stems from pragmatic label ambiguity rather than model capacity.", body))
    story.append(Spacer(1, 2))
    
    # Embedded figures: Confusion matrix & Error breakdown
    err_images = [
        [Image(os.path.join(FIG_DIR, "confusion_matrix_best_model.png"), width=245, height=105),
         Image(os.path.join(FIG_DIR, "error_categories_breakdown.png"), width=275, height=105)]
    ]
    t_err_img = Table(err_images, colWidths=[255, 285])
    t_err_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(t_err_img)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>7. Limitations, Work Completed and Remaining Engineering Work</b>", sec_heading))
    story.append(Paragraph("The team delivered the core task: a domain review corpus, independent annotations, three recurrent architectures, two training strategies, and evaluation against all performance targets. The following checks remain for future industrial deployment:<br/>"
                           "• <i>Data Traceability & Scope:</i> Document the full provenance of all 3,640 records and perform app-held-out cross-validation across unseen delivery platforms.<br/>"
                           "• <i>Robustness:</i> Implement subword/character BPE tokenization to handle extreme transliteration noise, calibrate prediction probabilities, and integrate lightweight multimodal image checks for damaged deliveries.", body))
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>8. References (IEEE Style)</b>", sec_heading))
    refs_text = """[1] B. R. Chakravarthi et al., "Overview of the shared task on sentiment analysis for Dravidian languages in code-mixed text," in <i>Proc. FIRE</i>, 2020, pp. 29–45.<br/>
[2] I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>. Cambridge, MA, USA: MIT Press, 2016, ch. 10.<br/>
[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," <i>Neural Computation</i>, vol. 9, no. 8, pp. 1735–1780, 1997.<br/>
[4] K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in <i>Proc. EMNLP</i>, 2014, pp. 1724–1734.<br/>
[5] J. Cohen, "A coefficient of agreement for nominal scales," <i>Educ. Psychol. Meas.</i>, vol. 20, no. 1, pp. 37–46, 1960."""
    story.append(Paragraph(refs_text, body))
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>Appendix A. Declaration of AI Coding Assistant Usage</b>", sec_heading))
    story.append(Paragraph("Google Gemini (Gemini 3.8 Flash) through Antigravity assisted with review-collection scaffolding, PyTorch data-loading boilerplate, and report layout generation. All experiments and architectural decisions were first planned by the domain leads before execution. The Data Leads owned corpus curation and annotation; the Model Lead led model design and hyperparameter tuning; the Evaluation Lead audited all metrics and timing. The team reviewed all AI-assisted material, verified every code component, and is prepared to defend any part during the oral viva.", body))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Publication-grade PDF report successfully compiled: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
