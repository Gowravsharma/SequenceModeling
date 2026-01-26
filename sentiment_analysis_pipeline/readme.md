# End-to-End Sentiment Analysis Pipeline (Amazon Reviews)

## Overview

This project implements an **end-to-end NLP pipeline for sentiment classification** on real-world Amazon product reviews. The goal is not only to achieve high accuracy, but to demonstrate a **complete machine learning workflow**: data acquisition, preprocessing, baseline modeling, deep learning, evaluation, and analysis.

The project compares a traditional machine learning baseline with a transformer-based model to highlight the impact of contextual language understanding.

---

## Problem Statement

User-generated reviews contain noisy, unstructured text with domain-specific language. The task is to classify reviews into **positive** or **negative** sentiment while handling real-world challenges such as long texts, negation, and mixed sentiment.

---

## Dataset

* **Source:** IMDB Movie Reviews Dataset (Hugging Face)
* **Size used:** 50,000 reviews (official train/test split)
* **Labels:**

  * `0` → Negative
  * `1` → Positive

The IMDB dataset consists of long-form movie reviews with clear sentiment polarity and is a standard benchmark for sentiment classification tasks.

---

## Project Structure

```
sentiment_analysis_pipeline/
│
├── data/
│   ├── raw/              # Raw downloaded data (IMDB reviews)
│   └── processed/        # Cleaned and preprocessed data
│
├── src/
│   ├── data_download.py  # IMDB dataset download
│   ├── preprocess.py     # Text cleaning and preprocessing
│   ├── train_baseline.py # TF-IDF + Logistic Regression
│   ├── train_bert.py     # DistilBERT fine-tuning
│   └── evaluate.py       # Model evaluation
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Preprocessing

The raw reviews were cleaned using standard NLP preprocessing steps:

* Lowercasing
* Removal of HTML tags and URLs
* Removal of non-alphabetic characters
* Whitespace normalization

These steps reduce noise and improve generalization across models.

---

### 2. Baseline Model

A traditional machine learning baseline was implemented to establish a reference point:

* **TF-IDF Vectorization** (10,000 features)
* **Logistic Regression** classifier

This model captures lexical sentiment cues but lacks contextual understanding.

**Baseline Performance (TF-IDF + Logistic Regression):**

* Accuracy: ~0.89
* F1-score: ~0.89
* F1-score: ~0.89

---

### 3. Deep Learning Model

A transformer-based model was used to capture contextual semantics:

* **Model:** DistilBERT (pretrained)
* **Framework:** PyTorch + Hugging Face Transformers
* **Loss Function:** Cross-Entropy Loss (default)
* **Optimizer:** AdamW

DistilBERT was chosen for its balance between performance and computational efficiency.

---

### 4. Evaluation Strategy

The model was evaluated on a held-out test set of **10,000 reviews** using:

* Precision
* Recall
* F1-score
* Confusion Matrix

F1-score was emphasized due to its robustness to class imbalance.

---

## Results

### ### DistilBERT Evaluation Results

```
Accuracy: 0.96
F1-score: 0.96
```

**Classification Report (Test Set):**

* Negative (0): Precision 0.97 | Recall 0.96 | F1 0.96
* Positive (1): Precision 0.96 | Recall 0.97 | F1 0.96

**Confusion Matrix:**

```
[[4841  224]
 [ 152 4783]]
```

### Performance Comparison

| Model                        | F1-score |
| ---------------------------- | -------- |
| TF-IDF + Logistic Regression | ~0.89    |
| DistilBERT                   | ~0.96    |

The transformer-based model significantly outperforms the baseline by leveraging contextual word representations.

---

## Analysis & Observations

* The baseline model performs well due to strong sentiment keywords but struggles with negation and long-range dependencies.
* DistilBERT effectively captures context, reducing both false positives and false negatives.
* Remaining errors are primarily due to:

  * Mixed-sentiment reviews
  * Neutral reviews forced into binary labels
  * Domain-specific sarcasm

---

## How to Run

```bash
pip install -r requirements.txt
python src/data_download.py
python src/preprocess.py
python src/train_baseline.py
python src/train_bert.py
python src/evaluate.py
```

---

## Key Takeaways

* Strong baselines are essential for meaningful deep learning comparisons.
* Evaluation beyond accuracy is critical for real-world NLP systems.
* Transformer models provide substantial gains for contextual text understanding.

---

## Future Work

* Extend to multi-class or aspect-based sentiment analysis
* Apply knowledge distillation for model compression
* Deploy the model using FastAPI or Streamlit
* Perform detailed error analysis with visualization

---

## Skills Demonstrated

* Natural Language Processing
* Data preprocessing and feature engineering
* Classical ML and deep learning
* Model evaluation and analysis
* PyTorch and Hugging Face Transformers

---

This project demonstrates an industry-aligned approach to building, evaluating, and interpreting NLP models on real-world data.
