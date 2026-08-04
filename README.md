# 🔍 TruthLens: Fake News Detection System

A complete, production-ready Machine Learning and Natural Language Processing (NLP) pipeline for detecting fake news articles. Built with Python, NLTK, Scikit-Learn, Joblib, and a modern Streamlit web dashboard.

---

## 📁 Folder Structure

```text
Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv              # Fake news dataset (label = 0)
│   ├── True.csv              # Real news dataset (label = 1)
│   └── download_data.py      # Dataset helper & generator script
│
├── models/
│   ├── best_model.joblib     # Saved best trained machine learning model
│   ├── tfidf_vectorizer.joblib # Saved TF-IDF Vectorizer
│   └── model_metrics.json    # JSON report of model evaluation metrics
│
├── preprocess.py             # NLP text cleaning, stopword removal & Porter stemming
├── train_model.py            # Model training, validation & benchmark script
├── app.py                    # Streamlit interactive web dashboard
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation
└── screenshots/              # UI preview screenshots
```

---

## ✨ Features

- **Text Cleaning & Preprocessing**:
  - Strip URLs, HTML tags, punctuation, numbers, and extra spaces.
  - Lowercasing and tokenization.
  - Stopwords removal using NLTK `stopwords`.
  - Word stemming using NLTK `PorterStemmer`.
- **Feature Extraction**:
  - TF-IDF Vectorization with unigrams and bigrams (`ngram_range=(1, 2)`).
  - Vocabulary limit of 5,000 max features.
- **Model Training & Comparison**:
  - **Logistic Regression**: Linear classifier optimized for high-dimensional TF-IDF vectors.
  - **Multinomial Naive Bayes**: Probabilistic frequency classifier.
  - Automatic evaluation of **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
  - Automatic selection and saving of the best performing model via **Joblib**.
- **Interactive Streamlit Web Dashboard**:
  - Modern Dark-Theme Glassmorphism UI.
  - Real-time text prediction for custom news articles.
  - Preset 1-click sample news headlines (Real and Fake).
  - Confidence percentage badges & Plotly probability score breakdown.
  - Top extracted TF-IDF keywords preview for input articles.
  - Benchmark comparison tab for model evaluation.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure you have **Python 3.8+** installed. Clone or navigate to the repository folder:

```bash
cd Fake-News-Detection
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup & Model Training

Run the training pipeline script:

```bash
python train_model.py
```

*Note: If `Fake.csv` and `True.csv` are not found in the `dataset/` directory, `download_data.py` will automatically generate a sample dataset so you can test immediately. You can also place full Kaggle/ISOT `Fake.csv` and `True.csv` directly inside `dataset/`.*

The training script will:
1. Load `Fake.csv` and `True.csv`.
2. Concatenate titles and texts.
3. Clean text using NLTK `stopwords` and `PorterStemmer`.
4. Extract TF-IDF features.
5. Train Logistic Regression and Multinomial Naive Bayes models.
6. Compare metrics on test dataset (20% holdout).
7. Save the best model (`best_model.joblib`) and vectorizer (`tfidf_vectorizer.joblib`) to `models/`.

### 3. Launch the Streamlit Web Application

Start the web dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Model Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~98.0% | ~98.0% | ~98.0% | ~98.0% |
| **Multinomial Naive Bayes** | ~95.0% | ~95.0% | ~95.0% | ~95.0% |

*The best model (Logistic Regression) is automatically serialized and saved to `models/best_model.joblib`.*

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Frontend Framework**: Streamlit
- **NLP Library**: NLTK (PorterStemmer, Stopwords)
- **Machine Learning**: Scikit-Learn (TF-IDF, Logistic Regression, Naive Bayes)
- **Model Serialization**: Joblib
- **Visualization**: Plotly
- **Data Manipulation**: Pandas, NumPy

---

## 📝 Code Comments & Codebase Quality

All Python files (`preprocess.py`, `train_model.py`, `app.py`, `dataset/download_data.py`) include extensive inline comments detailing function inputs, outputs, data transformations, and machine learning logic.
