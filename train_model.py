"""
Fake News Detection Model Training Pipeline.

This script performs data loading, text preprocessing, TF-IDF vectorization,
trains Logistic Regression and Multinomial Naive Bayes models, compares their
performances, saves the best model and vectorizer using Joblib, and exports
a comprehensive evaluation report.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Reconfigure stdout to UTF-8 for Windows consoles with unicode paths
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import preprocessing utility function
from preprocess import clean_text
from dataset.download_data import ensure_datasets_exist

# Define File Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FAKE_PATH = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATASET_DIR, "True.csv")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def load_and_prepare_data():
    """
    Loads Fake.csv and True.csv datasets, assigns binary labels,
    combines title and text content, and pre-processes text data.

    Returns:
        pd.DataFrame: Processed dataframe with 'cleaned_text' and 'label'.
                      (0 = FAKE, 1 = REAL)
    """
    # Step 1: Ensure dataset files exist
    ensure_datasets_exist()

    print("\n[1/6] Loading datasets...")
    df_fake = pd.read_csv(FAKE_PATH)
    df_true = pd.read_csv(TRUE_PATH)

    # Assign class labels: 0 for FAKE news, 1 for REAL news
    df_fake['label'] = 0
    df_true['label'] = 1

    print(f"  - Loaded Fake News samples: {len(df_fake)}")
    print(f"  - Loaded Real News samples: {len(df_true)}")

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Fill missing values if any
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    # Step 2: Combine title and text to enrich textual context
    print("[2/6] Combining headline title and main text content...")
    df['combined_content'] = df['title'] + " " + df['text']

    # Step 3: Apply NLTK text preprocessing & stemming
    print("[3/6] Cleaning text (removing punctuation, stopwords, & applying Porter stemming)...")
    df['cleaned_text'] = df['combined_content'].apply(clean_text)

    # Shuffle dataset rows to prevent ordering bias
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def train_and_evaluate():
    """
    Executes TF-IDF feature extraction, trains Logistic Regression & Multinomial Naive Bayes,
    evaluates metric scores, saves the top-performing model, vectorizer, and metrics JSON.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load dataset
    df = load_and_prepare_data()

    X = df['cleaned_text']
    y = df['label']

    # Step 4: Split data into training and testing sets (80% train, 20% test)
    print("\n[4/6] Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 5: Convert text into TF-IDF (Term Frequency-Inverse Document Frequency) vectors
    print("[5/6] Extracting features using TfidfVectorizer (unigrams & bigrams)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 6: Initialize and train Machine Learning models
    print("[6/6] Training Logistic Regression and Multinomial Naive Bayes models...\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)
    }

    metrics_results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_object = None

    print("=" * 70)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("=" * 70)

    for name, model in models.items():
        # Train model
        model.fit(X_train_tfidf, y_train)

        # Predict test data
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm
        }

        print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

        # Track best model based on F1-score / Accuracy
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model_object = model

    print("=" * 70)
    print(f"\n🏆 Best Selected Model: {best_model_name} (F1-Score: {best_f1_score:.4f})")

    # Save the trained vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"  - Saved TF-IDF Vectorizer to: {VECTORIZER_PATH}")

    # Save the best trained model using Joblib
    joblib.dump(best_model_object, BEST_MODEL_PATH)
    print(f"  - Saved Best Model ({best_model_name}) to: {BEST_MODEL_PATH}")

    # Save metrics evaluation report
    report_data = {
        "best_model": best_model_name,
        "models": metrics_results
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"  - Exported model metrics to: {METRICS_PATH}")
    print("\nModel training pipeline completed successfully!\n")


if __name__ == "__main__":
    train_and_evaluate()
