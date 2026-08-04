"""
Text Preprocessing Module for Fake News Detection.

This module provides data cleaning, tokenization, stopword removal,
and stemming functionalities using NLTK and Python standard libraries.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are downloaded silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize the Porter Stemmer and Stopwords list
stemmer = PorterStemmer()
english_stopwords = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Cleans raw text by performing the following steps:
    1. Lowercasing the string.
    2. Removing URLs, HTML tags, and special characters.
    3. Removing punctuation characters.
    4. Tokenizing words and removing NLTK English stopwords.
    5. Applying Porter Stemming to reduce words to their base root form.
    
    Parameters:
        text (str): Raw input text (news title, body, or combined text).

    Returns:
        str: Processed, stemmed, space-separated cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Convert text to lowercase
    text = text.lower()

    # Step 2: Remove URLs (http/https/www links)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Step 3: Remove HTML tags if present
    text = re.sub(r'<.*?>', '', text)

    # Step 4: Remove digits and numbers
    text = re.sub(r'\d+', '', text)

    # Step 5: Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 6: Split into word tokens
    words = text.split()

    # Step 7: Remove stop words and apply stemming using PorterStemmer
    cleaned_words = [
        stemmer.stem(word)
        for word in words
        if word not in english_stopwords and len(word) > 1
    ]

    # Step 8: Join stemmed tokens back into a single clean string
    return " ".join(cleaned_words)


if __name__ == "__main__":
    # Self-test block for preprocessing verification
    sample_raw_news = "BREAKING: Scientists discover revolutionary energy source in 2026! Visit https://example.com for details."
    processed = clean_text(sample_raw_news)
    print("Raw Text:      ", sample_raw_news)
    print("Processed Text:", processed)
