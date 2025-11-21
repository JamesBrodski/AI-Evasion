#################################################################
# Library Installation (bash)
#################################################################

# Install the AI Library (or update it)
# pip install --upgrade git+https://github.com/PandaSt0rm/htb-ai-library

#################################################################
# Initial Setup and Dependencies (python)
#################################################################

import json
import pickle
import random
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

# Reproducibility
random.seed(1337)
np.random.seed(1337)

#################################################################
# Machine learning libraries (python)
#################################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#################################################################
# HTB libraries (python)
#################################################################

from htb_ai_library import (
    AZURE,
    HACKER_GREY,
    HTB_GREEN,
    MALWARE_RED,
    NODE_BLACK,
    NUGGET_YELLOW,
    WHITE,
    AQUAMARINE,
    load_model,
    save_model,
)

#################################################################
# Loading the SMS Spam Dataset (python)
#################################################################

print("\n[*] Loading SMS Spam Dataset...")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
dataset_path = data_dir / "sms_spam.csv"

#################################################################
# Checking for Cached Data (python)
#################################################################

if dataset_path.exists():
    print(f"[+] Using cached dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
else:
    print("[*] Downloading from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = data_dir / "sms_spam.zip"
    urllib.request.urlretrieve(url, zip_path)

#################################################################
# Extracting and Processing the Download (python)
#################################################################

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("utf-8").strip() for line in f]

    # Parse tab-separated format
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            data.append({"label": parts[0].lower(), "message": parts[1]})

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    zip_path.unlink()
    print(f"[+] Dataset saved to {dataset_path}")

#################################################################
# Understanding the Dataset (python)
#################################################################

# print(f"[+] Loaded {len(df)} messages")
# print(f"    Spam: {sum(df['label'] == 'spam')}")
# print(f"    Ham: {sum(df['label'] == 'ham')}")

#################################################################
# Examine a few message samples (python)
#################################################################

# print("\n[*] Sample messages:")
# print("\nSPAM samples:")
# for msg in df[df['label'] == 'spam']['message'].head(3):
#     print(f"  - {msg[:80]}...")
# print("\nHAM samples:")
# for msg in df[df['label'] == 'ham']['message'].head(3):
#     print(f"  - {msg[:80]}...")

#################################################################
# Minimal Cleaning for Analysis (python)
#################################################################

import html as html_module
import unicodedata

def minimal_clean(text):
    """
    Minimal cleaning that preserves spam indicators.

    Parameters: text (str) raw SMS message
    Returns: str cleaned text with entities decoded, unicode normalized, and
             whitespace collapsed while keeping informative symbols.
    """
    # Decode HTML entities (e.g., &amp; -> &)
    text = html_module.unescape(text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)

    return text.strip()

#################################################################
# Final Cleaning for Vectorization
#################################################################

def clean_text(text):
    """
    Final cleaning for vectorization.

    Converts to lowercase and removes only problematic characters so that
    informative symbols remain available to the vectorizer.

    Parameters:
        text (str): Preprocessed message from `minimal_clean`.

    Returns:
        str: Normalized, whitespace‑collapsed text ready for tokenization.
    """
    text = text.lower()
    # Keep numbers, currency symbols, punctuation - they're spam features!
    # Only remove truly problematic characters
    text = re.sub(r'[^\w\s£$€¥!?.,;:\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#################################################################
# Applying Preprocessing
#################################################################

print("[*] Applying minimal text cleaning (preserving spam indicators)...")
df['preprocessed'] = df['message'].apply(minimal_clean)

# Apply final cleaning for vectorization
df['clean_message'] = df['preprocessed'].apply(clean_text)

#################################################################
# Inspect what preprocessing preserved
#################################################################

# print("\n[*] Sample spam messages with preserved features:")
# spam_samples = df[df['label'] == 'spam'].sample(3, random_state=42)
# for idx, row in spam_samples.iterrows():
#     msg = row['preprocessed'][:100] + "..." if len(row['preprocessed']) > 100 else row['preprocessed']
#     print(f"  - {msg}")

#################################################################
# Removing Duplicates and Empty Messages
#################################################################

# Remove only exact duplicates
original_size = len(df)
df = df.drop_duplicates(subset=['label', 'clean_message'])
print(f"\n[+] Removed {original_size - len(df)} duplicates")

# Remove empty messages
before_empty = len(df)
df = df[df['clean_message'].str.len() > 0]
print(f"[+] Removed {before_empty - len(df)} empty messages")

#################################################################
# Creating Train and Test Sets
#################################################################

X = df['clean_message'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[+] Data split:")
print(f"    Training: {len(X_train)} messages")
print(f"    Testing: {len(X_test)} messages")

#################################################################
# Training the Naive Bayes Classifier
#################################################################

print("\n[*] Training Naive Bayes classifier...")

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "spam_classifier.pkl"

if model_path.exists():
    print(f"[+] Loading saved model from {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data['vectorizer']
        classifier = saved_data['classifier']

    # Transform data using existing vocabulary
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

else:
    # Configure vectorizer to capture spam patterns
    vectorizer = CountVectorizer(
        max_features=3000,
        token_pattern=r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+',
        lowercase=True,
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Save model for reproducibility
    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': classifier}, f)
    print(f"[+] Model saved to {model_path}")

#################################################################
# Evaluating Model Performance
#################################################################

# Calculate accuracy scores
train_acc = classifier.score(X_train_vec, y_train)
test_acc = classifier.score(X_test_vec, y_test)
print(f"[+] Training accuracy: {train_acc:.4f}")
print(f"[+] Testing accuracy: {test_acc:.4f}")

# Get detailed predictions
y_pred = classifier.predict(X_test_vec)
print("\n[*] Classification Report:")
print(classification_report(y_test, y_pred))
