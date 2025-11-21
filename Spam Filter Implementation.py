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

print(f"[+] Loaded {len(df)} messages")
print(f"    Spam: {sum(df['label'] == 'spam')}")
print(f"    Ham: {sum(df['label'] == 'ham')}")