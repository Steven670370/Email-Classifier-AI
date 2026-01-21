import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
raw_folder = "../data/raw"        # Folder with raw emails
output_folder = "../data/processed"  # Folder to save processed data
os.makedirs(output_folder, exist_ok=True)

# Read all emails
emails = []
labels = []
for filename in os.listdir(raw_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(raw_folder, filename), "r", encoding="utf-8") as f:
            emails.append(f.read())
        # Simple labeling rule: spam_*.txt = 1, otherwise 0
        labels.append(1.0 if filename.lower().startswith("spam") else 0.0)

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=10)
X = vectorizer.fit_transform(emails).toarray()

# Save processed data
np.savetxt(os.path.join(output_folder, "processed_emails.csv"), X, delimiter=",")

# Load function for train.py
def load_data():
    """
    Returns:
        dataset: list of tuples ([feature_vector], label)
    """
    X_loaded = np.loadtxt(os.path.join(output_folder, "processed_emails.csv"), delimiter=",")
    dataset = list(zip(X_loaded, labels))
    return dataset