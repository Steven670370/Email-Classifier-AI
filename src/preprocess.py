import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
raw_folder = "../data/raw"
output_folder = "../data/processed"
os.makedirs(output_folder, exist_ok=True)


# Preprocess email data
def run_preprocess():
    # Read all emails
    emails = []
    labels = []

    for filename in os.listdir(raw_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(raw_folder, filename), "r", encoding="utf-8") as f:
                emails.append(f.read())
            name = filename.lower()
            # Assign labels based on filename conventions
            if name.startswith("spam"):
                labels.append(0.9)
            elif name.startswith("maybe_spam"):
                labels.append(0.75)
            elif name.startswith("uncertain"):
                labels.append(0.5)
            elif name.startswith("maybe_ham"):
                labels.append(0.25)
            else:
                labels.append(0.1)

    if len(emails) == 0:
        raise RuntimeError("No .txt files found in ../data/raw")

    # Check if a saved vectorizer exists to avoid re-fitting
    VECTORIZER_PATH = "tfidf_vectorizer.pkl"
    # Check if vectorizer already exists
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        X = vectorizer.transform(emails).toarray()
    else:
        # Convert to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=10)
        X = vectorizer.fit_transform(emails).toarray()
        # Save the vectorizer for future use
        joblib.dump(vectorizer, VECTORIZER_PATH)

    # Save processed data
    np.savetxt(os.path.join(output_folder, "processed_emails.csv"), X, delimiter=",")
    np.savetxt(os.path.join(output_folder, "labels.csv"), labels, delimiter=",")

    print("Preprocessing done.")
    print("Saved processed_emails.csv and labels.csv")


# Load processed data
def load_data():
    """
    Returns:
        dataset: list of tuples (feature_vector, label)
    """
    X = np.loadtxt(os.path.join(output_folder, "processed_emails.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(output_folder, "labels.csv"), delimiter=",")

    return list(zip(X, y))


# Run preprocessing if this file is executed directly
if __name__ == "__main__":
    run_preprocess()