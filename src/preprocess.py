import os
import joblib
import numpy as np
from config import load_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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
                labels.append(0)
            elif name.startswith("maybe_spam"):
                labels.append(0.2)
            elif name.startswith("uncertain"):
                labels.append(0.5)
            elif name.startswith("maybe_ham"):
                labels.append(0.8)
            else:
                labels.append(1)

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
        config = load_config()
        vectorizer = TfidfVectorizer(
            max_features=config["num_input"],
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        X = vectorizer.fit_transform(emails).toarray()
        # Save the vectorizer for future use
        joblib.dump(vectorizer, VECTORIZER_PATH)
       
    # Step 1: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Compute label-wise means
    y = np.array(labels)
    unique_labels = np.unique(y)
    label_means = np.array([X_scaled[np.isclose(y, lbl)].mean(axis=0) for lbl in unique_labels])

    # Step 3: Compute feature differences
    diff_vector = label_means.max(axis=0) - label_means.min(axis=0)
    diff_vector = diff_vector / (diff_vector.max() + 1e-6)  # normalize

    # Step 4: Weight features by label difference
    X_weighted = X_scaled * diff_vector

    # Save processed data
    np.savetxt(os.path.join(output_folder, "processed_emails.csv"), X_weighted, delimiter=",")
    np.savetxt(os.path.join(output_folder, "labels.csv"), labels, delimiter=",")

    print("Preprocessing done.")
    print("Saved processed_emails.csv and labels.csv")



# Sanity check function
def sanity_check_tfidf(X, labels, label_values=[0, 0.2, 0.5, 0.8, 1], verbose=False):
    y = np.array(labels)
    for v in label_values:
        mask = np.isclose(y, v)
        if mask.any():
            print(f"Label {v}: mean TF-IDF vector = {X[mask].mean(axis=0)}")


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
    # Optionally run sanity check
    view = input("Do you want to see TF-IDF sanity check? (y/n): ").strip().lower()
    # If yes, load data and run sanity check
    if view == 'y':
        X = np.loadtxt(os.path.join(output_folder, "processed_emails.csv"), delimiter=",")
        labels = np.loadtxt(os.path.join(output_folder, "labels.csv"), delimiter=",")
        sanity_check_tfidf(X, labels, verbose=True)