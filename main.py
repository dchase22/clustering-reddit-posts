from datasets import load_dataset
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = f"[{string.punctuation}]"
DATA_ITEMS = 10000

def preprocess(text_arr):
    processed = []
    for val in text_arr:
        val = val.lower().strip()
        val = re.sub(PUNCTUATION, "", val)
        val = re.sub(r"\s+", " ", val)
        val = " ".join([word for word in val.split() if word not in STOP_WORDS])
        processed.append(val)
    return processed

def main():
    # Stream dataset from hugging face using HTTP
    ds = load_dataset("wenknow/reddit_dataset_44", split="train", streaming=True)

    # Load posts out of first 10000 data items
    posts = [row["text"] for row in islice(ds, DATA_ITEMS) if row["dataType"] == "post"]

    # Preprosess text
    posts = preprocess(posts)

    # Vectorize posts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(posts)
    feature_names = vectorizer.get_feature_names_out()

    # Used TruncatedSVD to reduce the demensionality of TF-IDF matrix X
    svd = TruncatedSVD(n_components=100, random_state=0)
    X_reduced = svd.fit_transform(X)

    # Run KMeans algorithm to generate 5 clusters in vectorized posts
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X_reduced)

    # Print the top terms of each document in each cluster
    for i in range(kmeans.n_clusters):
        print(f"================Cluster {i}================")
        mask = kmeans.labels_ == i
        rows_in_cluster = X[mask]
        
        for j in range(min(10, rows_in_cluster.shape[0])):
            row = rows_in_cluster[j]
            top3_vals = row.data.argsort()[::-1][:3]
            top_indices = row.indices[top3_vals]
            print(feature_names[top_indices])

if __name__ == "__main__":
    main()
