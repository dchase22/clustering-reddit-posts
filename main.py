from datasets import load_dataset
from itertools import islice
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
import re

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
    print(X.shape)

if __name__ == "__main__":
    main()
