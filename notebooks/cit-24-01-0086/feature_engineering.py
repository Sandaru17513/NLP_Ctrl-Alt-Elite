import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf():
    """
    Create TF-IDF Vectorizer
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    return vectorizer


def fit_transform(vectorizer, train_text):
    """
    Learn vocabulary from training data
    and transform training text.
    """

    return vectorizer.fit_transform(train_text)


def transform(vectorizer, test_text):
    """
    Transform new text using trained TF-IDF.
    """

    return vectorizer.transform(test_text)


def save_vectorizer(vectorizer, path="models/tfidf.pkl"):
    """
    Save TF-IDF model
    """

    joblib.dump(vectorizer, path)


def load_vectorizer(path="models/tfidf.pkl"):
    """
    Load saved TF-IDF model
    """

    return joblib.load(path)