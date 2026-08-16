import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf():
    """
    Create the TF-IDF vectorizer used by the SVM model.
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )

    return vectorizer


def fit_transform(vectorizer, train_text):
    """
    Learn the vocabulary from training text
    and transform the training text.
    """

    return vectorizer.fit_transform(train_text)


def transform(vectorizer, text):
    """
    Transform new text using the trained TF-IDF vectorizer.
    """

    return vectorizer.transform(text)


def save_vectorizer(
    vectorizer,
    path="models/cit-24-01-0086/tfidf.pkl"
):
    """
    Save the trained TF-IDF vectorizer.
    """

    joblib.dump(vectorizer, path)


def load_vectorizer(
    path="models/cit-24-01-0086/tfidf.pkl"
):
    """
    Load the saved TF-IDF vectorizer.
    """

    return joblib.load(path)