import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only first time)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean the input text by removing URLs, punctuation,
    numbers and special characters.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation & special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


def lowercase(text):
    """Convert text to lowercase."""
    return text.lower()


def tokenize(text):
    """Split text into words."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Remove common English stop words."""
    return [word for word in tokens if word not in stop_words]


def lemmatize(tokens):
    """Convert words to base form."""
    return [lemmatizer.lemmatize(word) for word in tokens]


def preprocess_text(text):
    """
    Complete preprocessing pipeline
    """

    text = clean_text(text)
    text = lowercase(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)