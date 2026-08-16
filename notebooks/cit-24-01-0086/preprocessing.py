import re
import nltk

from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def detect_language(text):
    """
    Detect the language of the input text.
    Returns 'en' for English and the detected language code otherwise.
    """

    if not isinstance(text, str) or not text.strip():
        return "unknown"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def clean_text(text):
    """
    Remove URLs, numbers, punctuation and special characters.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Keep English letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """Convert text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Remove common English stop words."""
    return [
        word for word in tokens
        if word not in stop_words
    ]


def lemmatize(tokens):
    """Convert words into their base form."""
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


def preprocess_text(text):
    """
    Complete text preprocessing pipeline.

    Steps:
    1. Clean text
    2. Lowercase
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize

    Returns:
        Cleaned preprocessed text.
    """

    text = clean_text(text)

    if not text.strip():
        return ""

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)