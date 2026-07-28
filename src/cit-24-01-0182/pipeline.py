"""
src/pipeline.py
Member 1 -- P. Chamika Janith Piyasena (CIT-24-01-0182)
Group 6 -- Spam Email/SMS Detection System

Reusable NLP preprocessing pipeline for Member 1's unique steps:
    Language Detection -> Sentence Segmentation -> Text Normalization
    -> Tokenization / Stop-word removal / Lemmatization

Import this module from any notebook OR from the group's final web
application so preprocessing is 100% consistent between training and
inference (avoids train/serve skew).

Usage:
    from pipeline import full_pipeline, detect_language, segment_sentences

    clean_text = full_pipeline(raw_email_text)
"""

import re
import nltk
from langdetect import detect, LangDetectException
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# One-time NLTK downloads (quiet + safe to call repeatedly / from Colab)
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        # Some NLTK versions don't have punkt_tab / omw-1.4 -- safe to ignore
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Slang / abbreviation dictionary -- common patterns seen in spam & SMS text
# ---------------------------------------------------------------------------
SLANG = {
    r"\bu\b": "you", r"\bur\b": "your", r"\bgr8\b": "great",
    r"\bplz\b": "please", r"\blol\b": "laugh", r"\bbtw\b": "by the way",
    r"\bfree\b": "free", r"\bwin\b": "win", r"\bcash\b": "cash",
    r"\bpriz\b": "prize", r"\btxt\b": "text", r"\bmob\b": "mobile",
    r"\bwk\b": "week", r"\bwon\b": "won", r"\bcall\b": "call",
    r"\bclick\b": "click", r"\bclaim\b": "claim", r"\boffer\b": "offer",
    r"\burgent\b": "urgent", r"\blimited\b": "limited",
}


def detect_language(text: str) -> str:
    """Return the ISO-639-1 language code for `text`. Returns 'unknown' on
    failure (e.g. empty string, purely numeric content)."""
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def segment_sentences(text: str) -> list:
    """Split an email/message body into a list of sentences (NLTK punkt)."""
    return sent_tokenize(str(text))


def count_sentences(text: str) -> int:
    """Number of sentences in `text`."""
    return len(sent_tokenize(str(text)))


def avg_sentence_length(text: str) -> float:
    """Average number of words per sentence in `text`. Returns 0 for empty
    input to avoid division-by-zero."""
    sentences = sent_tokenize(str(text))
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def normalize_text(text: str) -> str:
    """Full text-normalization pass:
    1. Lowercase
    2. Expand common slang/abbreviations
    3. Strip URLs
    4. Strip email addresses
    5. Strip long digit runs (phone numbers)
    6. Strip non-alphabetic characters
    7. Collapse repeated whitespace
    """
    text = str(text).lower()
    for pattern, replacement in SLANG.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"\S+@\S+", " ", text)                # emails
    text = re.sub(r"\b\d{7,}\b", " ", text)              # long digit runs
    text = re.sub(r"[^a-z\s]", " ", text)                # special chars
    text = re.sub(r"\s+", " ", text).strip()             # whitespace
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, drop stop-words / non-alphabetic tokens, lemmatize, and
    rejoin into a single space-separated string ready for vectorization."""
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def full_pipeline(text: str) -> str:
    """Run the complete Member 1 preprocessing pipeline on a single raw
    email/message string and return the final, model-ready text.

    NOTE: this does not run language detection (that's a dataset-level
    filter, not a per-message step) -- run `detect_language` separately
    if you need to reject non-English input before calling this.
    """
    normalized = normalize_text(text)
    return tokenize_and_lemmatize(normalized)


if __name__ == "__main__":
    # Tiny smoke test -- run with: python src/pipeline.py
    sample = "FREE entry! Txt WIN to 80086 now, ur gr8 prize awaits!! Click http://spam.example.com"
    print("Language:", detect_language(sample))
    print("Sentences:", segment_sentences(sample))
    print("Normalized:", normalize_text(sample))
    print("Final:", full_pipeline(sample))
