"""
preprocessing.py

Text preprocessing pipeline for the spam/phishing classification project
(Ctrl-Alt-Elite). Extracted from cit_24_01_0014_pipeline___lrmodel.ipynb.

Steps:
    1. Language detection (keep English-only rows)
    2. Regex cleaning (lowercase, strip HTML/URLs/emails/numbers/punctuation)
    3. POS-aware lemmatization

Usage as a library (e.g. in app.py before vectorizing user input):
    from preprocessing import preprocess_text
    cleaned = preprocess_text(raw_string)

Usage as a script (reproduces the notebook's dataset cleaning):
    python preprocessing.py input.csv output.csv --text-col text --label-col label
"""

import argparse
import re

import nltk
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

DetectorFactory.seed = 0
_lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------
# Step 1: Language detection
# ---------------------------------------------------
def lang_detect(text: str) -> str:
    """Detect the language of a string. Returns 'Unknown' if undetectable."""
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"


# ---------------------------------------------------
# Step 2: Regex cleaning
# ---------------------------------------------------
def regex_clean(text: str) -> str:
    """Lowercase and strip HTML, URLs, emails, numbers, and punctuation."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\d+", " ", text)                        # numbers
    text = re.sub(r"[^\w\s]", " ", text)                    # punctuation
    text = re.sub(r"\s+", " ", text).strip()                # collapse spaces

    return text


# ---------------------------------------------------
# Step 3: POS-aware lemmatization
# ---------------------------------------------------
def _get_wordnet_pos_from_tag(tag: str):
    if not tag:
        return wordnet.NOUN
    first_letter = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(first_letter, wordnet.NOUN)


def lemmatize_text(text: str) -> str:
    """Lemmatize each word in a cleaned string using its POS tag."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    word_tags = nltk.pos_tag(words)
    lemmatized_words = [
        _lemmatizer.lemmatize(word, _get_wordnet_pos_from_tag(tag))
        for word, tag in word_tags
    ]
    return " ".join(lemmatized_words)


# ---------------------------------------------------
# Combined single-string pipeline (for inference, e.g. Flask app)
# ---------------------------------------------------
def preprocess_text(text: str) -> str:
    """Run the full pipeline (clean + lemmatize) on a single raw string.

    Note: language filtering is a dataset-level step (drops non-English
    rows) and is not applied here, since a single input string should
    still be classified rather than silently dropped.
    """
    cleaned = regex_clean(text)
    return lemmatize_text(cleaned)


# ---------------------------------------------------
# Dataframe-level pipeline (for reproducing dataset cleaning)
# ---------------------------------------------------
def preprocess_dataframe(
    df: pd.DataFrame, text_col: str = "text", filter_english: bool = True
) -> pd.DataFrame:
    """Apply language filtering, regex cleaning, and lemmatization to a df.

    Adds 'cleaned_text' and 'cleaned_lemmatized' columns. If filter_english
    is True, also adds a 'language' column and drops non-English rows.
    """
    df = df.copy()

    if filter_english:
        df["language"] = df[text_col].apply(lang_detect)
        df = df[df["language"] == "en"].copy()

    df["cleaned_text"] = df[text_col].apply(regex_clean)
    df["cleaned_lemmatized"] = df["cleaned_text"].apply(lemmatize_text)

    return df


def main():
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline on a CSV.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to write the cleaned CSV file")
    parser.add_argument("--text-col", default="text", help="Name of the raw text column")
    parser.add_argument("--label-col", default="label", help="Name of the label column")
    parser.add_argument(
        "--no-lang-filter", action="store_true",
        help="Skip English-only language filtering",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = preprocess_dataframe(df, text_col=args.text_col, filter_english=not args.no_lang_filter)

    out_cols = ["cleaned_lemmatized"]
    if args.label_col in df.columns:
        out_cols.append(args.label_col)

    df.to_csv(args.output_csv, columns=out_cols, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
