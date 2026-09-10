"""
src/preprocessing.py: Text extraction, PDF reading, cleaning, tokenization, and lemmatization.

This module provides explainable classical NLP preprocessing:
1. Safe PDF text extraction using pypdf with OCR-fallback detection.
2. Text cleaning and whitespace normalization.
3. Sentence and word tokenization using NLTK and regex.
4. Stopword removal and lemmatization for textual similarity calculations.
"""

import re
import io
from pathlib import Path
from typing import Tuple, List, Union
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure necessary NLTK data is accessible
for corpus in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{corpus}" if corpus in ["stopwords", "wordnet"] else f"tokenizers/{corpus}")
    except LookupError:
        try:
            nltk.download(corpus, quiet=True)
        except Exception:
            pass

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "about",
        "against", "between", "into", "through", "during", "before", "after", "above", "below",
        "from", "up", "down", "of", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
    }

try:
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _LEMMATIZER = None


class PDFTextExtractionError(Exception):
    """Exception raised when a PDF cannot be read or contains no selectable text."""
    pass


def extract_text_from_pdf(file_source: Union[str, bytes, io.BytesIO, Path]) -> Tuple[str, str]:
    """
    Safely extract selectable text from a PDF file path or byte stream.

    Parameters:
        file_source: File path (str or Path), raw bytes (bytes), or BytesIO/file-like stream.

    Returns:
        Tuple[str, str]: (extracted_text, status_message)
        If no selectable text is found, status_message will explain that OCR is not enabled.
    """
    try:
        if isinstance(file_source, (str, Path)):
            stream = open(file_source, "rb")
        elif isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        elif isinstance(file_source, io.BytesIO):
            stream = file_source
        elif hasattr(file_source, "read"):
            # File-like object (such as Streamlit UploadedFile)
            stream = io.BytesIO(file_source.read())
        else:
            return "", "Unsupported file source type provided for PDF extraction."

        reader = PdfReader(stream)
        if len(reader.pages) == 0:
            return "", "The uploaded PDF document contains 0 pages."

        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages).strip()

        if not full_text:
            return "", (
                "No selectable text found in the PDF. The document may be a scanned image "
                "or flattened raster file. Note: OCR (Optical Character Recognition) is not "
                "implemented. Please upload a text-based PDF or copy/paste text directly."
            )

        return full_text, f"Successfully extracted text from {len(reader.pages)} page(s)."

    except Exception as e:
        return "", f"Error reading PDF file: {str(e)}"


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text:
    - Replaces smart quotes, hyphens, and bullets with standard ASCII.
    - Normalizes multiple spaces and empty lines.
    - Preserves case for subsequent POS/Entity matching, but removes non-printable chars.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace smart quotes and special typographical symbols
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": " - ",  # bullet point
        "\u2023": " - ",
        "\u25e6": " - ",
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\t": " ",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)

    # Remove non-printable characters except newlines
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Normalize multiple horizontal whitespaces
    text = re.sub(r"[ ]{2,}", " ", text)
    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK sent_tokenize."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    try:
        return sent_tokenize(cleaned)
    except Exception:
        # Fallback regex sentence splitter
        return [s.strip() for s in re.split(r"[.!?]+(?:\s+|\n)", cleaned) if s.strip()]


def tokenize_words(text: str, remove_stop_words: bool = False) -> List[str]:
    """
    Tokenize text into lowercased words.
    Optionally filter out standard English stopwords.
    """
    cleaned = clean_text(text).lower()
    # Retain alphanumeric tokens and hyphenated terms
    tokens = re.findall(r"\b[a-z0-9+#.-]+\b", cleaned)

    if remove_stop_words:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    return tokens


def lemmatize_text(text: str) -> str:
    """
    Lemmatize tokens in text using NLTK WordNetLemmatizer.
    Useful for TF-IDF textual similarity preprocessing.
    """
    words = tokenize_words(text, remove_stop_words=True)
    if _LEMMATIZER:
        try:
            lemmatized = [_LEMMATIZER.lemmatize(w) for w in words]
            return " ".join(lemmatized)
        except Exception:
            pass
    return " ".join(words)
