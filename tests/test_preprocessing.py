"""
tests/test_preprocessing.py: Unit tests for text cleaning, tokenization, and PDF handling.
"""

import unittest
import io
from src.preprocessing import (
    clean_text,
    tokenize_words,
    tokenize_sentences,
    lemmatize_text,
    extract_text_from_pdf,
)


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_quotes_and_whitespace(self):
        raw = "Smart \u2018single\u2019 and \u201cdouble\u201d quotes \u2022 bullet \n\n\n\n extra lines"
        cleaned = clean_text(raw)
        self.assertIn("'single'", cleaned)
        self.assertIn('"double"', cleaned)
        self.assertIn("- bullet", cleaned)
        self.assertNotIn("\n\n\n\n", cleaned)

    def test_tokenize_words(self):
        text = "Data Scientists analyze complex datasets using Python and SQL."
        tokens = tokenize_words(text, remove_stop_words=False)
        self.assertIn("python", tokens)
        self.assertIn("sql", tokens)
        self.assertIn("scientists", tokens)

    def test_tokenize_words_with_stopwords(self):
        text = "This is an example of natural language processing."
        tokens = tokenize_words(text, remove_stop_words=True)
        self.assertNotIn("is", tokens)
        self.assertNotIn("an", tokens)
        self.assertIn("natural", tokens)
        self.assertIn("language", tokens)

    def test_tokenize_sentences(self):
        text = "Machine learning is powerful. NLP is explainable! Do you like data?"
        sentences = tokenize_sentences(text)
        self.assertEqual(len(sentences), 3)

    def test_lemmatize_text(self):
        text = "Running predictive models on multiple clusters"
        lemmatized = lemmatize_text(text)
        self.assertIn("model", lemmatized)
        self.assertIn("cluster", lemmatized)

    def test_empty_pdf_extraction(self):
        # Empty stream should return empty string with descriptive error message
        empty_stream = io.BytesIO(b"")
        text, msg = extract_text_from_pdf(empty_stream)
        self.assertEqual(text, "")
        self.assertTrue("Error" in msg or "0 pages" in msg)


if __name__ == "__main__":
    unittest.main()
