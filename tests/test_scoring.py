"""
tests/test_scoring.py: Unit tests for scoring engine, thresholds, and TF-IDF similarity.
"""

import unittest
from src.scoring import ScoringEngine


class TestScoring(unittest.TestCase):

    def setUp(self):
        self.engine = ScoringEngine()

    def test_gap_level_thresholds(self):
        self.assertEqual(self.engine.determine_gap_level(95.0)[0], "Strong Match")
        self.assertEqual(self.engine.determine_gap_level(80.0)[0], "Strong Match")
        self.assertEqual(self.engine.determine_gap_level(75.0)[0], "Good Match")
        self.assertEqual(self.engine.determine_gap_level(60.0)[0], "Good Match")
        self.assertEqual(self.engine.determine_gap_level(50.0)[0], "Moderate Gap")
        self.assertEqual(self.engine.determine_gap_level(40.0)[0], "Moderate Gap")
        self.assertEqual(self.engine.determine_gap_level(30.0)[0], "Large Gap")
        self.assertEqual(self.engine.determine_gap_level(0.0)[0], "Large Gap")

    def test_tfidf_similarity(self):
        res_text = "Experienced Senior Python Developer with Django, FastAPI, Docker, and PostgreSQL experience."
        job_text = "Looking for a Python Developer proficient in Django, FastAPI, Docker, and PostgreSQL databases."

        res = self.engine.compute_tfidf_similarity(res_text, job_text)
        self.assertGreater(res["text_similarity_percentage"], 40.0)
        self.assertIn("top_shared_terms", res)
        self.assertTrue(len(res["top_shared_terms"]) > 0)

    def test_tfidf_empty_text(self):
        res = self.engine.compute_tfidf_similarity("", "Some job text")
        self.assertEqual(res["text_similarity_percentage"], 0.0)


if __name__ == "__main__":
    unittest.main()
