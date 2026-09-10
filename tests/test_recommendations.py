"""
tests/test_recommendations.py: Unit tests for missing skill priority categorization.
"""

import unittest
from src.recommendations import RecommendationEngine


class TestRecommendations(unittest.TestCase):

    def setUp(self):
        self.engine = RecommendationEngine()

    def test_priority_categorization(self):
        missing = ["python", "docker", "machine learning"]
        res = self.engine.prioritize_missing_skills(missing)

        self.assertEqual(res["total_recommended"], 3)
        self.assertIn("high_priority", res)
        self.assertIn("medium_priority", res)
        self.assertIn("low_priority", res)

        # Total counts across levels must equal total missing skills
        total = res["high_count"] + res["medium_count"] + res["low_count"]
        self.assertEqual(total, 3)

        # Check structure of item
        if res["high_priority"]:
            item = res["high_priority"][0]
            self.assertIn("skill", item)
            self.assertIn("reason", item)
            self.assertIn("priority", item)


if __name__ == "__main__":
    unittest.main()
