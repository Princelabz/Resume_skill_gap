"""
tests/test_matcher.py: Unit tests for skill matching and gap calculation.
"""

import unittest
from src.matcher import SkillMatcher


class TestSkillMatcher(unittest.TestCase):

    def setUp(self):
        self.matcher = SkillMatcher()

    def test_full_match(self):
        resume_skills = ["python", "sql", "machine learning"]
        job_skills = ["python", "sql", "machine learning"]
        res = self.matcher.match_skills(resume_skills, job_skills)

        self.assertEqual(res["match_percentage"], 100.0)
        self.assertEqual(len(res["matched_skills"]), 3)
        self.assertEqual(len(res["missing_skills"]), 0)
        self.assertEqual(len(res["extra_skills"]), 0)

    def test_partial_match(self):
        resume_skills = ["python", "sql", "docker"]
        job_skills = ["python", "sql", "machine learning", "kubernetes"]
        res = self.matcher.match_skills(resume_skills, job_skills)

        # 2 matched out of 4 required = 50.0%
        self.assertEqual(res["match_percentage"], 50.0)
        self.assertCountEqual(res["matched_skills"], ["python", "sql"])
        self.assertCountEqual(res["missing_skills"], ["kubernetes", "machine learning"])
        self.assertCountEqual(res["extra_skills"], ["docker"])

    def test_zero_match(self):
        resume_skills = ["html", "css"]
        job_skills = ["python", "machine learning"]
        res = self.matcher.match_skills(resume_skills, job_skills)

        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(len(res["matched_skills"]), 0)
        self.assertEqual(len(res["missing_skills"]), 2)
        self.assertEqual(len(res["extra_skills"]), 2)

    def test_empty_job_skills(self):
        resume_skills = ["python"]
        job_skills = []
        res = self.matcher.match_skills(resume_skills, job_skills)

        self.assertEqual(res["match_percentage"], 0.0)
        self.assertEqual(res["total_required_skills"], 0)


if __name__ == "__main__":
    unittest.main()
