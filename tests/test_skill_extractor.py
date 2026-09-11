"""
tests/test_skill_extractor.py: Unit tests for multi-word phrase matching and alias normalization.
"""

import unittest
from src.skill_normalizer import SkillNormalizer
from src.skill_extractor import SkillExtractor


class TestSkillExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.normalizer = SkillNormalizer()
        cls.extractor = SkillExtractor(normalizer=cls.normalizer)

    def test_multi_word_skill_extraction(self):
        text = """
        Candidate has strong expertise in Machine Learning, Deep Learning,
        Data Analysis, Power BI, Microsoft Excel, Natural Language Processing,
        Computer Vision, and Amazon Web Services.
        """
        extracted = self.extractor.extract_skills(text)

        expected_skills = [
            "machine learning",
            "deep learning",
            "data analysis",
            "power bi",
            "microsoft excel",
            "natural language processing",
            "computer vision",
            "aws",
        ]

        for skill in expected_skills:
            self.assertIn(skill, extracted, f"Expected skill '{skill}' was not extracted.")

    def test_alias_normalization(self):
        text = "Experience with ML, DL, NLP, JS, MS Excel, AWS, and PowerBI."
        extracted = self.extractor.extract_skills(text)

        self.assertIn("machine learning", extracted)
        self.assertIn("deep learning", extracted)
        self.assertIn("natural language processing", extracted)
        self.assertIn("javascript", extracted)
        self.assertIn("microsoft excel", extracted)
        self.assertIn("aws", extracted)
        self.assertIn("power bi", extracted)

    def test_programming_special_patterns(self):
        text = "Proficient in C++, C#, .NET, Python, and Node.js."
        extracted = self.extractor.extract_skills(text)

        self.assertIn("c++", extracted)
        self.assertIn("c#", extracted)
        self.assertIn(".net", extracted)
        self.assertIn("python", extracted)
        self.assertIn("node.js", extracted)


if __name__ == "__main__":
    unittest.main()
