"""
src/skill_extractor.py: Explainable Skill Extraction using NLP Phrase Matching,
Known Skill Dictionaries, Regex, and spaCy / NLTK Preprocessing.

This extractor recognizes both single-word and multi-word skills (e.g.
'Machine Learning', 'Natural Language Processing', 'Microsoft Excel',
'Power BI', 'SQL Server', 'Amazon Web Services', 'Computer Vision')
without chopping them into isolated words.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
import spacy
from spacy.matcher import PhraseMatcher

from src.config import SKILLS_DB_PATH
from src.preprocessing import clean_text
from src.skill_normalizer import SkillNormalizer


class SkillExtractor:
    """
    Explainable NLP skill extractor using:
    - O*NET canonical skill taxonomy
    - Longest-phrase-first matching (multi-word entity protection)
    - Boundary-aware Regex patterns for programming symbols (C++, C#, .NET, etc.)
    - spaCy PhraseMatcher for efficient case-insensitive token-level lookup
    """

    def __init__(
        self,
        skills_db_path: Union[str, Path] = SKILLS_DB_PATH,
        normalizer: Optional[SkillNormalizer] = None,
    ):
        self.skills_db_path = Path(skills_db_path) if skills_db_path else None
        self.normalizer = normalizer or SkillNormalizer()
        self.skills_db: Dict[str, dict] = {}
        self.known_skills: Set[str] = set()

        # Load NLP model (spaCy)
        try:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load(disable=["ner"])
        except Exception:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                # Fallback blank English model
                self.nlp = spacy.blank("en")

        # Load skill dictionary
        self._load_skills_db()

        # Initialize phrase matcher
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._init_phrase_matcher()

        # Special symbol regexes (for C++, C#, .NET, Node.js, etc.)
        self._init_special_patterns()

    def _load_skills_db(self) -> None:
        """Load skills database from JSON file if it exists."""
        if self.skills_db_path and self.skills_db_path.exists():
            try:
                with open(self.skills_db_path, "r", encoding="utf-8") as f:
                    self.skills_db = json.load(f)
                    self.known_skills = set(self.skills_db.keys())
            except Exception as e:
                print(f"[Warning] Failed loading skills DB from {self.skills_db_path}: {e}")

        # If DB is empty, provide essential fallback skills
        if not self.known_skills:
            fallback = [
                "python", "java", "c++", "c#", "javascript", "typescript", "r", "sql",
                "html", "css", "machine learning", "deep learning", "natural language processing",
                "computer vision", "data analysis", "data science", "microsoft excel",
                "power bi", "tableau", "docker", "kubernetes", "git", "github",
                "amazon web services", "google cloud platform", "microsoft azure",
                "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "react", "node.js"
            ]
            for s in fallback:
                self.known_skills.add(s.lower())
                self.skills_db[s.lower()] = {
                    "name": s.title(),
                    "category": "Software & Technology",
                    "hot_technology": True,
                    "in_demand": True,
                }

    def _init_phrase_matcher(self) -> None:
        """Populate spaCy PhraseMatcher with known multi-word and single-word skills."""
        # Add canonical skills + alias phrases into the matcher
        all_phrases = set(self.known_skills)
        all_phrases.update(self.normalizer.alias_map.keys())

        # Sort by length descending so multi-word phrases take precedence
        sorted_phrases = sorted(all_phrases, key=lambda x: len(x.split()), reverse=True)

        patterns = []
        for phrase in sorted_phrases:
            # Avoid single short character patterns in general PhraseMatcher (handle in regex)
            if len(phrase.strip()) < 2:
                continue
            doc = self.nlp.make_doc(phrase)
            if len(doc) > 0:
                patterns.append(doc)

        if patterns:
            # Batch add patterns
            self.phrase_matcher.add("SKILL_PATTERN", patterns)

    def _init_special_patterns(self) -> None:
        """Regex patterns for tricky programming terms with symbols."""
        self.special_terms = {
            r"(?i)(?<![\w#])c\+\+(?![\w+])": "c++",
            r"(?i)(?<![\w#])c#(?![\w#])": "c#",
            r"(?i)(?<!\w)\.net\b": ".net",
            r"(?i)(?<!\w)node\.js\b": "node.js",
            r"(?i)(?<!\w)react\.js\b": "react",
            r"(?i)(?<!\w)vue\.js\b": "vue.js",
            r"(?i)\br\b(?=\s*(?:programming|language|script|code|studio|\,|$))": "r",
            r"(?i)\bc\b(?=\s*(?:programming|language|compiler|\,|$))": "c",
        }

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract normalized unique skills from input text.

        Parameters:
            text: Raw input text from resume or job description.

        Returns:
            List[str]: List of canonical, normalized skill names found in text.
        """
        if not text or not isinstance(text, str):
            return []

        cleaned = clean_text(text)
        if not cleaned:
            return []

        extracted_raw_skills: Set[str] = set()

        # Step 1: Special regex matches (C++, C#, .NET, etc.)
        for pattern, canonical_name in self.special_terms.items():
            if re.search(pattern, cleaned, re.IGNORECASE):
                extracted_raw_skills.add(canonical_name)

        # Step 2: spaCy PhraseMatcher for multi-word phrases & standard tokens
        doc = self.nlp(cleaned)
        matches = self.phrase_matcher(doc)

        # To resolve overlapping matches (e.g., "Natural Language Processing" vs "Language Processing")
        # we prioritize longer spans
        spans = [doc[start:end] for _, start, end in matches]
        # Filter overlapping spans: keep the longest span
        filtered_spans = spacy.util.filter_spans(spans)

        for span in filtered_spans:
            skill_text = span.text.strip().lower()
            extracted_raw_skills.add(skill_text)

        # Step 3: Exact Word Boundary Regex fallback for high-value aliases & skills
        cleaned_lower = cleaned.lower()
        for alias_key in self.normalizer.alias_map.keys():
            # Check standalone occurrences for short acronyms like 'ml', 'nlp', 'aws', 'js'
            if len(alias_key) <= 4 and re.search(r"\b" + re.escape(alias_key) + r"\b", cleaned_lower):
                extracted_raw_skills.add(alias_key)

        # Step 4: Normalize and deduplicate all extracted skills
        normalized_skills = self.normalizer.normalize_skill_list(list(extracted_raw_skills))

        # Return sorted by alphabetical order for clean display
        return sorted(normalized_skills)

    def extract_skills_with_details(self, text: str) -> List[dict]:
        """
        Extract skills along with their O*NET metadata (category, demand status, display name).
        """
        skill_names = self.extract_skills(text)
        detailed_skills = []

        for skill_key in skill_names:
            meta = self.skills_db.get(skill_key, {})
            display_name = meta.get("name", skill_key.title())
            category = meta.get("category", "Software & Technology")
            hot_tech = meta.get("hot_technology", False)
            in_demand = meta.get("in_demand", False)
            importance = meta.get("importance_score", 0.0)

            detailed_skills.append({
                "skill_key": skill_key,
                "display_name": display_name,
                "category": category,
                "hot_technology": hot_tech,
                "in_demand": in_demand,
                "importance_score": importance,
            })

        return detailed_skills
