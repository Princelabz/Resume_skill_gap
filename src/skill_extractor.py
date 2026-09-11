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
from src.skill_taxonomy import (
    is_valid_skill,
    check_context_for_false_positive,
    detect_sections,
    get_char_section,
    SINGLE_TOKEN_BLOCKLIST,
)


class SkillExtractor:
    """
    Explainable NLP skill extractor using:
    - Curated Skill Taxonomy & O*NET canonical knowledge base
    - Longest-phrase-first matching (multi-word entity protection via spaCy PhraseMatcher)
    - Boundary-aware Regex patterns for programming symbols (C++, C#, .NET, etc.)
    - Resume / Job Description section detection for evidence tracking
    - Context validation to reject false positives (e.g., social links, generic verbs)
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
                "aws", "amazon web services", "google cloud platform", "microsoft azure",
                "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "react", "node.js",
                "data visualization", "statistics", "business intelligence"
            ]
            for s in fallback:
                self.known_skills.add(s.lower())
                self.skills_db[s.lower()] = {
                    "name": self.normalizer.format_skill_display(s),
                    "category": "Software & Technology",
                    "hot_technology": True,
                    "in_demand": True,
                }

    def _init_phrase_matcher(self) -> None:
        """Populate spaCy PhraseMatcher with valid multi-word and single-word skills."""
        all_phrases = set(self.known_skills)
        all_phrases.update(self.normalizer.alias_map.keys())

        # Sort by length descending so multi-word phrases take precedence
        sorted_phrases = sorted(all_phrases, key=lambda x: len(x.split()), reverse=True)

        patterns = []
        for phrase in sorted_phrases:
            phrase_str = phrase.strip().lower()
            if len(phrase_str) < 2:
                continue
            # Block invalid terms from ever entering PhraseMatcher
            if phrase_str in SINGLE_TOKEN_BLOCKLIST or not is_valid_skill(phrase_str):
                continue
            doc = self.nlp.make_doc(phrase)
            if len(doc) > 0:
                patterns.append(doc)

        if patterns:
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

    def extract_skills_with_evidence(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract normalized skills with section evidence and metadata.

        Returns:
            Dict[str, dict]: Mapping of canonical skill key ->
                {
                    'skill': canonical_key,
                    'display_name': formatted_name,
                    'sections': list of section names where detected,
                    'evidence': formatted evidence string (e.g. 'Skills section + Experience section')
                }
        """
        if not text or not isinstance(text, str):
            return {}

        cleaned = clean_text(text)
        if not cleaned:
            return {}

        # Detect resume / JD sections for evidence tracking
        section_ranges = detect_sections(cleaned)
        evidence_tracker: Dict[str, Set[str]] = {}

        # Helper to record evidence
        def add_evidence(raw_term: str, char_pos: int):
            if check_context_for_false_positive(raw_term, cleaned, char_pos):
                return
            if not is_valid_skill(raw_term, cleaned, char_pos):
                return

            normalized = self.normalizer.normalize_skill(raw_term)
            if not normalized or not is_valid_skill(normalized):
                return

            section = get_char_section(char_pos, section_ranges)
            if normalized not in evidence_tracker:
                evidence_tracker[normalized] = set()
            evidence_tracker[normalized].add(section)

        # Step 1: Special regex matches (C++, C#, .NET, etc.)
        for pattern, canonical_name in self.special_terms.items():
            for m in re.finditer(pattern, cleaned):
                add_evidence(canonical_name, m.start())

        # Step 2: spaCy PhraseMatcher for multi-word phrases & standard tokens
        doc = self.nlp(cleaned)
        matches = self.phrase_matcher(doc)

        spans = [doc[start:end] for _, start, end in matches]
        filtered_spans = spacy.util.filter_spans(spans)

        for span in filtered_spans:
            skill_text = span.text.strip().lower()
            # Calculate character start offset in cleaned text
            char_pos = span.start_char
            add_evidence(skill_text, char_pos)

        # Step 3: High-confidence boundary-aware acronym check
        # High value short acronyms (aws, sql, nlp, ml, dl, js, ts, r, bi)
        cleaned_lower = cleaned.lower()
        acronym_candidates = ["aws", "sql", "ml", "dl", "nlp", "js", "ts", "r", "bi"]
        for acr in acronym_candidates:
            if acr in self.normalizer.alias_map:
                for m in re.finditer(r"\b" + re.escape(acr) + r"\b", cleaned_lower):
                    add_evidence(acr, m.start())

        # Order of precedence for section display
        section_order = [
            "Skills section",
            "Experience section",
            "Project section",
            "Education section",
            "Certifications section",
            "Summary section",
            "General section",
        ]

        # Format results
        results = {}
        for skill_key, sec_set in evidence_tracker.items():
            sorted_secs = sorted(list(sec_set), key=lambda s: section_order.index(s) if s in section_order else 99)
            evidence_str = " + ".join(sorted_secs) if sorted_secs else "General section"
            display_name = self.normalizer.format_skill_display(skill_key)

            results[skill_key] = {
                "skill": skill_key,
                "display_name": display_name,
                "sections": sorted_secs,
                "evidence": evidence_str,
            }

        return results

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract normalized unique skills from input text.

        Returns:
            List[str]: List of canonical, normalized skill names found in text.
        """
        evidence_dict = self.extract_skills_with_evidence(text)
        return sorted(list(evidence_dict.keys()))

    def extract_skills_with_details(self, text: str) -> List[dict]:
        """
        Extract skills along with their O*NET metadata and section evidence.
        """
        evidence_dict = self.extract_skills_with_evidence(text)
        detailed_skills = []

        for skill_key in sorted(evidence_dict.keys()):
            ev = evidence_dict[skill_key]
            meta = self.skills_db.get(skill_key, {})
            display_name = ev["display_name"]
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
                "evidence": ev["evidence"],
                "sections": ev["sections"],
            })

        return detailed_skills
