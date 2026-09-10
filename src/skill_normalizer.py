"""
src/skill_normalizer.py: Skill Normalization and Alias Resolution.

This module maps acronyms, abbreviations, punctuation variations, and synonyms
to standard canonical skill names (e.g., 'ML' -> 'machine learning',
'MS Excel' -> 'microsoft excel', 'power-bi' -> 'power bi').
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Union
from src.config import ALIAS_MAP_PATH, DEFAULT_ALIASES


class SkillNormalizer:
    """
    Normalizes extracted skill strings into canonical representations.
    """

    def __init__(self, alias_file: Union[str, Path] = ALIAS_MAP_PATH):
        self.alias_file = Path(alias_file) if alias_file else None
        self.alias_map: Dict[str, str] = dict(DEFAULT_ALIASES)
        self._load_alias_map()

    def _load_alias_map(self) -> None:
        """Load configured aliases from JSON if available."""
        if self.alias_file and self.alias_file.exists():
            try:
                with open(self.alias_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.alias_map.update({k.lower().strip(): v.lower().strip() for k, v in data.items()})
            except Exception as e:
                print(f"[Warning] Could not load alias map from {self.alias_file}: {e}")

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a single skill string:
        - Lowercase and whitespace strip.
        - Strip common trailing punctuation (e.g., 'Python,' -> 'python').
        - Check alias lookup dictionary.
        - Normalize hyphens and multiple spaces.
        """
        if not skill or not isinstance(skill, str):
            return ""

        s = skill.lower().strip()
        # Remove trailing and leading punctuation except +, # (to protect C++, C#)
        s = re.sub(r"^[^\w+#]+|[^\w+#]+$", "", s)

        # 1. Direct alias check
        if s in self.alias_map:
            return self.alias_map[s]

        # 2. Check hyphen to space replacement
        hyphen_to_space = s.replace("-", " ")
        if hyphen_to_space in self.alias_map:
            return self.alias_map[hyphen_to_space]

        # 3. Check dot / slash replacements (e.g. react.js -> react)
        dot_to_space = s.replace(".", " ").replace("/", " ")
        dot_to_space = " ".join(dot_to_space.split())
        if dot_to_space in self.alias_map:
            return self.alias_map[dot_to_space]

        return s

    def normalize_skill_list(self, skills: List[str]) -> List[str]:
        """
        Normalize and deduplicate an iterable of skills while preserving order.
        """
        seen: Set[str] = set()
        normalized_list: List[str] = []

        for skill in skills:
            norm = self.normalize_skill(skill)
            if norm and norm not in seen:
                seen.add(norm)
                normalized_list.append(norm)

        return normalized_list

    def add_alias(self, alias: str, canonical: str) -> None:
        """Add a custom alias mapping in-memory."""
        if alias and canonical:
            self.alias_map[alias.lower().strip()] = canonical.lower().strip()

    def save_aliases(self, output_path: Union[str, Path] = None) -> None:
        """Save current alias dictionary to JSON file."""
        target = Path(output_path) if output_path else self.alias_file
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.alias_map, f, indent=2)
