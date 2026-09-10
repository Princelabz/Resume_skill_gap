"""
src/recommendations.py: Demand-Aware Recommendation System for Missing Skills.

Prioritizes learning recommendations using authentic O*NET occupational attributes:
- 'Hot Technology' flags
- 'In Demand' ratings
- Occupational prevalence and importance values
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
from src.config import SKILLS_DB_PATH, PRIORITY_LEVELS


class RecommendationEngine:
    """
    Produces structured, explainable learning recommendations for missing skills.
    """

    def __init__(self, skills_db_path: Union[str, Path] = SKILLS_DB_PATH):
        self.skills_db_path = Path(skills_db_path) if skills_db_path else None
        self.skills_db: Dict[str, dict] = {}
        self._load_skills_db()

    def _load_skills_db(self) -> None:
        """Load O*NET skills database from JSON file."""
        if self.skills_db_path and self.skills_db_path.exists():
            try:
                with open(self.skills_db_path, "r", encoding="utf-8") as f:
                    self.skills_db = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed loading skills DB in RecommendationEngine: {e}")

    def prioritize_missing_skills(self, missing_skills: List[str]) -> Dict[str, Any]:
        """
        Categorize and prioritize missing skills into High, Medium, and Low Priority.

        Parameters:
            missing_skills: List of normalized missing skills.

        Returns:
            Dict containing:
            - high_priority: List of skill recommendation objects
            - medium_priority: List of skill recommendation objects
            - low_priority: List of skill recommendation objects
            - summary: Human-readable summary
        """
        high_priority = []
        medium_priority = []
        low_priority = []

        for skill_key in missing_skills:
            norm_key = skill_key.lower().strip()
            meta = self.skills_db.get(norm_key, {})

            display_name = meta.get("name", skill_key.title())
            category = meta.get("category", "General Skill")
            hot_tech = meta.get("hot_technology", False)
            in_demand = meta.get("in_demand", False)
            occ_count = meta.get("occupations_count", 1)
            importance = meta.get("importance_score", 0.0)

            # Assign priority based strictly on authentic O*NET data
            if hot_tech or in_demand or importance >= 4.0:
                priority = PRIORITY_LEVELS["HIGH"]
                if hot_tech and in_demand:
                    reason = "Flagged as an O*NET Hot Technology and In-Demand skill across industries."
                elif hot_tech:
                    reason = "Classified as an O*NET Hot Technology frequently required in recent job postings."
                elif in_demand:
                    reason = "Marked as high-growth and In-Demand in the O*NET database."
                else:
                    reason = f"Essential competency with high O*NET importance rating ({importance}/5.0)."
                
                high_priority.append({
                    "skill": display_name,
                    "skill_key": norm_key,
                    "category": category,
                    "priority": priority,
                    "reason": reason,
                    "hot_technology": hot_tech,
                    "in_demand": in_demand,
                    "importance_score": importance,
                })

            elif occ_count >= 3 or importance >= 3.0:
                priority = PRIORITY_LEVELS["MEDIUM"]
                reason = f"Standard technical requirement associated with multiple occupations in O*NET (frequency: {occ_count})."
                medium_priority.append({
                    "skill": display_name,
                    "skill_key": norm_key,
                    "category": category,
                    "priority": priority,
                    "reason": reason,
                    "hot_technology": hot_tech,
                    "in_demand": in_demand,
                    "importance_score": importance,
                })

            else:
                priority = PRIORITY_LEVELS["LOW"]
                reason = "Role-specific or specialized technical skill."
                low_priority.append({
                    "skill": display_name,
                    "skill_key": norm_key,
                    "category": category,
                    "priority": priority,
                    "reason": reason,
                    "hot_technology": hot_tech,
                    "in_demand": in_demand,
                    "importance_score": importance,
                })

        return {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "total_recommended": len(missing_skills),
            "high_count": len(high_priority),
            "medium_count": len(medium_priority),
            "low_count": len(low_priority),
        }
