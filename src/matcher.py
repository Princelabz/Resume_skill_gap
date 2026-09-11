"""
src/matcher.py: Skill Matching and Set-Theoretic Gap Analysis.

Compares normalized candidate resume skills against job description skill requirements.
"""

from typing import Dict, List, Set, Any, Optional
from rapidfuzz import fuzz
from src.skill_normalizer import SkillNormalizer


class SkillMatcher:
    """
    Computes explainable set-theoretic skill matching, conservative RapidFuzz
    resolution for minor variations, and detailed section-evidence tracking.
    """

    def __init__(self, normalizer: Optional[SkillNormalizer] = None):
        self.normalizer = normalizer or SkillNormalizer()

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
        resume_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        job_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        fuzzy_threshold: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Perform explainable skill matching with RapidFuzz fallback and evidence tracking.

        Parameters:
            resume_skills: List of normalized skills found in resume.
            job_skills: List of normalized required skills found in job description.
            resume_evidence: Optional section evidence mapping from SkillExtractor.
            job_evidence: Optional section evidence mapping for job description.
            fuzzy_threshold: Minimum RapidFuzz similarity ratio (default 90.0) for matching.

        Returns:
            Dict containing match metrics, sets, and a structured evidence table.
        """
        resume_ev = resume_evidence or {}
        job_ev = job_evidence or {}

        # Canonical lowercase keys
        r_list = [s.lower().strip() for s in resume_skills if s]
        j_list = [s.lower().strip() for s in job_skills if s]

        # Preserve order while deduplicating
        r_set: Set[str] = set(r_list)
        j_unique = []
        for j in j_list:
            if j not in j_unique:
                j_unique.append(j)

        matched_skills: Set[str] = set()
        fuzzy_matches: Dict[str, tuple] = {}  # req_skill -> (matched_res_skill, score)
        unmatched_resume: List[str] = [r for r in r_list if r not in j_unique]

        # Step 1: Exact / Canonical Normalized Match (Highest Priority)
        for req in j_unique:
            if req in r_set:
                matched_skills.add(req)

        # Step 2: Conservative RapidFuzz for remaining unmatched required skills
        unmatched_required = [req for req in j_unique if req not in matched_skills]
        for req in unmatched_required:
            best_res = None
            best_score = 0.0
            for res in unmatched_resume:
                # Conservative similarity check
                sim = max(
                    fuzz.ratio(req, res),
                    fuzz.token_sort_ratio(req, res),
                )
                if sim >= fuzzy_threshold and sim > best_score:
                    best_score = sim
                    best_res = res

            if best_res is not None:
                fuzzy_matches[req] = (best_res, best_score)
                matched_skills.add(req)
                if best_res in unmatched_resume:
                    unmatched_resume.remove(best_res)

        # Step 3: Missing skills
        missing_skills = [req for req in j_unique if req not in matched_skills]

        # Step 4: Extra candidate skills beyond job requirements
        extra_skills = sorted(list(set(unmatched_resume)))

        total_req = len(j_unique)
        total_res = len(r_set)
        matched_count = len(matched_skills)

        if total_req > 0:
            match_pct = round((matched_count / total_req) * 100.0, 2)
        else:
            match_pct = 0.0

        # Step 5: Build Comprehensive Evidence Table (Skill | Status | Evidence)
        evidence_table = []
        skill_evidence_map = {}

        # 1. Required Skills (Matched or Missing)
        for req in j_unique:
            display_name = self.normalizer.format_skill_display(req)
            if req in fuzzy_matches:
                res_match, score = fuzzy_matches[req]
                ev_str = resume_ev.get(res_match, {}).get("evidence", "Detected in resume")
                status = "✓ Matched"
                evidence = f"{ev_str} (fuzzy match {score:.0f}%)"
            elif req in matched_skills:
                ev_str = resume_ev.get(req, {}).get("evidence", "Detected in resume")
                status = "✓ Matched"
                evidence = ev_str if ev_str else "Detected in resume"
            else:
                status = "✗ Missing"
                evidence = "not detected"

            evidence_table.append({
                "Skill": display_name,
                "Status": status,
                "Evidence": evidence,
                "Type": "Required",
            })
            skill_evidence_map[req] = {
                "status": "matched" if "Matched" in status else "missing",
                "evidence": evidence,
                "display_name": display_name,
            }

        # 2. Extra Skills (Candidate skills not required by job)
        for extra in extra_skills:
            display_name = self.normalizer.format_skill_display(extra)
            ev_str = resume_ev.get(extra, {}).get("evidence", "Skills section")
            evidence_table.append({
                "Skill": display_name,
                "Status": "+ Extra",
                "Evidence": ev_str if ev_str else "Skills section",
                "Type": "Additional",
            })
            skill_evidence_map[extra] = {
                "status": "extra",
                "evidence": ev_str if ev_str else "Skills section",
                "display_name": display_name,
            }

        return {
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(missing_skills),
            "extra_skills": extra_skills,
            "total_matched": matched_count,
            "total_missing": len(missing_skills),
            "total_extra": len(extra_skills),
            "total_required_skills": total_req,
            "total_resume_skills": total_res,
            "match_percentage": match_pct,
            "evidence_table": evidence_table,
            "skill_evidence_map": skill_evidence_map,
        }
