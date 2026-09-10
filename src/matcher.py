"""
src/matcher.py: Skill Matching and Set-Theoretic Gap Analysis.

Compares normalized candidate resume skills against job description skill requirements.
"""

from typing import Dict, List, Set, Any


class SkillMatcher:
    """
    Computes exact set-theoretic matching, gap identification,
    and match percentages between resume skills and job requirements.
    """

    def match_skills(
        self,
        resume_skills: List[str],
        job_skills: List[str],
    ) -> Dict[str, Any]:
        """
        Perform deterministic skill matching and gap comparison.

        Parameters:
            resume_skills: List of normalized skills extracted from resume.
            job_skills: List of normalized skills extracted from job description.

        Returns:
            Dict containing:
            - matched_skills: Skills present in both resume and job.
            - missing_skills: Skills required by job but absent in resume.
            - extra_skills: Skills candidate possesses beyond job requirements.
            - total_required_skills: Count of required skills in job description.
            - total_resume_skills: Count of skills found in resume.
            - match_percentage: (matched / required) * 100.
        """
        r_set: Set[str] = set(s.lower() for s in resume_skills if s)
        j_set: Set[str] = set(s.lower() for s in job_skills if s)

        matched_set = r_set.intersection(j_set)
        missing_set = j_set.difference(r_set)
        extra_set = r_set.difference(j_set)

        total_req = len(j_set)
        total_res = len(r_set)
        matched_count = len(matched_set)

        if total_req > 0:
            match_pct = round((matched_count / total_req) * 100.0, 2)
        else:
            match_pct = 0.0

        return {
            "matched_skills": sorted(list(matched_set)),
            "missing_skills": sorted(list(missing_set)),
            "extra_skills": sorted(list(extra_set)),
            "total_matched": matched_count,
            "total_missing": len(missing_set),
            "total_extra": len(extra_set),
            "total_required_skills": total_req,
            "total_resume_skills": total_res,
            "match_percentage": match_pct,
        }
