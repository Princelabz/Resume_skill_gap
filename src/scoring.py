"""
src/scoring.py: Skill Gap Level Determination, TF-IDF Vectorization,
and Cosine Similarity Calculation.

This module provides two distinct, explainable metrics:
1. Skill Match Percentage & Gap Level (Deterministic rule-based skill overlap)
2. Textual Cosine Similarity (TF-IDF vector space overlap)
"""

import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import GAP_THRESHOLDS
from src.preprocessing import clean_text, lemmatize_text


class ScoringEngine:
    """
    Computes explainable gap levels and textual TF-IDF cosine similarity.
    """

    def __init__(self, thresholds: Dict[str, Tuple[float, float]] = None):
        self.thresholds = thresholds or GAP_THRESHOLDS

    def determine_gap_level(self, match_percentage: float) -> Tuple[str, str]:
        """
        Classify skill match percentage into an explainable gap level.

        Parameters:
            match_percentage: Percentage of required skills matched (0.0 to 100.0)

        Returns:
            Tuple[str, str]: (level_name, description)
        """
        score = max(0.0, min(100.0, float(match_percentage)))

        if score >= self.thresholds["Strong Match"][0]:
            return "Strong Match", "Candidate possesses almost all mandatory skills for this role."
        elif score >= self.thresholds["Good Match"][0]:
            return "Good Match", "Candidate meets majority of requirements with minor skill gaps."
        elif score >= self.thresholds["Moderate Gap"][0]:
            return "Moderate Gap", "Candidate meets foundational skills but is missing key core technologies."
        else:
            return "Large Gap", "Substantial gap between candidate profile and role requirements."

    def compute_tfidf_similarity(
        self,
        resume_text: str,
        job_text: str,
    ) -> Dict[str, Any]:
        """
        Calculate full-text TF-IDF vector representations and Cosine Similarity.

        Parameters:
            resume_text: Raw or cleaned text of the resume.
            job_text: Raw or cleaned text of the job description.

        Returns:
            Dict containing:
            - cosine_similarity_score: Cosine similarity as float (0.0 to 1.0)
            - text_similarity_percentage: Cosine similarity multiplied by 100 (0.0 to 100.0%)
            - top_shared_terms: Top TF-IDF overlapping n-grams between documents.
        """
        # Preprocess with lemmatization & stopword cleaning for fair textual comparison
        clean_res = lemmatize_text(clean_text(resume_text))
        clean_job = lemmatize_text(clean_text(job_text))

        if not clean_res.strip() or not clean_job.strip():
            return {
                "cosine_similarity_score": 0.0,
                "text_similarity_percentage": 0.0,
                "top_shared_terms": [],
                "explanation": "One or both documents contain insufficient textual content.",
            }

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([clean_res, clean_job])
            sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            sim_score = float(sim_matrix[0][0])
            sim_pct = round(sim_score * 100.0, 2)

            # Find top shared TF-IDF vocabulary terms
            feature_names = np.array(vectorizer.get_feature_names_out())
            res_vec = tfidf_matrix[0].toarray()[0]
            job_vec = tfidf_matrix[1].toarray()[0]

            # Element-wise product shows which terms contributed most to cosine similarity
            overlap_scores = res_vec * job_vec
            top_indices = np.argsort(overlap_scores)[::-1]

            top_shared_terms = []
            for idx in top_indices:
                if overlap_scores[idx] > 0:
                    top_shared_terms.append({
                        "term": str(feature_names[idx]),
                        "overlap_weight": round(float(overlap_scores[idx]), 4),
                    })
                if len(top_shared_terms) >= 10:
                    break

            return {
                "cosine_similarity_score": round(sim_score, 4),
                "text_similarity_percentage": max(0.0, min(100.0, sim_pct)),
                "top_shared_terms": top_shared_terms,
                "explanation": (
                    f"Calculated via TF-IDF (1-2 n-grams) and Cosine Similarity in vector space. "
                    f"Top shared terms: {', '.join(t['term'] for t in top_shared_terms[:5])}."
                    if top_shared_terms
                    else "No significant shared vocabulary detected."
                ),
            }

        except Exception as e:
            return {
                "cosine_similarity_score": 0.0,
                "text_similarity_percentage": 0.0,
                "top_shared_terms": [],
                "explanation": f"Error calculating TF-IDF similarity: {str(e)}",
            }
