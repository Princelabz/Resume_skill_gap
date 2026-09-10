"""
Configuration settings, file paths, and thresholds for the Skill Gap Analyzer.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FALLBACK_DATA_DIR = BASE_DIR / "Dataset"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Required Raw Dataset Files
REQUIRED_RAW_FILES = [
    "software_skills.csv",
    "essential_skills.csv",
    "occupation_data.csv",
    "job_titles.csv",
]

# Processed Artifact Paths
SKILLS_DB_PATH = PROCESSED_DATA_DIR / "skills_db.json"
ALIAS_MAP_PATH = PROCESSED_DATA_DIR / "alias_map.json"

# Score Thresholds for Skill Gap Levels
GAP_THRESHOLDS = {
    "Strong Match": (80.0, 100.0),
    "Good Match": (60.0, 79.99),
    "Moderate Gap": (40.0, 59.99),
    "Large Gap": (0.0, 39.99),
}

# Demand Priority Weights based on O*NET attributes
PRIORITY_LEVELS = {
    "HIGH": "High Priority",
    "MEDIUM": "Medium Priority",
    "LOW": "Low Priority",
}

# Initial Default Skill Normalization Aliases
# Format: "alias/acronym/typo": "canonical normalized skill name"
DEFAULT_ALIASES = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "machinelearning": "machine learning",
    "dl": "deep learning",
    "deep-learning": "deep learning",
    "deeplearning": "deep learning",
    "nlp": "natural language processing",
    "natural-language-processing": "natural language processing",
    "cv": "computer vision",
    "computer-vision": "computer vision",
    "ai": "artificial intelligence",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ms excel": "microsoft excel",
    "excel": "microsoft excel",
    "ms-excel": "microsoft excel",
    "ms word": "microsoft word",
    "word": "microsoft word",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "power bi desktop": "power bi",
    "sql server": "microsoft sql server",
    "ms sql": "microsoft sql server",
    "mssql": "microsoft sql server",
    "aws": "amazon web services",
    "amazon aws": "amazon web services",
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "azure": "microsoft azure",
    "ms azure": "microsoft azure",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "tensor flow": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "git hub": "github",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "bi": "business intelligence",
    "eda": "exploratory data analysis",
    "etl": "extract transform load",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node js": "node.js",
    "vuejs": "vue.js",
    "vue js": "vue.js",
}
