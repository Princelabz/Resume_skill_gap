"""
train.py: Indexing and building the O*NET Knowledge Base for the Skill Gap Analyzer.

This script parses authentic O*NET CSV files to create:
1. data/processed/skills_db.json - Master skill dictionary with category, demand flags, and frequencies.
2. data/processed/alias_map.json - Canonical synonym mapping for skill normalization.
"""

import os
import json
import pandas as pd
from pathlib import Path
from src.config import (
    RAW_DATA_DIR,
    FALLBACK_DATA_DIR,
    REQUIRED_RAW_FILES,
    PROCESSED_DATA_DIR,
    SKILLS_DB_PATH,
    ALIAS_MAP_PATH,
    DEFAULT_ALIASES,
)
from src.skill_taxonomy import is_valid_skill, SINGLE_TOKEN_BLOCKLIST


def find_data_file(filename: str) -> Path:
    """Find a dataset file in RAW_DATA_DIR, or fall back to FALLBACK_DATA_DIR."""
    primary = RAW_DATA_DIR / filename
    if primary.exists():
        return primary
    fallback = FALLBACK_DATA_DIR / filename
    if fallback.exists():
        return fallback
    return None


def verify_raw_data() -> dict:
    """Check availability of all required raw O*NET files."""
    found_files = {}
    missing_files = []

    for fname in REQUIRED_RAW_FILES:
        path = find_data_file(fname)
        if path:
            found_files[fname] = path
        else:
            missing_files.append(fname)

    if missing_files:
        msg = (
            "\n[SETUP REQUIRED] The following O*NET data files are missing:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + f"\nPlease place the CSV files in '{RAW_DATA_DIR}' or '{FALLBACK_DATA_DIR}'."
        )
        raise FileNotFoundError(msg)

    return found_files


def build_skills_knowledge_base():
    """Extract, index, and save the O*NET skills knowledge base."""
    print("=" * 60)
    print("O*NET SKILL KNOWLEDGE BASE GENERATOR")
    print("=" * 60)

    # 1. Verify files
    files = verify_raw_data()
    print("[OK] Found all 4 required O*NET data files:")
    for k, v in files.items():
        print(f"  - {k}: {v}")

    # 2. Process software_skills.csv
    print("\nProcessing software tools and technologies...")
    df_sw = pd.read_csv(files["software_skills.csv"])
    print(f"  Total software skill records loaded: {len(df_sw):,}")

    skills_db = {}

    # Extract distinct software skills
    # Columns: 'O*NET-SOC Code', 'Title', 'Workplace Example', 'Element ID', 'Element Name', 'Hot Technology', 'In Demand'
    for _, row in df_sw.iterrows():
        raw_skill = str(row.get("Workplace Example", "")).strip()
        if not raw_skill or raw_skill.lower() == "nan":
            continue

        skill_key = raw_skill.lower()
        if skill_key in SINGLE_TOKEN_BLOCKLIST:
            continue
        hot_tech = str(row.get("Hot Technology", "N")).strip().upper() == "Y"
        in_demand = str(row.get("In Demand", "N")).strip().upper() == "Y"
        category = str(row.get("Element Name", "Software")).strip()
        soc_code = str(row.get("O*NET-SOC Code", "")).strip()

        if skill_key not in skills_db:
            skills_db[skill_key] = {
                "name": raw_skill,
                "category": category,
                "skill_type": "software",
                "hot_technology": hot_tech,
                "in_demand": in_demand,
                "occupations_count": 1,
                "soc_codes": [soc_code] if soc_code else [],
                "importance_score": 0.0,
            }
        else:
            # Update metadata
            if hot_tech:
                skills_db[skill_key]["hot_technology"] = True
            if in_demand:
                skills_db[skill_key]["in_demand"] = True
            skills_db[skill_key]["occupations_count"] += 1
            if soc_code and soc_code not in skills_db[skill_key]["soc_codes"]:
                skills_db[skill_key]["soc_codes"].append(soc_code)

    # 3. Process essential_skills.csv
    print("\nProcessing essential workplace competencies...")
    df_ess = pd.read_csv(files["essential_skills.csv"])
    print(f"  Total essential skill records loaded: {len(df_ess):,}")

    # Columns: 'O*NET-SOC Code', 'Title', 'Element Name', 'Data Value', etc.
    # We group by 'Element Name' to get the average importance score across occupations
    ess_grouped = df_ess.groupby("Element Name")["Data Value"].agg(["mean", "count"]).reset_index()

    for _, row in ess_grouped.iterrows():
        raw_skill = str(row["Element Name"]).strip()
        if not raw_skill or raw_skill.lower() == "nan":
            continue

        skill_key = raw_skill.lower()
        if skill_key in SINGLE_TOKEN_BLOCKLIST or not is_valid_skill(skill_key):
            continue
        avg_score = float(row["mean"])
        occ_count = int(row["count"])

        if skill_key not in skills_db:
            skills_db[skill_key] = {
                "name": raw_skill,
                "category": "Essential Skill",
                "skill_type": "essential",
                "hot_technology": False,
                "in_demand": avg_score >= 4.0,  # Score on 1-5 scale >= 4.0 considered in-demand
                "occupations_count": occ_count,
                "soc_codes": [],
                "importance_score": round(avg_score, 2),
            }
        else:
            skills_db[skill_key]["importance_score"] = round(avg_score, 2)

    # 4. Add common tech skills that might be written in various standard formats
    curated_common_skills = [
        ("machine learning", "Software & AI", "software", True, True),
        ("deep learning", "Software & AI", "software", True, True),
        ("natural language processing", "Software & AI", "software", True, True),
        ("computer vision", "Software & AI", "software", True, True),
        ("data analysis", "Data & Analytics", "software", True, True),
        ("data science", "Data & Analytics", "software", True, True),
        ("data engineering", "Data & Analytics", "software", True, True),
        ("data visualization", "Data & Analytics", "software", True, True),
        ("statistics", "Data & Analytics", "essential", True, True),
        ("microsoft excel", "Office & Productivity", "software", True, True),
        ("cloud computing", "Infrastructure", "software", True, True),
        ("big data", "Data & Analytics", "software", True, False),
        ("web development", "Software Development", "software", False, False),
        ("frontend development", "Software Development", "software", False, False),
        ("backend development", "Software Development", "software", False, False),
        ("full stack development", "Software Development", "software", True, True),
        ("devops", "DevOps & Cloud", "software", True, True),
        ("git", "Version Control", "software", True, True),
        ("github", "Version Control", "software", True, True),
        ("docker", "Containerization", "software", True, True),
        ("kubernetes", "Containerization", "software", True, True),
        ("rest api", "Software Development", "software", True, True),
        ("graphql", "Software Development", "software", False, False),
        ("scikit-learn", "Data & Analytics", "software", True, True),
        ("tensorflow", "Machine Learning", "software", True, True),
        ("pytorch", "Machine Learning", "software", True, True),
        ("pandas", "Data & Analytics", "software", True, True),
        ("numpy", "Data & Analytics", "software", True, True),
        ("matplotlib", "Data & Analytics", "software", False, False),
        ("seaborn", "Data & Analytics", "software", False, False),
        ("tableau", "Business Intelligence", "software", True, True),
        ("power bi", "Business Intelligence", "software", True, True),
        ("sql", "Database", "software", True, True),
        ("nosql", "Database", "software", True, True),
        ("mongodb", "Database", "software", True, True),
        ("postgresql", "Database", "software", True, True),
        ("mysql", "Database", "software", True, True),
        ("redis", "Database", "software", False, False),
        ("amazon web services", "Cloud", "software", True, True),
        ("aws", "Cloud", "software", True, True),
        ("azure", "Cloud", "software", True, True),
        ("microsoft azure", "Cloud", "software", True, True),
        ("gcp", "Cloud", "software", True, True),
        ("google cloud platform", "Cloud", "software", True, True),
        ("linux", "Operating Systems", "software", True, False),
        ("c++", "Programming Languages", "software", True, True),
        ("c#", "Programming Languages", "software", True, True),
        (".net", "Software Frameworks", "software", True, True),
        ("node.js", "Software Frameworks", "software", True, True),
        ("react", "Software Frameworks", "software", True, True),
        ("angular", "Software Frameworks", "software", True, False),
        ("vue.js", "Software Frameworks", "software", False, False),
        ("javascript", "Programming Languages", "software", True, True),
        ("typescript", "Programming Languages", "software", True, True),
        ("python", "Programming Languages", "software", True, True),
        ("java", "Programming Languages", "software", True, True),
        ("problem solving", "Essential Skill", "essential", False, True),
        ("communication", "Essential Skill", "essential", False, True),
        ("teamwork", "Essential Skill", "essential", False, True),
        ("leadership", "Essential Skill", "essential", False, True),
        ("time management", "Essential Skill", "essential", False, False),
        ("critical thinking", "Essential Skill", "essential", False, True),
        ("agile", "Project Management", "essential", True, True),
        ("scrum", "Project Management", "essential", True, False),
    ]

    for name, cat, stype, hot, indem in curated_common_skills:
        k = name.lower()
        if k not in skills_db:
            skills_db[k] = {
                "name": name.title() if len(name) > 4 else name.upper(),
                "category": cat,
                "skill_type": stype,
                "hot_technology": hot,
                "in_demand": indem,
                "occupations_count": 10,
                "soc_codes": [],
                "importance_score": 4.5 if indem else 3.5,
            }
        else:
            if hot:
                skills_db[k]["hot_technology"] = True
            if indem:
                skills_db[k]["in_demand"] = True

    # 5. Save skills database
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SKILLS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(skills_db, f, indent=2)
    print(f"\n[OK] Saved skills database to: {SKILLS_DB_PATH}")
    print(f"  Total distinct canonical skills: {len(skills_db):,}")
    hot_count = sum(1 for s in skills_db.values() if s.get("hot_technology"))
    indemand_count = sum(1 for s in skills_db.values() if s.get("in_demand"))
    print(f"  'Hot Technology' skills: {hot_count:,}")
    print(f"  'In Demand' skills: {indemand_count:,}")

    # 6. Build and save alias map
    alias_map = dict(DEFAULT_ALIASES)
    # Add auto-generated aliases for skills with punctuation or common formats
    for skill_key in skills_db.keys():
        cleaned_no_punct = skill_key.replace("-", " ").replace(".", " ").replace("/", " ")
        cleaned_no_punct = " ".join(cleaned_no_punct.split())
        if cleaned_no_punct != skill_key and cleaned_no_punct not in alias_map:
            alias_map[cleaned_no_punct] = skill_key

    with open(ALIAS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, indent=2)
    print(f"[OK] Saved alias map to: {ALIAS_MAP_PATH}")
    print(f"  Total normalization aliases configured: {len(alias_map):,}")
    print("=" * 60)
    print("O*NET KNOWLEDGE BASE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    build_skills_knowledge_base()
