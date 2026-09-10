# O*NET Occupational Database

This folder contains data derived from the **O*NET (Occupational Information Network)** database, a comprehensive taxonomy of occupational definitions and skills sponsored by the U.S. Department of Labor.

## Files in `data/raw/`

1. **`software_skills.csv`**
   - Contains software tools and technologies associated with occupational codes.
   - **Key Columns**:
     - `O*NET-SOC Code`: Standard Occupational Classification identifier.
     - `Title`: Occupation title (e.g., Computer Programmers, Data Scientists).
     - `Workplace Example`: Specific software/technology name (e.g., Python, Apache Spark, SQL Server).
     - `Element Name`: Broad software classification category (e.g., Database management system software).
     - `Hot Technology`: Indicates if technology is frequently cited in job postings (`Y` or `N`).
     - `In Demand`: Indicates if technology is in rapid growth (`Y` or `N`).

2. **`essential_skills.csv`**
   - Contains foundational occupational workplace skills and competencies.
   - **Key Columns**:
     - `O*NET-SOC Code`: SOC code identifier.
     - `Title`: Occupation title.
     - `Element Name`: Skill name (e.g., Critical Thinking, Complex Problem Solving, Programming).
     - `Data Value`: Numerical score indicating importance or level required.

3. **`occupation_data.csv`**
   - Occupational titles and comprehensive descriptions.
   - **Key Columns**: `O*NET-SOC Code`, `Title`, `Description`.

4. **`job_titles.csv`**
   - Alternative, colloquial, and industry-standard job titles cross-referenced to SOC occupations.
   - **Key Columns**: `O*NET-SOC Code`, `Title`, `Job Title`.

## Processed Artifacts in `data/processed/`

- **`skills_db.json`**: Indexed dictionary of skills, categories, aliases, frequency, and demand levels.
- **`alias_map.json`**: Configurable synonym dictionary for canonical normalization.
