# NLP-Based Skill Gap Analyzer for Resume and Job Description Matching

A transparent, explainable Natural Language Processing (NLP) web application that compares candidate resumes with target job descriptions. The system extracts technical and essential skills, normalizes acronyms and variants, computes skill match percentages, evaluates document textual similarity via TF-IDF and Cosine Similarity, classifies skill gap severity, and provides prioritized learning recommendations powered by the **O*NET Occupational Knowledge Base**.


---
### Output 

https://princelabz-resume-skill-gap-app-axr0zr.streamlit.app/
---


---

## 📌 Project Description

Matching a job seeker's resume against a job description is a fundamental challenge in talent acquisition and career planning. Traditional keyword search often fails due to synonyms and acronyms (e.g., *AWS* vs. *Amazon Web Services*), while complex "black-box" LLMs provide opaque similarity scores without explainability.

This project delivers an **explainable, deterministic, and educational NLP system** that:
- Ingests resumes and job descriptions (PDF or TXT).
- Extracts technical software competencies and essential workplace skills without external LLM APIs.
- Normalizes variations and aliases to canonical skill names.
- Computes separate **Skill Match Percentage** and **TF-IDF Cosine Text Similarity**.
- Classifies gap levels (*Strong Match*, *Good Match*, *Moderate Gap*, *Large Gap*).
- Recommends missing skills prioritized by authentic **O*NET Hot Technology** and **In-Demand** metrics.

---

## 🚀 Key Features

- **Multi-Format Input**: Upload PDF or TXT files, paste raw text, or load built-in sample resumes and job postings.
- **Safe PDF Extraction**: Built with `pypdf` with automated detection and clear warnings for non-selectable image-only scans.
- **Explainable Multi-Word Phrase Extraction**: Captures compound skills (*"Natural Language Processing"*, *"SQL Server"*, *"Microsoft Excel"*) without naive word splitting.
- **Configurable Alias Normalization**: Maps acronyms, typos, and variations (*"ml"* → *"machine learning"*, *"powerbi"* → *"power bi"*).
- **Dual Independent Metrics**:
  - **Skill Match %**: Deterministic set overlap of required skills.
  - **Text Similarity %**: Unigram/bigram TF-IDF cosine similarity across document content.
- **Demand-Aware Learning Roadmap**: Prioritizes missing skills into High, Medium, and Low Priority using authentic O*NET data.
- **Interactive Visualizations**: Interactive Plotly bar charts, priority donuts, and top shared TF-IDF vocabulary n-grams.
- **NLP Engine Inspector**: Real-time view of mathematical calculations and preprocessed text representations.

---

## 🔬 NLP Techniques Used

1. **Text Preprocessing & Normalization**:
   - Noise removal, unicode quotation and bullet normalization.
   - Case-insensitive tokenization and punctuation boundary handling (`C++`, `C#`, `.NET`, `Node.js`).
   - Stopword filtering and POS-aware lemmatization via NLTK `WordNetLemmatizer`.
2. **Explainable Skill Extraction**:
   - spaCy `PhraseMatcher` for case-insensitive token sequence matching.
   - Longest-span-first entity disambiguation (`filter_spans`).
   - Lookaround regex patterns for programming languages with symbols.
3. **Canonical Normalization**:
   - Standalone configurable synonym dictionary (`alias_map.json`).
4. **Vector Space Modeling & Similarity**:
   - TF-IDF Vectorizer with unigrams and bigrams ($1 \le n \le 2$).
   - Cosine Similarity $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
5. **Set-Theoretic Gap Analysis**:
   - Matched: $S_{\text{resume}} \cap S_{\text{job}}$
   - Missing: $S_{\text{job}} \setminus S_{\text{resume}}$
   - Extra: $S_{\text{resume}} \setminus S_{\text{job}}$

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.9+
- **NLP & Text Processing**: NLTK, spaCy, Regular Expressions (re)
- **Machine Learning & Vectorization**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Handling**: Pandas, NumPy
- **PDF Extraction**: pypdf
- **Interactive Dashboard**: Streamlit
- **Visualizations**: Plotly Express & Plotly Graph Objects

---

## 📊 Dataset Information

The system references authentic data from the **O*NET (Occupational Information Network)** database, sponsored by the U.S. Department of Labor:

- **`software_skills.csv`** (31,821 records): Software technologies mapped to SOC codes with `Hot Technology` and `In Demand` flags.
- **`essential_skills.csv`** (18,200 records): Workplace skills with numerical importance ratings (1.0–5.0 scale).
- **`occupation_data.csv`**: Standard SOC occupational titles and job descriptions.
- **`job_titles.csv`**: Alternative job titles and cross-referenced industry terms.
- **Processed Artifacts**:
  - `data/processed/skills_db.json`: 8,798 canonical indexed skills.
  - `data/processed/alias_map.json`: 636 normalization aliases.

---

## 📂 Project Structure

```
skill-gap-analyzer/
│
├── data/
│   ├── raw/                       # O*NET CSV files
│   │   ├── software_skills.csv
│   │   ├── essential_skills.csv
│   │   ├── occupation_data.csv
│   │   └── job_titles.csv
│   ├── processed/                 # Indexed JSON lookup tables
│   │   ├── skills_db.json
│   │   └── alias_map.json
│   ├── evaluation_set.json        # Labeled benchmark test set
│   └── README.md                  # Dataset documentation
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Thresholds, paths & default aliases
│   ├── preprocessing.py           # Text cleaning, PDF parsing, lemmatization
│   ├── skill_normalizer.py        # Synonym & acronym normalization
│   ├── skill_extractor.py         # Multi-word phrase & regex skill extraction
│   ├── matcher.py                 # Set-theoretic matching & gap calculation
│   ├── scoring.py                 # Match %, TF-IDF & Cosine similarity
│   └── recommendations.py         # O*NET demand-weighted recommendations
│
├── sample_data/                   # Sample resumes & job descriptions (TXT & PDF)
│   ├── sample_resume_data_scientist.txt / .pdf
│   ├── sample_resume_web_developer.txt / .pdf
│   ├── sample_job_data_scientist.txt / .pdf
│   └── sample_job_fullstack.txt / .pdf
│
├── tests/                         # Automated unit tests
│   ├── test_preprocessing.py
│   ├── test_skill_extractor.py
│   ├── test_matcher.py
│   ├── test_scoring.py
│   └── test_recommendations.py
│
├── app.py                         # Streamlit web application entry point
├── train.py                       # O*NET database build & indexing script
├── evaluate.py                    # Quantitative evaluation script
├── requirements.txt               # Essential dependencies
├── README.md                      # Complete project documentation
└── .gitignore                     # Git configuration
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/skill-gap-analyzer.git
cd skill-gap-analyzer
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize the O*NET Knowledge Base
```bash
python train.py
```

---

## ▶️ Running Locally

Launch the Streamlit dashboard from the project root directory:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Running Unit Tests
```bash
python -m unittest discover tests
```

### Running System Evaluation
```bash
python evaluate.py
```

---

## ☁️ How to Deploy on Streamlit Cloud

You can deploy this application for free on **Streamlit Community Cloud**:

1. **Push to GitHub**:
   - Ensure all project files, `requirements.txt`, and `.gitignore` are committed and pushed to your GitHub repository.
2. **Sign in to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Create New App**:
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the main file path to **`app.py`**.
4. **Deploy**:
   - Click **"Deploy!"**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch your web application.

---

## ⚠️ Limitations

1. **Scanned PDF Text**: The application parses text-based PDFs. Image-only flattened documents will prompt the user that OCR is not active.
2. **Domain-Specific Proprietary Jargon**: Niche internal enterprise tools not present in the O*NET taxonomy require adding an alias entry to `data/processed/alias_map.json`.
3. **Experience Duration**: The system evaluates skill presence and topical similarity; it does not infer years of seniority beyond explicit text statements.

---

## 🔮 Future Improvements

1. **Optical Character Recognition (OCR)**: Integrate Tesseract OCR (`pytesseract`) as a fallback for scanned resumes.
2. **Resume Section Segmentation**: Segment resumes into discrete sections (*Experience*, *Education*, *Certifications*) to give higher weight to skills used in recent employment.
3. **Career Transition Pathways**: Map candidate extra skills to alternate O*NET career roles for broader career counseling.
