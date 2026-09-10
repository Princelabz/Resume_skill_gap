"""
app.py: Streamlit User Interface for NLP-Based Skill Gap Analyzer.

Provides an interactive, explainable dashboard for comparing candidate resumes
against job descriptions using classical NLP techniques and the O*NET knowledge base.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    SKILLS_DB_PATH,
    ALIAS_MAP_PATH,
    GAP_THRESHOLDS,
    REQUIRED_RAW_FILES,
    RAW_DATA_DIR,
    FALLBACK_DATA_DIR,
)
from src.preprocessing import clean_text, extract_text_from_pdf, lemmatize_text
from src.skill_normalizer import SkillNormalizer
from src.skill_extractor import SkillExtractor
from src.matcher import SkillMatcher
from src.scoring import ScoringEngine
from src.recommendations import RecommendationEngine
from train import build_skills_knowledge_base, verify_raw_data


# Set Page Config
st.set_page_config(
    page_title="Skill Gap Analyzer | Explainable NLP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, accessible college-level UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-matched {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-missing {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .badge-extra {
        background-color: #E0E7FF;
        color: #4338CA;
        border: 1px solid #A5B4FC;
    }
    .badge-high {
        background-color: #FEF2F2;
        color: #991B1B;
        border-left: 4px solid #EF4444;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .badge-med {
        background-color: #FFFBEB;
        color: #92400E;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .badge-low {
        background-color: #F0FDF4;
        color: #166534;
        border-left: 4px solid #10B981;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_nlp_pipeline():
    """Cache and initialize NLP pipeline components."""
    # If processed database doesn't exist, try building it automatically from raw files
    if not (SKILLS_DB_PATH.exists() and ALIAS_MAP_PATH.exists()):
        try:
            build_skills_knowledge_base()
        except Exception:
            pass

    normalizer = SkillNormalizer(ALIAS_MAP_PATH)
    extractor = SkillExtractor(SKILLS_DB_PATH, normalizer=normalizer)
    matcher = SkillMatcher()
    scoring = ScoringEngine(GAP_THRESHOLDS)
    recommendations = RecommendationEngine(SKILLS_DB_PATH)
    return normalizer, extractor, matcher, scoring, recommendations


def check_knowledge_base_status():
    """Verify presence of O*NET dataset and processed JSON knowledge base."""
    has_processed = SKILLS_DB_PATH.exists() and ALIAS_MAP_PATH.exists()
    return has_processed


def load_sample_content(filename: str) -> str:
    """Read a sample text file from sample_data/ directory."""
    path = BASE_DIR / "sample_data" / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


# Sidebar: Setup, Information & Settings
with st.sidebar:
    st.header("⚙️ System Status & Config")
    kb_ready = check_knowledge_base_status()

    if kb_ready:
        st.success("✅ O*NET Knowledge Base Loaded")
    else:
        st.warning("⚠️ Knowledge base not initialized.")
        if st.button("🔨 Build Knowledge Base Now", use_container_width=True):
            with st.spinner("Processing O*NET software and essential skills..."):
                try:
                    build_skills_knowledge_base()
                    st.cache_resource.clear()
                    st.success("Knowledge Base Built Successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📚 Explainable NLP Pipeline")
    st.markdown("""
    1. **Text Preprocessing**: Cleaning, tokenization & lemmatization.
    2. **Multi-Word Skill Extraction**: Case-insensitive Phrase Matching.
    3. **Alias Normalization**: Acronym mapping (*'ML'* → *'machine learning'*).
    4. **Set-Theoretic Matching**: Matched, Missing & Extra skill sets.
    5. **TF-IDF & Cosine Similarity**: Document vector space overlap.
    6. **O*NET Demand Prioritization**: Hot Technology & In-Demand scoring.
    """)

    st.markdown("---")
    st.markdown("### 📊 Gap Level Thresholds")
    for level, (low, high) in GAP_THRESHOLDS.items():
        st.caption(f"**{level}**: {low:.0f}% – {high:.0f}%")

    st.markdown("---")
    st.caption("College-Level NLP Project • Classical & Explainable Methods")


# Main Header
st.markdown('<div class="main-header">🎯 NLP-Based Skill Gap Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explainable Resume and Job Description Matching powered by O*NET Occupational Data & Classical NLP.</div>',
    unsafe_allow_html=True
)

if not kb_ready:
    st.info(
        "👋 **First-time Setup Required**: The O*NET Knowledge Base has not been built yet. "
        "Click the **'Build Knowledge Base Now'** button in the sidebar or run `python train.py` in your terminal."
    )

# Input Section: 2 Columns
col_res, col_job = st.columns(2)

# Resume Input Column
with col_res:
    st.subheader("📄 1. Candidate Resume")
    resume_input_method = st.radio(
        "Resume Input Source:",
        ["Upload File (PDF / TXT)", "Paste Text Directly", "Load Sample Resume"],
        horizontal=True,
        key="res_method",
    )

    resume_text = ""
    if resume_input_method == "Upload File (PDF / TXT)":
        uploaded_res = st.file_uploader(
            "Upload Resume (PDF or TXT)",
            type=["pdf", "txt"],
            key="res_file",
        )
        if uploaded_res:
            if uploaded_res.name.lower().endswith(".pdf"):
                extracted, msg = extract_text_from_pdf(uploaded_res)
                if extracted:
                    resume_text = extracted
                    st.success(f"Extracted {len(resume_text):,} characters from PDF.")
                else:
                    st.error(msg)
            else:
                resume_text = uploaded_res.read().decode("utf-8", errors="replace")
                st.success(f"Loaded {len(resume_text):,} characters.")

    elif resume_input_method == "Paste Text Directly":
        resume_text = st.text_area(
            "Paste Resume Text here:",
            height=260,
            placeholder="Paste candidate resume plain text...",
            key="res_paste",
        )
    else:
        sample_choice = st.selectbox(
            "Select Sample Resume:",
            [
                ("sample_resume_data_scientist.txt", "Data Scientist (Alex Rivera)"),
                ("sample_resume_web_developer.txt", "Frontend Developer (Jordan Taylor)"),
            ],
            format_func=lambda x: x[1],
            key="res_sample_choice",
        )
        resume_text = load_sample_content(sample_choice[0])
        st.text_area("Sample Resume Preview:", resume_text, height=200, disabled=True)

# Job Description Input Column
with col_job:
    st.subheader("💼 2. Job Description")
    job_input_method = st.radio(
        "Job Description Input Source:",
        ["Upload File (PDF / TXT)", "Paste Text Directly", "Load Sample Job"],
        horizontal=True,
        key="job_method",
    )

    job_text = ""
    if job_input_method == "Upload File (PDF / TXT)":
        uploaded_job = st.file_uploader(
            "Upload Job Description (PDF or TXT)",
            type=["pdf", "txt"],
            key="job_file",
        )
        if uploaded_job:
            if uploaded_job.name.lower().endswith(".pdf"):
                extracted, msg = extract_text_from_pdf(uploaded_job)
                if extracted:
                    job_text = extracted
                    st.success(f"Extracted {len(job_text):,} characters from PDF.")
                else:
                    st.error(msg)
            else:
                job_text = uploaded_job.read().decode("utf-8", errors="replace")
                st.success(f"Loaded {len(job_text):,} characters.")

    elif job_input_method == "Paste Text Directly":
        job_text = st.text_area(
            "Paste Job Description here:",
            height=260,
            placeholder="Paste job posting text...",
            key="job_paste",
        )
    else:
        sample_job_choice = st.selectbox(
            "Select Sample Job Posting:",
            [
                ("sample_job_data_scientist.txt", "Senior Data Scientist / ML Engineer"),
                ("sample_job_fullstack.txt", "Full Stack Software Engineer"),
            ],
            format_func=lambda x: x[1],
            key="job_sample_choice",
        )
        job_text = load_sample_content(sample_job_choice[0])
        st.text_area("Sample Job Preview:", job_text, height=200, disabled=True)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analyze Match & Skill Gap", type="primary", use_container_width=True)

# Analysis Pipeline Execution
if analyze_btn:
    if not resume_text.strip():
        st.warning("⚠️ Please provide a resume before analyzing.")
    elif not job_text.strip():
        st.warning("⚠️ Please provide a job description before analyzing.")
    else:
        normalizer, extractor, matcher, scoring, recommendations = load_nlp_pipeline()

        with st.spinner("Executing NLP Pipeline: extracting skills, calculating TF-IDF cosine similarity, and matching..."):
            # 1. Skill Extraction
            res_skills = extractor.extract_skills(resume_text)
            job_skills = extractor.extract_skills(job_text)

            # 2. Skill Matching
            match_results = matcher.match_skills(res_skills, job_skills)

            # 3. Scoring & Gap Determination
            match_pct = match_results["match_percentage"]
            gap_level, gap_desc = scoring.determine_gap_level(match_pct)
            tfidf_res = scoring.compute_tfidf_similarity(resume_text, job_text)

            # 4. Demand Prioritization for Missing Skills
            recs = recommendations.prioritize_missing_skills(match_results["missing_skills"])

        # Display Top KPI Metric Cards
        st.markdown("---")
        st.header("📊 Match Summary & Gap Score")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight:600;">SKILL MATCH SCORE</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #1E40AF;">{match_pct:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #475569;">{match_results['total_matched']} of {match_results['total_required_skills']} required skills</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight:600;">TEXT SIMILARITY</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #0D9488;">{tfidf_res['text_similarity_percentage']:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #475569;">TF-IDF Cosine Similarity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            level_colors = {
                "Strong Match": "#15803D",
                "Good Match": "#0369A1",
                "Moderate Gap": "#B45309",
                "Large Gap": "#B91C1C",
            }
            color = level_colors.get(gap_level, "#1E293B")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight:600;">SKILL GAP LEVEL</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {color};">{gap_level}</div>
                    <div style="font-size: 0.8rem; color: #475569;">{gap_desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #64748B; font-weight:600;">MISSING SKILLS</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #DC2626;">{match_results['total_missing']}</div>
                    <div style="font-size: 0.8rem; color: #475569;">High Priority: {recs['high_count']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Views for in-depth inspection
        tab_skills, tab_charts, tab_recs, tab_nlp = st.tabs([
            "🧩 Skill Breakdown",
            "📈 Visual Analytics",
            "🚀 Learning Roadmap (O*NET)",
            "🔍 NLP Engine Inspector",
        ])

        # TAB 1: Skill Breakdown Chips
        with tab_skills:
            st.subheader("Skills Comparison Breakdown")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"#### 📄 Skills Found in Resume ({len(res_skills)})")
                if res_skills:
                    chips_html = "".join([f'<span class="badge badge-extra">{s.title()}</span>' for s in res_skills])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.info("No recognized technical skills found in resume.")

            with col_b:
                st.markdown(f"#### 💼 Skills Required by Job ({len(job_skills)})")
                if job_skills:
                    chips_html = "".join([f'<span class="badge badge-matched">{s.title()}</span>' for s in job_skills])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.info("No recognized technical skills found in job description.")

            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"#### ✅ Matched Skills ({match_results['total_matched']})")
                if match_results["matched_skills"]:
                    chips = "".join([f'<span class="badge badge-matched">{s.title()}</span>' for s in match_results["matched_skills"]])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.warning("No matching skills found.")

            with c2:
                st.markdown(f"#### ❌ Missing Skills ({match_results['total_missing']})")
                if match_results["missing_skills"]:
                    chips = "".join([f'<span class="badge badge-missing">{s.title()}</span>' for s in match_results["missing_skills"]])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.success("Zero missing skills! Perfect requirement coverage.")

            with c3:
                st.markdown(f"#### ➕ Extra Candidate Skills ({match_results['total_extra']})")
                if match_results["extra_skills"]:
                    chips = "".join([f'<span class="badge badge-extra">{s.title()}</span>' for s in match_results["extra_skills"]])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.info("No extra skills beyond requirements.")

        # TAB 2: Visual Analytics Charts
        with tab_charts:
            st.subheader("Visual Analytics")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Matched vs Missing vs Extra Bar Chart
                df_counts = pd.DataFrame({
                    "Skill Category": ["Matched Skills", "Missing Skills", "Extra Skills"],
                    "Count": [
                        match_results["total_matched"],
                        match_results["total_missing"],
                        match_results["total_extra"],
                    ],
                    "Color": ["#22C55E", "#EF4444", "#6366F1"],
                })

                fig_bar = px.bar(
                    df_counts,
                    x="Skill Category",
                    y="Count",
                    color="Skill Category",
                    color_discrete_map={
                        "Matched Skills": "#22C55E",
                        "Missing Skills": "#EF4444",
                        "Extra Skills": "#6366F1",
                    },
                    text="Count",
                    title="Skill Alignment Distribution",
                )
                fig_bar.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_bar, use_container_width=True)

            with chart_col2:
                # Missing Skill Priority Donut Chart
                if match_results["total_missing"] > 0:
                    df_priority = pd.DataFrame({
                        "Priority": ["High Priority", "Medium Priority", "Low Priority"],
                        "Count": [recs["high_count"], recs["medium_count"], recs["low_count"]],
                    })
                    fig_pie = px.pie(
                        df_priority,
                        names="Priority",
                        values="Count",
                        color="Priority",
                        color_discrete_map={
                            "High Priority": "#DC2626",
                            "Medium Priority": "#F59E0B",
                            "Low Priority": "#10B981",
                        },
                        hole=0.45,
                        title="Missing Skill Priority Breakdown (O*NET)",
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.success("No missing skills to prioritize.")

            # Shared TF-IDF Terms Bar Chart
            if tfidf_res.get("top_shared_terms"):
                st.markdown("#### 🔤 Top Contributing TF-IDF Overlap N-Grams")
                df_terms = pd.DataFrame(tfidf_res["top_shared_terms"])
                fig_terms = px.bar(
                    df_terms,
                    x="overlap_weight",
                    y="term",
                    orientation="h",
                    title="Top Shared TF-IDF Terms (Vocabulary Overlap)",
                    labels={"overlap_weight": "TF-IDF Overlap Weight", "term": "Vocabulary Term / N-Gram"},
                    color="overlap_weight",
                    color_continuous_scale="Blues",
                )
                fig_terms.update_layout(yaxis=dict(autorange="reversed"), height=300)
                st.plotly_chart(fig_terms, use_container_width=True)

        # TAB 3: Recommendations
        with tab_recs:
            st.subheader("🎯 Prioritized Learning Recommendations for Missing Skills")
            st.markdown(
                "Priorities are weighted using authentic **O*NET Occupational Knowledge Base** data "
                "(Hot Technologies, In-Demand flags, and occupational frequency)."
            )

            if match_results["total_missing"] == 0:
                st.success("🎉 You meet all mandatory skill requirements for this position!")
            else:
                if recs["high_priority"]:
                    st.markdown("### 🔥 High Priority Skills (Immediate Focus)")
                    for item in recs["high_priority"]:
                        st.markdown(
                            f"""
                            <div class="badge-high">
                                <strong>{item['skill']}</strong> ({item['category']})<br>
                                <span style="font-size:0.9rem;">{item['reason']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if recs["medium_priority"]:
                    st.markdown("### ⚡ Medium Priority Skills (Secondary Focus)")
                    for item in recs["medium_priority"]:
                        st.markdown(
                            f"""
                            <div class="badge-med">
                                <strong>{item['skill']}</strong> ({item['category']})<br>
                                <span style="font-size:0.9rem;">{item['reason']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if recs["low_priority"]:
                    st.markdown("### 📌 Low Priority / Supplementary Skills")
                    for item in recs["low_priority"]:
                        st.markdown(
                            f"""
                            <div class="badge-low">
                                <strong>{item['skill']}</strong> ({item['category']})<br>
                                <span style="font-size:0.9rem;">{item['reason']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # TAB 4: NLP Engine Inspector
        with tab_nlp:
            st.subheader("🔍 Explainable NLP Pipeline Inspector")
            st.markdown("Inspect how traditional NLP algorithms processed the input step-by-step.")

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("#### 1. Skill Match Percentage Calculation")
                st.latex(r"\text{Skill Match \%} = \left(\frac{|\text{Matched Skills}|}{|\text{Required Job Skills}|}\right) \times 100")
                st.code(
                    f"Matched Skills Count   = {match_results['total_matched']}\n"
                    f"Required Skills Count  = {match_results['total_required_skills']}\n"
                    f"Match Percentage       = ({match_results['total_matched']} / {match_results['total_required_skills']}) * 100 = {match_pct:.2f}%\n"
                    f"Gap Level Threshold    = {gap_level}"
                )

            with exp_col2:
                st.markdown("#### 2. TF-IDF & Cosine Similarity Calculation")
                st.latex(r"\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}")
                st.code(
                    f"Vector Space Model     = TF-IDF (Unigrams & Bigrams, English Stopwords Removed)\n"
                    f"Cosine Similarity (0-1)= {tfidf_res['cosine_similarity_score']:.4f}\n"
                    f"Text Similarity %      = {tfidf_res['text_similarity_percentage']:.2f}%\n"
                    f"Explanation            = {tfidf_res['explanation']}"
                )

            st.markdown("---")
            st.markdown("#### 3. Preprocessed Text Preview")
            p1, p2 = st.columns(2)
            with p1:
                st.caption("Lemmatized Resume Text (First 300 chars):")
                st.text(lemmatize_text(clean_text(resume_text))[:300] + "...")
            with p2:
                st.caption("Lemmatized Job Description Text (First 300 chars):")
                st.text(lemmatize_text(clean_text(job_text))[:300] + "...")
