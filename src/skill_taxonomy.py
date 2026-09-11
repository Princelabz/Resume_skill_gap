"""
src/skill_taxonomy.py: Curated Technical Skill Taxonomy and Filtering Layer.

This module provides:
1. A curated vocabulary of legitimate technical/professional skills
2. A blocklist of generic words, verbs, company names, and platform names
   that must NOT be treated as skills
3. A context-validator to reject false positives (e.g. "linkedin" in contact info)
4. Section detector for resume/JD structure

This sits BETWEEN the O*NET database and the final extracted skill list,
acting as a validation/filtering layer.
"""

import re
from typing import Set, List, Tuple, Dict

# ---------------------------------------------------------------------------
# BLOCKLIST: terms that should NEVER be treated as standalone skills
# ---------------------------------------------------------------------------

# Generic single-word verbs that appear in both O*NET and resumes but are NOT skills
GENERIC_VERBS: Set[str] = {
    "analyze", "analyses", "analysed", "analyzed", "analyzing",
    "build", "builds", "built", "building",
    "create", "creates", "created", "creating",
    "use", "uses", "used", "using",
    "work", "works", "worked", "working",
    "manage", "manages", "managed", "managing",
    "develop", "develops", "developed", "developing",
    "design", "designs", "designed", "designing",
    "implement", "implements", "implemented", "implementing",
    "support", "supports", "supported", "supporting",
    "maintain", "maintains", "maintained", "maintaining",
    "test", "tests", "tested", "testing",
    "deploy", "deploys", "deployed", "deploying",
    "monitor", "monitors", "monitored", "monitoring",
    "review", "reviews", "reviewed", "reviewing",
    "define", "defines", "defined", "defining",
    "identify", "identifies", "identified", "identifying",
    "prepare", "prepares", "prepared", "preparing",
    "perform", "performs", "performed", "performing",
    "provide", "provides", "provided", "providing",
    "ensure", "ensures", "ensured", "ensuring",
    "research", "researches", "researched", "researching",
    "optimize", "optimizes", "optimized", "optimizing",
    "automate", "automates", "automated", "automating",
    "integrate", "integrates", "integrated", "integrating",
    "extract", "extracts", "extracted", "extracting",
    "transform", "transforms", "transformed", "transforming",
    "load", "loads", "loaded", "loading",
    "collect", "collects", "collected", "collecting",
    "process", "processes", "processed", "processing",
    "generate", "generates", "generated", "generating",
    "present", "presents", "presented", "presenting",
    "report", "reports", "reported", "reporting",
    "lead", "leads", "led", "leading",
    "coordinate", "coordinates", "coordinated", "coordinating",
    "collaborate", "collaborates", "collaborated", "collaborating",
    "communicate", "communicates", "communicated", "communicating",
    "assist", "assists", "assisted", "assisting",
    "troubleshoot", "troubleshoots", "troubleshooted", "troubleshooting",
    "resolve", "resolves", "resolved", "resolving",
    "configure", "configures", "configured", "configuring",
    "install", "installs", "installed", "installing",
    "update", "updates", "updated", "updating",
    "migrate", "migrates", "migrated", "migrating",
    "document", "documents", "documented", "documenting",
    "train", "trains", "trained", "training",
    "evaluate", "evaluates", "evaluated", "evaluating",
    "apply", "applies", "applied", "applying",
    "establish", "establishes", "established", "establishing",
    "structure", "structures", "structured", "structuring",
    "conduct", "conducts", "conducted", "conducting",
}

# Generic non-technical nouns/adjectives that appear in O*NET but are NOT skills
GENERIC_NON_SKILLS: Set[str] = {
    "science", "technology", "engineering", "mathematics",
    "information", "system", "systems", "solution", "solutions",
    "data", "analysis", "analytics", "intelligence", "platform",
    "service", "services", "application", "applications",
    "management", "strategy", "strategies", "approach",
    "model", "models", "framework", "frameworks",
    "process", "processes", "workflow", "workflows",
    "tool", "tools", "technique", "techniques",
    "method", "methods", "methodology", "methodologies",
    "project", "projects", "product", "products",
    "team", "teams", "stakeholder", "stakeholders",
    "requirement", "requirements", "specification", "specifications",
    "document", "documents", "documentation",
    "report", "reports", "presentation", "presentations",
    "meeting", "meetings", "client", "clients",
    "company", "companies", "organization", "organizations",
    "industry", "field", "domain", "sector",
    "experience", "background", "knowledge", "understanding",
    "skill", "skills", "ability", "abilities", "capability",
    "degree", "bachelor", "master", "phd", "certification",
    "year", "years", "month", "months",
    "responsible", "responsibilities", "role", "roles",
    "task", "tasks", "activity", "activities",
    "best", "practices", "good", "excellent", "strong",
    "bachelor", "masters", "degree", "diploma",
    "etc", "including", "using", "via", "based",
    "related", "relevant", "various", "multiple",
}

# Company/platform names that should ONLY be skills if part of a recognized product phrase
COMPANY_PLATFORM_NAMES: Set[str] = {
    "google", "microsoft", "amazon", "apple", "facebook", "meta",
    "linkedin", "twitter", "instagram", "bitbucket",
    "slack", "trello", "asana", "notion",
    "salesforce", "oracle", "sap", "ibm", "hp", "dell",
    "netflix", "spotify", "uber", "airbnb", "shopify",
    "zoom", "teams", "webex", "skype",
    "adobe", "sketch", "canva",
    "heroku", "vercel", "netlify", "cloudflare",
    "stripe", "twilio", "sendgrid",
    "coursera", "udemy", "edx", "pluralsight",
}

# Words that appear in legitimate O*NET skill names but are ambiguous alone
AMBIGUOUS_SINGLE_WORDS: Set[str] = {
    "natural", "language", "deep", "machine", "artificial",
    "computer", "neural", "network", "networks",
    "cloud", "web", "mobile", "desktop",
    "front", "back", "full",
    "large", "small", "big",
    "transform", "extract", "pipeline",
    "model", "algorithm", "library",
    "query", "database", "server",
    "security", "testing", "debugging",
}

# Combined master blocklist for single-token terms
SINGLE_TOKEN_BLOCKLIST: Set[str] = (
    GENERIC_VERBS
    | GENERIC_NON_SKILLS
    | COMPANY_PLATFORM_NAMES
    | AMBIGUOUS_SINGLE_WORDS
)

# ---------------------------------------------------------------------------
# ALLOWED SINGLE-WORD SKILLS: recognized legitimate one-word technical skills
# ---------------------------------------------------------------------------

ALLOWED_SINGLE_WORD_SKILLS: Set[str] = {
    # Programming languages & Special Symbol terms
    "python", "java", "javascript", "typescript", "sql", "r",
    "scala", "kotlin", "swift", "go", "rust", "perl", "ruby",
    "matlab", "html", "css", "bash", "shell", "powershell",
    "c", "c++", "c#", ".net", "node.js", "cobol", "fortran", "assembly",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite",
    "cassandra", "elasticsearch", "hbase", "neo4j", "dynamodb",
    # Tools & Frameworks (unambiguous single-word)
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "tensorflow", "pytorch", "keras", "sklearn", "spacy", "scikit-learn",
    "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "git", "github", "gitlab", "linux", "ubuntu", "debian", "bash",
    "tableau", "excel", "powerpoint", "word", "powerbi",
    "hadoop", "spark", "kafka", "airflow", "dbt",
    "sklearn", "xgboost", "lightgbm", "catboost",
    "flask", "django", "fastapi", "react", "angular", "vue",
    "spark", "hive", "pig", "sqoop", "flume",
    "snowflake", "redshift", "bigquery", "databricks",
    "looker", "domo", "qlik",
    "regex", "json", "xml", "yaml", "csv", "api",
    "postman", "swagger", "graphql",
    "jira", "confluence",
    "scrum", "agile", "kanban",
    "statistics", "probability", "calculus", "algebra",
    "ml", "dl", "nlp", "cv", "js", "ts",
    # Cloud platforms (acronym only)
    "aws", "gcp", "azure",
}

# ---------------------------------------------------------------------------
# MULTI-WORD SKILL ALLOWLIST: terms that MUST be kept together as a phrase
# These are protected multi-word skills that should always be extracted as units
# ---------------------------------------------------------------------------

REQUIRED_MULTI_WORD_SKILLS: Set[str] = {
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "artificial intelligence",
    "data science",
    "data analysis",
    "data engineering",
    "data visualization",
    "data modeling",
    "data modelling",
    "data mining",
    "data wrangling",
    "data cleaning",
    "data pipelines",
    "data warehouse",
    "data lake",
    "data governance",
    "data quality",
    "big data",
    "exploratory data analysis",
    "statistical analysis",
    "statistical modeling",
    "predictive modeling",
    "predictive analytics",
    "business intelligence",
    "business analytics",
    "extract transform load",
    "etl pipeline",
    "feature engineering",
    "model deployment",
    "a/b testing",
    "ab testing",
    "hypothesis testing",
    "time series analysis",
    "time series forecasting",
    "anomaly detection",
    "sentiment analysis",
    "text mining",
    "neural network",
    "neural networks",
    "random forest",
    "decision tree",
    "decision trees",
    "support vector machine",
    "support vector machines",
    "linear regression",
    "logistic regression",
    "gradient boosting",
    "reinforcement learning",
    "transfer learning",
    "dimensionality reduction",
    "principal component analysis",
    "cluster analysis",
    "k-means clustering",
    "microsoft excel",
    "microsoft word",
    "microsoft powerpoint",
    "microsoft access",
    "microsoft sql server",
    "microsoft azure",
    "microsoft office",
    "google cloud platform",
    "google analytics",
    "google sheets",
    "google bigquery",
    "google data studio",
    "amazon web services",
    "amazon redshift",
    "amazon s3",
    "amazon ec2",
    "amazon sagemaker",
    "power bi",
    "power bi desktop",
    "power query",
    "power automate",
    "power apps",
    "qlik sense",
    "qlik view",
    "looker studio",
    "sql server",
    "sql server management studio",
    "sql server reporting services",
    "ssrs",
    "ssas",
    "ssis",
    "oracle database",
    "oracle sql",
    "postgresql database",
    "version control",
    "source control",
    "continuous integration",
    "continuous deployment",
    "ci/cd",
    "cicd",
    "rest api",
    "restful api",
    "web scraping",
    "web development",
    "full stack development",
    "front end development",
    "back end development",
    "object oriented programming",
    "object-oriented programming",
    "functional programming",
    "agile methodology",
    "scrum methodology",
    "project management",
    "product management",
    "stakeholder management",
    "risk management",
    "change management",
    "lean methodology",
    "six sigma",
    "critical thinking",
    "problem solving",
    "data storytelling",
    "report writing",
    "verbal communication",
    "written communication",
    "communication skills",
    "team collaboration",
    "cross functional",
    "attention to detail",
    "node.js",
    "react.js",
    "vue.js",
    "scikit-learn",
    "scikit learn",
    "apache spark",
    "apache kafka",
    "apache hadoop",
    "apache airflow",
    "apache hive",
}

# ---------------------------------------------------------------------------
# Section header patterns for resume section detection
# ---------------------------------------------------------------------------

SECTION_PATTERNS: Dict[str, List[str]] = {
    "skills": [
        r"(?i)^(required\s+|desired\s+|preferred\s+)?(technical\s+)?skills?(\s+summary)?[\s:]*$",
        r"(?i)^core\s+competenc(y|ies)[\s:]*$",
        r"(?i)^technologies[\s:]*$",
        r"(?i)^tools?\s+(and\s+technologies?)?[\s:]*$",
        r"(?i)^tech\s+stack[\s:]*$",
        r"(?i)^programming\s+languages?[\s:]*$",
        r"(?i)^key\s+skills?[\s:]*$",
        r"(?i)^software\s+skills?[\s:]*$",
        r"(?i)^relevant\s+skills?[\s:]*$",
        r"(?i)^expertise[\s:]*$",
        r"(?i)^proficienc(y|ies)[\s:]*$",
    ],
    "experience": [
        r"(?i)^(work\s+|professional\s+|relevant\s+)?experience[\s:]*$",
        r"(?i)^(key\s+|job\s+)?responsibilities[\s:]*$",
        r"(?i)^employment(\s+history)?[\s:]*$",
        r"(?i)^work\s+history[\s:]*$",
        r"(?i)^career(\s+history)?[\s:]*$",
        r"(?i)^positions?\s+held[\s:]*$",
        r"(?i)^preferred[\s:]*$",
    ],
    "projects": [
        r"(?i)^(key\s+|major\s+|selected\s+)?projects?[\s:]*$",
        r"(?i)^portfolio[\s:]*$",
        r"(?i)^academic\s+projects?[\s:]*$",
        r"(?i)^personal\s+projects?[\s:]*$",
    ],
    "education": [
        r"(?i)^education(\s+background)?[\s:]*$",
        r"(?i)^academic(\s+background)?[\s:]*$",
        r"(?i)^qualifications?[\s:]*$",
        r"(?i)^degrees?[\s:]*$",
    ],
    "certifications": [
        r"(?i)^certifications?(\s+and\s+training)?[\s:]*$",
        r"(?i)^licenses?\s+and\s+certifications?[\s:]*$",
        r"(?i)^credentials?[\s:]*$",
        r"(?i)^awards?\s+and\s+certifications?[\s:]*$",
    ],
    "summary": [
        r"(?i)^(professional\s+)?summary[\s:]*$",
        r"(?i)^(professional\s+)?profile[\s:]*$",
        r"(?i)^objective[\s:]*$",
        r"(?i)^about[\s:]*$",
        r"(?i)^overview[\s:]*$",
        r"(?i)^role\s+summary[\s:]*$",
    ],
}

SECTION_DISPLAY_NAMES: Dict[str, str] = {
    "skills": "Skills section",
    "experience": "Experience section",
    "projects": "Project section",
    "education": "Education section",
    "certifications": "Certifications section",
    "summary": "Summary section",
    "body": "General section",
}


def detect_sections(text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Detect section boundaries in a resume or JD.

    Returns a dict mapping section_type -> list of (start_char, end_char) spans.
    """
    lines = text.split("\n")
    section_ranges: Dict[str, List[Tuple[int, int]]] = {}
    current_section: str = "summary"
    current_start: int = 0
    char_pos: int = 0

    for line in lines:
        stripped = line.strip()
        matched_section = None

        for sec_type, patterns in SECTION_PATTERNS.items():
            for pat in patterns:
                if re.match(pat, stripped):
                    matched_section = sec_type
                    break
            if matched_section:
                break

        if matched_section:
            # Close the previous section
            if current_section not in section_ranges:
                section_ranges[current_section] = []
            section_ranges[current_section].append((current_start, char_pos))
            current_section = matched_section
            current_start = char_pos

        char_pos += len(line) + 1  # +1 for newline

    # Close the last section
    if current_section not in section_ranges:
        section_ranges[current_section] = []
    section_ranges[current_section].append((current_start, char_pos))

    return section_ranges


def get_char_section(char_pos: int, section_ranges: Dict[str, List[Tuple[int, int]]]) -> str:
    """Return the display section name that contains the given character position."""
    for sec_type, spans in section_ranges.items():
        for start, end in spans:
            if start <= char_pos < end:
                return SECTION_DISPLAY_NAMES.get(sec_type, "General section")
    return "General section"


def is_valid_skill(
    candidate: str,
    context_text: str = "",
    match_start: int = -1,
) -> bool:
    """
    Determine whether a candidate term should be accepted as a skill.

    Logic:
    1. Multi-word phrases in REQUIRED_MULTI_WORD_SKILLS -> always accept
    2. Single word in ALLOWED_SINGLE_WORD_SKILLS -> accept
    3. Single word in SINGLE_TOKEN_BLOCKLIST -> reject
    4. Multi-word phrases: reject if ALL individual words are generic
    5. Context check: reject if surrounding text suggests it's not a skill
    """
    if not candidate or not candidate.strip():
        return False

    c = candidate.lower().strip()
    c_spaced = c.replace("-", " ")
    tokens = c.split()
    n = len(tokens)
    tokens_spaced = c_spaced.split()
    n_spaced = len(tokens_spaced)

    # Rule 1: explicitly allowed multi-word skills (spaced or hyphenated)
    if (n > 1 and c in REQUIRED_MULTI_WORD_SKILLS) or (n_spaced > 1 and c_spaced in REQUIRED_MULTI_WORD_SKILLS):
        return True

    # Rule 2: explicitly allowed single words
    if (n == 1 and c in ALLOWED_SINGLE_WORD_SKILLS) or (n_spaced == 1 and c_spaced in ALLOWED_SINGLE_WORD_SKILLS):
        return True

    # Rule 3: explicitly blocked single words
    if n == 1 and c in SINGLE_TOKEN_BLOCKLIST and c not in ALLOWED_SINGLE_WORD_SKILLS:
        return False

    # Rule 4: multi-word check - if it's not in the required list,
    # only accept if NOT all words are generic/blocked
    if n > 1:
        # Reject if the phrase is clearly just a verb phrase
        if c in GENERIC_VERBS:
            return False
        # Count how many tokens are in the single-token blocklist
        blocked_count = sum(1 for t in tokens if t in SINGLE_TOKEN_BLOCKLIST)
        # If ALL tokens are blocked terms, reject
        if blocked_count == n:
            return False
        # Specific known false positives
        known_false_multi = {
            "extract transform load",  # Only ETL is acceptable, not as an O*NET prose phrase
        }
        # "extract transform load" IS a valid skill name (it's ETL)
        # We allow it because it IS a recognized technical term
        return True

    # Rule 5: unknown single word - be conservative and reject
    return False


def check_context_for_false_positive(term: str, text: str, match_start: int) -> bool:
    """
    Check if the surrounding context suggests this is NOT a skill.
    Returns True if the context suggests a false positive (should reject).
    """
    # Get a window of text around the match
    window_start = max(0, match_start - 60)
    window_end = min(len(text), match_start + len(term) + 60)
    window = text[window_start:window_end].lower()

    # Check for contact/social media contexts for company/platform names
    if term.lower() in COMPANY_PLATFORM_NAMES:
        # Look for social/contact indicators near the term
        contact_indicators = [
            r"contact", r"reach\s+out", r"connect", r"follow",
            r"profile", r"account", r"linkedin", r"twitter",
            r"email", r"@", r"http", r"www\.", r"\.com",
            r"find\s+me", r"visit", r"check\s+out",
        ]
        for pattern in contact_indicators:
            if re.search(pattern, window):
                return True  # False positive - reject

    return False  # Accept
