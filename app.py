import streamlit as st
import docx
import PyPDF2

from utils import predict_domain, get_decision, domain_descriptions

# ==============================
# FORCE LIGHT MODE
# ==============================
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# OVERRIDE STREAMLIT DARK THEME
# ==============================
st.markdown("""
<style>

/* Force entire app background */
.stApp {
    background-color: #f3f6f9;
}

/* Remove dark container */
section[data-testid="stSidebar"] {
    display: none;
}

/* Main Title */
.main-title {
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    color: #1d2226;
    margin-bottom: 35px;
}

/* Upload + Input area spacing */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Card Container */
.result-card {
    background-color: white;
    padding: 45px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    margin-top: 40px;
}

/* Section Labels */
.result-label {
    font-size: 15px;
    color: #6b7280;
    text-align: center;
    margin-top: 28px;
    letter-spacing: 0.5px;
}

/* Big Values */
.result-value {
    font-size: 32px;
    font-weight: 600;
    color: #111827;
    text-align: center;
    margin-top: 6px;
}

/* Divider */
.divider {
    margin: 35px 0;
    border-top: 1px solid #e5e7eb;
}

/* LinkedIn style button */
.stButton>button {
    background-color: #0a66c2;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 25px;
}

.stButton>button:hover {
    background-color: #004182;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.markdown("<div class='main-title'>AI Resume Analyzer</div>", unsafe_allow_html=True)

# ==============================
# INPUT SECTION
# ==============================
uploaded_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"])

selected_domain = st.selectbox(
    "Select Target Domain",
    list(domain_descriptions.keys())
)

# ==============================
# TEXT EXTRACTION
# ==============================
def extract_resume_text(file):
    text = ""

    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "

    return text.lower()

# ==============================
# ANALYZE
# ==============================
if st.button("Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume first.")
    else:
        text = extract_resume_text(uploaded_file)

        all_skills = []
        for skills in domain_descriptions.values():
            all_skills.extend(skills)

        extracted_skills = [
            skill for skill in set(all_skills)
            if skill.lower() in text
        ]

        if not extracted_skills:
            st.error("No skills extracted. Check resume format.")
        else:
            skills_text = " ".join(extracted_skills)

            predicted_domain, model_confidence = predict_domain(skills_text)

            domain_skills = set(skill.lower() for skill in domain_descriptions[selected_domain])
            resume_skills_set = set(skill.lower() for skill in extracted_skills)

            common_skills = domain_skills & resume_skills_set
            match_score = (len(common_skills) / len(domain_skills) * 100) if domain_skills else 0

            decision = get_decision(match_score)

            # 🔥 RESULT CARD
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.markdown("<div class='result-label'>Final Domain</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-value'>{selected_domain}</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-label'>Model Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-value'>{model_confidence:.1f}%</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-label'>Skill Match</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-value'>{match_score:.1f}%</div>", unsafe_allow_html=True)

            st.markdown("<div class='result-label'>Decision</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-value'>{decision}</div>", unsafe_allow_html=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            st.markdown("<div class='result-label'>Matched Skills</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-value'>{', '.join(common_skills)}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)