from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import docx
import PyPDF2

# Import your existing functions
from utils import predict_domain, get_decision, domain_descriptions

app = FastAPI()

# Allow Framer frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later restrict to your framer domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Extract Resume Text + Skills
# ==============================
def extract_resume_data(file_path):
    text = ""

    try:
        if file_path.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""

        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + " "

        text = text.lower()

        # Extract skills
        all_skills = []
        for skills in domain_descriptions.values():
            all_skills.extend(skills)

        extracted_skills = [
            skill for skill in set(all_skills)
            if skill.lower() in text
        ]

        return text, extracted_skills

    except Exception as e:
        print("Error:", e)
        return "", []


# ==============================
# API Endpoint
# ==============================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    selected_domain: str = Form(...)
):

    # Save uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run resume analysis
    text, skills_list = extract_resume_data(temp_file_path)

    if not skills_list:
        os.remove(temp_file_path)
        return {
            "final_domain": "No skills extracted",
            "confidence": "0%",
            "skill_match": "0%",
            "decision": "Check resume format",
            "skills": []
        }

    skills_text = " ".join(skills_list)

    predicted_domain, model_confidence = predict_domain(skills_text)

    domain_skills = set(skill.lower() for skill in domain_descriptions[selected_domain])
    resume_skills_set = set(skill.lower() for skill in skills_list)

    common_skills = domain_skills & resume_skills_set
    match_score = (len(common_skills) / len(domain_skills) * 100) if domain_skills else 0

    final_domain = selected_domain
    final_confidence = match_score

    decision = get_decision(match_score)

    os.remove(temp_file_path)

    return {
        "final_domain": final_domain,
        "confidence": f"{final_confidence:.1f}%",
        "skill_match": f"{match_score:.1f}%",
        "decision": decision,
        "skills": list(common_skills)
    }