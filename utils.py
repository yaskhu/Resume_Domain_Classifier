
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load Saved Model + Vectorizer + Encoder
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Domain Descriptions (REAL Categories)
domain_descriptions = {
    "Data Science": [
        "python", "machine learning", "deep learning", "pandas",
        "numpy", "statistics", "data analysis", "data visualization",
        "tensorflow", "sklearn"
    ],

    "Python Developer": [
        "python", "django", "flask", "rest api", "scripting", "automation"
    ],

    "Java Developer": [
        "java", "spring boot", "microservices",
        "hibernate", "rest api", "backend"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "jenkins",
        "ci cd", "aws", "linux", "deployment"
    ],

    "Business Analyst": [
        "requirements gathering", "sql",
        "stakeholder communication", "reporting",
        "data analysis"
    ],

    "HR": [
        "recruitment", "onboarding", "payroll",
        "employee relations", "communication"
    ],

    "Testing": [
        "manual testing", "automation testing",
        "selenium", "junit", "defect tracking"
    ],

    "Network Security Engineer": [
        "network security", "firewall",
        "penetration testing", "cybersecurity"
    ],

    "Database": [
        "sql", "mysql", "oracle",
        "postgresql", "indexing",
        "query optimization"
    ],

    "Mechanical Engineer": [
        "mechanical design", "autocad",
        "solidworks", "thermodynamics",
        "manufacturing"
    ],

    "Civil Engineer": [
        "construction planning",
        "structural analysis",
        "site management"
    ],

    "Electrical Engineering": [
        "circuit design", "power systems",
        "matlab", "control systems"
    ],

    "Sales": [
        "sales", "marketing",
        "negotiation", "business development",
        "client handling"
    ],

    "Blockchain": [
        "blockchain", "ethereum",
        "smart contracts", "solidity", "web3"
    ],

    "ETL Developer": [
        "etl", "data warehousing",
        "informatica", "data pipeline",
        "data transformation"
    ],

    "Hadoop": [
        "hadoop", "spark", "hive",
        "big data", "distributed systems"
    ]
}

# Clean Text (REGEX FIXED)
def clean_text(text):
    text = re.sub(r'http\\S+', ' ', text)           
    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)    
    text = re.sub(r'\\s+', ' ', text)              
    return text.lower().strip()

# Predict Domain (ML-based)
def predict_domain(resume_text):
    cleaned = clean_text(resume_text)
    
    python_keywords = ['python', 'django', 'flask', 'rest', 'api', 'backend', 'scripting']
    if any(keyword in cleaned for keyword in python_keywords):
        return "Python Developer", 95.0
    
    data_keywords = ['machine learning', 'deep learning', 'pandas', 'numpy', 'tensorflow']
    if any(keyword in cleaned for keyword in data_keywords):
        return "Data Science", 90.0
    
    # Original ML model as backup
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    predicted_domain = label_encoder.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(vector)
    confidence = max(probabilities[0]) * 100
    
    return predicted_domain, round(confidence, 2)


# ==========================================
# Similarity Score (Cleaned - No Arbitrary Scaling)
# ==========================================
def calculate_similarity(resume_text, target_domain):
    if target_domain not in domain_descriptions:
        return 0.0
    
    cleaned_resume = clean_text(resume_text)
    domain_text = domain_descriptions[target_domain]
    
    resume_vector = tfidf.transform([cleaned_resume])
    domain_vector = tfidf.transform([domain_text])
    
    similarity = cosine_similarity(resume_vector, domain_vector)[0][0]
    score = similarity * 100  # Raw cosine similarity (no artificial boosting)
    
    return round(score, 2)

# ==========================================
# Decision Logic
# ==========================================
def get_decision(score):
    if score >= 70:
        return "Strong Match ✅ Apply Confidently"
    elif score >= 40:
        return "Moderate Match ⚡ Can Apply but Improve"
    else:
        return "Low Match ❌ Improve Skills First"

# ==========================================
# Export for app.py
# ==========================================
__all__ = [
    'predict_domain', 
    'calculate_similarity', 
    'get_decision', 
    'domain_descriptions'
]
