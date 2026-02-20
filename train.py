# import pandas as pd
# import numpy as np
# import re
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from pathlib import Path
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# import pickle
# data = pd.read_csv("Resume_Domain_Classifier/data/UpdatedResumeDataSet.csv")

# def clean_text(text):
#     text=re.sub(r'http\S+',' ',text)
#     text=re.sub(r'[^a-zA-Z0-9]',' ',text)
#     text=re.sub(r'\W',' ',text)
#     text=re.sub(r'\s+',' ',text)
#     return text.lower()

# data['Resume']=data['Resume'].apply(clean_text)
# x=data['Resume']
# y=data['Category']

# le=LabelEncoder()
# data['Category']=le.fit_transform(data['Category'])

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# # tfidf = TfidfVectorizer(
# #     max_features=5000,
# #     ngram_range=(1,2),
# #     stop_words="english"
# # )
# # x_train_tfidf=tfidf.fit_transform(x_train)
# # x_test_tfidf=tfidf.transform(x_test)

# # model = LinearSVC(class_weight="balanced", max_iter=10000)
# # model.fit(x_train_tfidf,y_train)
# tfidf = TfidfVectorizer(
#     max_features=5000,
#     ngram_range=(1,2),
#     stop_words="english"
# )

# x_train_tfidf = tfidf.fit_transform(x_train)
# x_test_tfidf = tfidf.transform(x_test)

# model = LinearSVC(
#     class_weight="balanced",
#     max_iter=10000
# )

# model.fit(x_train_tfidf, y_train)


# y_pred=model.predict(x_test_tfidf)
# accu=accuracy_score(y_test,y_pred)
# print(f"Model Accuracy: {accu * 100:.2f}%")

# import os
# os.makedirs("models", exist_ok=True)
# pickle.dump(model,open("models/model.pkl", "wb"))
# pickle.dump(tfidf,open("models/tfidf.pkl", "wb"))
# pickle.dump(le, open("models/label_encoder.pkl", "wb"))

# print("Model saved successfully.")
import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# 1️⃣ Load Dataset
# =====================================================

data = pd.read_csv("Resume_Domain_Classifier/data/UpdatedResumeDataSet.csv")

print("Dataset Loaded Successfully.")
print("Total Samples:", len(data))


# =====================================================
# 2️⃣ Text Cleaning
# =====================================================

def clean_text(text):
    text = re.sub(r'http\\S+', ' ', text)           # ← DOUBLE BACKSLASH
    text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)    # ← DOUBLE BACKSLASH  
    text = re.sub(r'\\s+', ' ', text)               # ← DOUBLE BACKSLASH
    return text.lower().strip()
data["Resume"] = data["Resume"].apply(clean_text)


# =====================================================
# 3️⃣ Encode Labels
# =====================================================

label_encoder = LabelEncoder()
data["Category"] = label_encoder.fit_transform(data["Category"])


# =====================================================
# 4️⃣ Define Features & Target
# =====================================================

X = data["Resume"]
y = data["Category"]


# =====================================================
# 5️⃣ Train-Test Split (Stratified)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # Important for balanced class distribution
)


# =====================================================
# 6️⃣ TF-IDF Vectorizer (Improved Configuration)
# =====================================================

tfidf = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# =====================================================
# 7️⃣ Model (Better than LinearSVC for probabilities)
# =====================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)


# =====================================================
# 8️⃣ Evaluation
# =====================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =====================================================
# 9️⃣ Save Model Files
# =====================================================

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
pickle.dump(label_encoder, open("models/label_encoder.pkl", "wb"))

print("\nModel, TF-IDF, and Label Encoder saved successfully.")
