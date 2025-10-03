import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('news.csv')

# Prepare the data
df['combined'] = df['title'] + " " + df['text']
df['label'] = df['label'].str.strip()

X = df['combined']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train the PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Streamlit App UI
st.title("📰 AI Fake News Detector")
st.subheader("Enter a news article to check if it's REAL or FAKE")

user_input = st.text_area("Paste your news article here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        if prediction == "REAL":
            st.success("✅ The news article is **REAL**.")
        else:
            st.error("🚨 The news article is **FAKE**.")

# Optional: Show metrics
if st.checkbox("Show Model Accuracy & Confusion Matrix"):
    y_pred = model.predict(tfidf_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"**Accuracy**: {round(acc * 100, 2)}%")
    st.write("**Confusion Matrix**:")
    st.write(cm)
