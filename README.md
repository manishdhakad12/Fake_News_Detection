# Fake News Detection (Machine Learning)

A machine learning project to detect whether a news article is **fake** or **real** using Natural Language Processing (NLP) techniques.  
This project uses **TF-IDF vectorization** and a **Passive Aggressive Classifier** to build a binary classification model.

---

## Project Overview

Fake news is a growing problem in the digital age. This project demonstrates how we can use **machine learning** to classify news articles into two categories:

- **FAKE** – Misleading or fabricated news
- **REAL** – Verified and legitimate news

The project includes:
- **Data preprocessing** (cleaning text, removing stopwords)
- **Feature extraction** using TF-IDF
- **Model training** with a Passive Aggressive Classifier
- **Evaluation** using accuracy score and confusion matrix

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Libraries:**  
  - `pandas` – Data handling  
  - `scikit-learn` – TF-IDF, model training, evaluation  
  - `streamlit` – Simple UI for trying the model interactively  

---
##  How to Run

1. **Clone the repository**
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection

2.  **Install dependencies**
pip install -r requirements.txt

3. **Run the Jupyter Notebook**
jupyter notebook fake_news_detection.ipynb

4. **(Optional) Run with Streamlit**
streamlit run app.py
