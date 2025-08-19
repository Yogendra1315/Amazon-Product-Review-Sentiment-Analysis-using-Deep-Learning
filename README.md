# 🧠 Amazon Product Review Sentiment Analysis using Deep Learning

This project uses Natural Language Processing (NLP) and a Deep Learning model (LSTM) to classify Amazon product reviews as **Positive**, **Negative**, or **Neutral**.

---

## 📌 Problem Statement
Classify user-generated reviews into sentiment categories to help businesses understand customer feedback better.

---

## 📂 Dataset
- **Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Contains: Product reviews, ratings, and review text

---

## 🧼 Data Preprocessing
- Removed stopwords, punctuation, and special characters
- Tokenized and lemmatized review text
- Converted text to sequences using Keras Tokenizer
- Padded sequences to uniform input length

---

## 🧠 Model
- **Type**: LSTM (Long Short-Term Memory)
- **Framework**: Keras with TensorFlow backend
- Layers used:
  - Embedding Layer
  - LSTM Layer
  - Dense Output Layer with Softmax

---

## 🎯 Performance
- Accuracy: ~89% on test set
- Model evaluated using accuracy and confusion matrix

---

## 🛠️ Tools & Technologies
- Python, Pandas, Numpy
- NLTK for text preprocessing
- Keras/TensorFlow for model building
- Matplotlib & Seaborn for basic visualization

---

## 🚀 Future Work
- Add real-time sentiment prediction
- Integrate a dashboard using Streamlit or Power BI

---
