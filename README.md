# 😊 Emotion Detection in Text Using Machine Learning and NLP

## 📌 Project Description

Emotion Detection in Text is a Natural Language Processing (NLP) and Machine Learning project that identifies the emotion expressed in a given text.

The system uses **TF-IDF Vectorization** to convert text into numerical features and a **Logistic Regression** classifier to predict the emotion.

The application supports six emotions:

- 😡 Anger
- 😨 Fear
- 😄 Joy
- ❤️ Love
- 😢 Sadness
- 😲 Surprise

The trained model is integrated with a **Streamlit web application** that provides real-time emotion prediction along with an emotion probability graph.

## 🚀 Features

- Text-based emotion detection
- NLP text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Six emotion categories
- Emotion probability visualization
- Interactive Streamlit interface
- GitHub version control
- Render deployment

## 🛠️ Tech Stack

- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Logistic Regression
- Pandas
- NumPy
- Streamlit
- Git
- GitHub
- Render

## 📂 Project Structure

```text
Emotional_Detection/
│
├── data/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
│
├── app.py
├── train_model.py
├── emotion_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md