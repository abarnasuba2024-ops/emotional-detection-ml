import streamlit as st
import pickle


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# -----------------------------------
# Load ML Model
# -----------------------------------

with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------------
# Emotion Responses
# -----------------------------------

responses = {
    "happy": {
        "emoji": "😊",
        "message": "You seem happy! That's wonderful. Keep enjoying this positive moment! 🎉"
    },

    "sad": {
        "emoji": "😢",
        "message": "You seem sad. It's okay to feel this way. Take some time for yourself."
    },

    "angry": {
        "emoji": "😡",
        "message": "You seem angry. Take a deep breath and give yourself a moment before reacting."
    },

    "fear": {
        "emoji": "😨",
        "message": "You seem worried or scared. Take a deep breath and focus on what you can control."
    },

    "surprise": {
        "emoji": "😲",
        "message": "You seem surprised! That sounds unexpected!"
    },

    "love": {
        "emoji": "❤️",
        "message": "You seem to be expressing love and affection. That's a beautiful feeling!"
    },

    "neutral": {
        "emoji": "🙂",
        "message": "I don't detect a strong emotion in your text."
    }
}


# -----------------------------------
# Application UI
# -----------------------------------

st.title("😊 Emotion Detection from Text")

st.write(
    "Enter any sentence below and the Machine Learning model "
    "will predict the emotion."
)

st.info(
    "Machine Learning Algorithm: Logistic Regression\n\n"
    "NLP Technique: TF-IDF Vectorization"
)


# -----------------------------------
# Text Input
# -----------------------------------

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I am very happy today!",
    height=150
)


# -----------------------------------
# Analyze Button
# -----------------------------------

if st.button("🔍 Analyze Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Convert text into TF-IDF features
        text_tfidf = vectorizer.transform([text])

        # Predict emotion
        prediction = model.predict(text_tfidf)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(text_tfidf)[0]

        confidence = max(probabilities) * 100

        # Get response
        result = responses.get(
            prediction,
            {
                "emoji": "🙂",
                "message": "I could not determine a strong emotion."
            }
        )

        # -----------------------------------
        # Display Result
        # -----------------------------------

        st.success("Emotion detected successfully!")

        st.markdown(
            f"# {result['emoji']} {prediction.capitalize()}"
        )

        st.write(result["message"])

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        # -----------------------------------
        # Show Probability
        # -----------------------------------

        st.subheader("Emotion Probability")

        probability_data = {
            emotion.capitalize(): round(prob * 100, 2)
            for emotion, prob in zip(
                model.classes_,
                probabilities
            )
        }

        st.bar_chart(probability_data)