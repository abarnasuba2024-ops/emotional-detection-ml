import streamlit as st
import pickle
import re
import pandas as pd


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    with open("emotion_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================
# EMOTION EMOJIS
# =========================

emotion_emojis = {

    "anger": "😡",

    "fear": "😨",

    "joy": "😄",

    "love": "❤️",

    "sadness": "😢",

    "surprise": "😲"
}


# =========================
# TITLE
# =========================

st.title("😊 Emotion Detection in Text")

st.write(
    "Enter a sentence and the machine learning model "
    "will detect the emotion."
)

st.divider()


# =========================
# TEXT INPUT
# =========================

user_text = st.text_area(

    "Enter your text:",

    placeholder="Example: I am very happy today!",

    height=150
)


# =========================
# PREDICT BUTTON
# =========================

if st.button(
    "🔍 Predict Emotion",
    use_container_width=True
):

    if user_text.strip() == "":

        st.warning(
            "⚠️ Please enter some text."
        )

    else:

        # Clean text

        cleaned_text = clean_text(user_text)


        # Convert text to TF-IDF

        text_vector = vectorizer.transform(
            [cleaned_text]
        )


        # Prediction

        prediction = model.predict(
            text_vector
        )[0]


        # =========================
        # RESULT
        # =========================

        st.subheader("Prediction Result")

        emoji = emotion_emojis.get(
            prediction,
            "😊"
        )

        st.success(
            f"{emoji} Detected Emotion: "
            f"**{prediction.upper()}**"
        )


        # =========================
        # PROBABILITY GRAPH
        # =========================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                text_vector
            )[0]

            classes = model.classes_


            # Create dataframe

            probability_data = pd.DataFrame({

                "Emotion": classes,

                "Probability": probabilities

            })


            # Convert to percentage

            probability_data["Probability"] = (
                probability_data["Probability"] * 100
            )


            # Sort highest probability first

            probability_data = (
                probability_data
                .sort_values(
                    "Probability",
                    ascending=False
                )
            )


            st.subheader(
                "📊 Emotion Probability"
            )


            # Show graph

            st.bar_chart(

                probability_data.set_index(
                    "Emotion"
                )["Probability"]
            )


            # =========================
            # PROBABILITY TABLE
            # =========================

            st.subheader(
                "📈 Emotion Scores"
            )

            display_data = probability_data.copy()

            display_data["Probability"] = (
                display_data["Probability"]
                .round(2)
                .astype(str)
                + "%"
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


# =========================
# MODEL INFORMATION
# =========================

st.divider()

st.subheader("📌 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.write(
        "**Algorithm:** "
        "TF-IDF + Logistic Regression"
    )

    st.write(
        "**Emotions:** 6"
    )


with col2:

    st.write(
        "**Accuracy:** 88.8%"
    )

    st.write(
        "**Dataset:** Emotion Dataset for NLP"
    )


# =========================
# SUPPORTED EMOTIONS
# =========================

st.subheader(
    "🎭 Supported Emotions"
)

st.write(
    "😡 Anger   |   😨 Fear   |   😄 Joy   |   "
    "❤️ Love   |   😢 Sadness   |   😲 Surprise"
)