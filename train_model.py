import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Load dataset
# =========================

def load_data(file_path):
    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Dataset format:
            # text;emotion
            parts = line.rsplit(";", 1)

            if len(parts) == 2:
                text = parts[0].strip()
                emotion = parts[1].strip()

                texts.append(text)
                labels.append(emotion)

    return pd.DataFrame({
        "text": texts,
        "emotion": labels
    })


# =========================
# 2. Load train and test
# =========================

print("Loading dataset...")

train_df = load_data("data/train.txt")
test_df = load_data("data/test.txt")

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nEmotion classes:")
print(train_df["emotion"].unique())


# =========================
# 3. Separate X and y
# =========================

X_train = train_df["text"]
y_train = train_df["emotion"]

X_test = test_df["text"]
y_test = test_df["emotion"]


# =========================
# 4. TF-IDF Vectorizer
# =========================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# =========================
# 5. Train ML model
# =========================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_vectorized, y_train)


# =========================
# 6. Test model
# =========================

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# 7. Save model
# =========================

with open("emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nFiles saved successfully:")
print("emotion_model.pkl")
print("vectorizer.pkl")