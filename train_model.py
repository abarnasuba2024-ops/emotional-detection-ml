import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Number of records:", len(data))

print("\nEmotion distribution:")
print(data["emotion"].value_counts())


# -----------------------------------
# 2. Separate Input and Output
# -----------------------------------

X = data["text"]
y = data["emotion"]


# -----------------------------------
# 3. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# 4. TF-IDF Vectorization
# -----------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# 5. Train ML Model
# -----------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# 6. Test Model
# -----------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# 7. Save Model
# -----------------------------------

with open("emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\n-----------------------------------")
print("Training completed successfully!")
print("emotion_model.pkl created")
print("vectorizer.pkl created")
print("-----------------------------------")