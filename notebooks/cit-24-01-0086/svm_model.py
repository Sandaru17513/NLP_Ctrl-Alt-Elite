import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import preprocess_text
from feature_engineering import (
    create_tfidf,
    save_vectorizer
)


# =============================
# Load Dataset
# =============================

df = pd.read_csv("../../data/dataset.csv")

print("\n========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nClasses:")
print(df["label"].value_counts())


# =============================
# Validate Dataset
# =============================

df = df.dropna(subset=["text", "label"])

df["label"] = df["label"].astype(int)

# Keep only binary labels
df = df[df["label"].isin([0, 1])]


# =============================
# Preprocessing
# =============================

print("\nPreprocessing text...")

df["clean_text"] = df["text"].apply(preprocess_text)

print("\nPreprocessed Samples:")
print(df[["label", "clean_text"]].head())


# Remove empty texts
df = df[df["clean_text"].str.strip() != ""]


# =============================
# Features and Labels
# =============================

X = df["clean_text"]
y = df["label"]


# =============================
# Train / Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# =============================
# TF-IDF
# =============================

vectorizer = create_tfidf()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Shape (Train):", X_train_tfidf.shape)
print("TF-IDF Shape (Test) :", X_test_tfidf.shape)


# =============================
# Save TF-IDF Vectorizer
# =============================

os.makedirs("../../models/cit-24-01-0086", exist_ok=True)

save_vectorizer(
    vectorizer,
    "../../models/cit-24-01-0086/tfidf.pkl"
)

print("\nTF-IDF vectorizer saved successfully!")


# =============================
# Train SVM Model
# =============================

print("\nTraining SVM model...")

model = SVC(
    kernel="linear",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# =============================
# Save SVM Model
# =============================

joblib.dump(
    model,
    "../../models/cit-24-01-0086/svm_model.pkl"
)

print("SVM model saved successfully!")


# =============================
# Predictions
# =============================

y_pred = model.predict(X_test_tfidf)


# =============================
# Evaluation
# =============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


print("\n========== MODEL RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


print("\n0 = HAM / NOT SPAM")
print("1 = SPAM")

print("\n========== TRAINING COMPLETED ==========")