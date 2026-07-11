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

# Load Dataset
df = pd.read_csv("data/email.csv")

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nClasses :")
print(df["Category"].value_counts())
# Keep only valid labels
df = df[df["Category"].isin(["ham", "spam"])]

# Preprocess messages
df["clean_text"] = df["Message"].apply(preprocess_text)

print("\nPreprocessed Samples:")
print(df[["Category", "clean_text"]].head())
# Convert labels to numbers
df["label"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

# Features and Labels
X = df["clean_text"]
y = df["label"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# TF-IDF
vectorizer = create_tfidf()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save TF-IDF Vectorizer
save_vectorizer(vectorizer)
print("TF-IDF vectorizer saved successfully!")

print("\nTF-IDF Shape (Train):", X_train_tfidf.shape)
print("TF-IDF Shape (Test):", X_test_tfidf.shape)
# -----------------------------
# Train SVM Model
# -----------------------------
model = SVC(kernel="linear", random_state=42)

model.fit(X_train_tfidf, y_train)

# Save trained SVM model
joblib.dump(model, "models/svm_model.pkl")

print("SVM model saved successfully!")

print("\nModel training completed!")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_tfidf)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))