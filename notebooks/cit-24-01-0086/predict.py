import joblib

from preprocessing import preprocess_text
from feature_engineering import load_vectorizer


# =============================
# Load trained model
# =============================

model = joblib.load(
    "../../models/cit-24-01-0086/svm_model.pkl"
)

# Load trained TF-IDF vectorizer
vectorizer = load_vectorizer(
    "../../models/cit-24-01-0086/tfidf.pkl"
)


# =============================
# Email Prediction
# =============================

print("========== Spam Email Detection ==========\n")

email = input("Enter Email Text:\n\n")


# Preprocess input
clean_text = preprocess_text(email)


# Convert text to TF-IDF
email_vector = vectorizer.transform([clean_text])


# Predict
prediction = model.predict(email_vector)

decision_score = model.decision_function(email_vector)
print("Decision Score:", decision_score[0])


# =============================
# Display Binary Prediction
# =============================

print("\nPrediction:")
print(int(prediction[0]))

print("\n0 = HAM / NOT SPAM")
print("1 = SPAM")