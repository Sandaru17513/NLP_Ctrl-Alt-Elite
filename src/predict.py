import joblib

from preprocessing import preprocess_text
from feature_engineering import load_vectorizer


# Load trained model
model = joblib.load("models/svm_model.pkl")

# Load TF-IDF vectorizer
vectorizer = load_vectorizer()


print("========== Spam Email Detection ==========\n")

email = input("Enter Email Text:\n\n")

# Preprocess
clean_text = preprocess_text(email)

# Convert to TF-IDF
email_vector = vectorizer.transform([clean_text])

# Predict
prediction = model.predict(email_vector)

print("\nPrediction:")

if prediction[0] == 1:
    print("🚨 SPAM EMAIL")
else:
    print("✅ HAM EMAIL")