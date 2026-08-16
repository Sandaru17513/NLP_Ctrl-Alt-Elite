**Project title: Spam Email detection system**

**Group members**
- Member 1: [CIT-24-01-0182: Chamika Janith]
- Member 2: [CIT-24-01-0086: Dineth Ushira]
- Member 3: [CIT-24-01-0014: Sandaru Bhagya]

**Problem statement**
Spam emails can contain malvare, phishing scripts or unwanted advertisements. the goal of this project is to create a machine learning model and a deep learningg model to accurrately classify emails as spam or not spam.

**Dataset information**
The datasets used for this project is "Spam Email Datasets" from Kaggle, Github, Seven Phishing Email Datasets collction etc. which contains a collection of emails labeled as spam(spam) or not spam(ham). Sources: 
https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification/data, https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108, 
https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset, 
https://github.com/MWiechmann/enron_spam_data/tree/master, 
https://www.kaggle.com/datasets/tinu10kumar/sms-spam-dataset?resource=download, 
https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset 

## Setup instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation steps

1. **Clone or download the repository**
   ```bash
   cd "SLTC Large Assignments/NLP/NLP_Ctrl-Alt-Elite"
   ```

2. **Create and activate a virtual environment** (if not already created)
   ```powershell
   python -m venv src/venv
   cd src
   .\venv\Scripts\Activate.ps1
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages include:
   - Flask (web framework)
   - TensorFlow/Keras (LSTM model)
   - scikit-learn (Logistic Regression)
   - nltk (Natural Language Toolkit)
   - langdetect (Language detection)
   - pandas & numpy (Data processing)

## How to run the project

See for detailed step-by-step instructions.

Quick summary:
```powershell
cd src
.\venv\Scripts\Activate.ps1
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

### Project structure
```
NLP_Ctrl-Alt-Elite/
├── src/
│   ├── app.py                 # Flask web application
│   ├── index.html             # Web UI
│   └── venv/                  # Virtual environment
├── notebooks/
│   └── cit-24-01-0014/
│       ├── preprocessing.py   # Text preprocessing pipeline
│       ├── cit_24_01_0014_pipeline_&_lrmodel.ipynb  # Logistic Regression notebook
│       └── cit-24-01-0014_lstm.ipynb               # LSTM notebook
├── models/
│   └── cit-24-01-0014/
│       ├── logistic_regression/
│       │   ├── logistic_regression_model.pkl
│       │   └── tfidf_vectorizer.pkl
│       └── lstm/
│           ├── spam_lstm_model.keras
│           ├── tokenizer.pkl
│           └── best_threshold.npy
└── README.md
```

## Model summary

The spam classification system uses two complementary machine learning models:

### 1. Logistic Regression with TF-IDF
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model**: Logistic Regression
- **Input**: Preprocessed text (cleaned and lemmatized)
- **Output**: Binary classification (Ham/Spam) with confidence probability
- **Advantages**: Fast inference, interpretable, lightweight

### 2. LSTM (Long Short-Term Memory) Neural Network
- **Architecture**: Bidirectional LSTM with embedding layer
- **Input**: Tokenized sequences (padded to max length of 100)
- **Output**: Binary classification (Ham/Spam) with threshold-based decision
- **Optimal Threshold**: 0.313 (optimized during training)
- **Advantages**: Captures sequential patterns, handles variable-length inputs

### Text Preprocessing Pipeline
Both models use the same preprocessing pipeline:
1. **Language Detection**: Filter for English text only
2. **Regex Cleaning**: 
   - Convert to lowercase
   - Remove HTML tags, URLs, emails, numbers, punctuation
3. **POS-aware Lemmatization**: 
   - Part-of-speech tagging
   - Lemmatize words based on their grammatical role

## Results summary

### Model Performance
Both models work together to provide robust spam classification:

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| Logistic Regression | ~96% | High | High | High |
| LSTM | ~97% | High | High | High |

### Key Features
- **Ensemble Approach**: Combines linear (LR) and non-linear (LSTM) models for better generalization
- **Agreement Detection**: Web UI shows when both models agree or disagree
- **Confidence Scores**: Each prediction includes a confidence percentage
- **Real-time Classification**: Fast inference suitable for production use

### Example Predictions
- **Spam Example**: "Click here to claim your free prize now!"
  - LR: Spam (99.96% confidence)
  - LSTM: Spam (99.87% confidence)

- **Ham Example**: "Hey, let's grab coffee tomorrow afternoon?"
  - LR: Ham (98.08% confidence)
  - LSTM: Ham (99.91% confidence)

### Dataset
- **Total Samples**: Combined from multiple sources
- **Classes**: 2 (Spam and Ham)
- **Languages**: English only (filtered via language detection)
- **Sources**: Kaggle, GitHub, Phishing Email Datasets, and SMS collections

