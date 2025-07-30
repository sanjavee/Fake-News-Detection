# Fake News Detection

This project detects fake news using machine learning and natural language processing.

## Features

- Data preprocessing and cleaning
- Feature extraction (TF-IDF)
- Model training (Logistic Regression)
- Evaluation metrics (accuracy, precision, recall)
- Prediction on new articles via Streamlit web app

## Installation

```bash
git clone https://github.com/sanjavee/Fake-News-Detection
cd Fake-News-Detection
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset as `Fake.csv` and `True.csv` in the project folder.
2. Train the model and save artifacts:
    ```bash
    python fake_news_detector.py
    ```
    This will generate `vectorize.joblib` and `model.joblib`.
3. Launch the web app for predictions:
    ```bash
    streamlit run app.py
    ```
    Enter a news article in the text area to check if it is fake or real.

## Files

- `fake_news_detector.py`: Data cleaning, training, and model saving.
- `app.py`: Streamlit app for interactive fake news