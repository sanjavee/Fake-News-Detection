import streamlit as st
import joblib
import re
import string
def clean_text(text):
    """Cleans and normalizes input text for further processing.
    
    This function removes unwanted characters, punctuation, numbers, and
    formatting from the input text to prepare it for analysis.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'\W', " ", text) 
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r'<.*?>+', "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', "", text)
    text = re.sub(r'\w*\d\w*', "", text)
    return text
vectorizer = joblib.load('vectorize.joblib')
model = joblib.load('model.joblib')

st.title("Fake News Detector")

input_text = st.text_area("Enter news Article")

if st.button("Check News"):
    if input_text.strip():
        cleaned_text = clean_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        if prediction[0] == 0:
            st.error("The News is Fake!")
        else:
            st.success("The News is Real!")
    else:
        st.warning("Please enter some text to analyze.")