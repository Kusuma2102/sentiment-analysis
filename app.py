import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax
import torch
from optimum.bettertransformer import BetterTransformer
import numpy as np

# Load models with caching
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = BetterTransformer.transform(model)
    return tokenizer, config, model

# Preprocess function
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Sentiment/Emotion analysis function
def analyze(texts, model_type):
    if model_type == "Sentiment":
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    else:
        MODEL = "SamLowe/roberta-base-go_emotions"

    tokenizer, config, model = load_model(MODEL)
    labels = config.id2label
    results = []

    for text in texts:
        text = preprocess(str(text))
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        result = {labels[i]: scores[i] for i in range(len(scores))}
        results.append(result)

    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("💬 Real-time Comment Analyzer (Sentiment & Emotion)")

model_choice = st.radio("Choose a model", ["Sentiment", "Emotion"])
input_type = st.radio("Input type", ["Paste text", "Upload CSV"])

if input_type == "Paste text":
    user_input = st.text_area("Enter comment text here")
    if st.button("Analyze"):
        if user_input:
            result = analyze([user_input], model_choice)
            st.write(result)
        else:
            st.warning("Please enter some text.")
else:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'CommentText' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'CommentText' not in df.columns:
            st.error("The uploaded CSV must contain a 'CommentText' column.")
        else:
            if st.button("Analyze All Comments"):
                result = analyze(df['CommentText'].tolist(), model_choice)
                full_df = pd.concat([df, result], axis=1)
                st.write(full_df.head())
                csv = full_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "analysis_results.csv", "text/csv")
