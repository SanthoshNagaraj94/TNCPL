import pandas as pd
import streamlit as st
import requests

from transformers import pipeline

def Classifier(Sentence):
    classifier = pipeline("zero-shot-classification")
    result=classifier(
        Sentence,
        candidate_labels=["Camera", "Battery", "Perfomance","Display"],
    )
    return result['labels'][0]

def Sentiment_Analyzer(Sentence):
    classifier = pipeline("sentiment-analysis")
    result=classifier(Sentence)
    return result

st.title("FlipKart Review Analyzer")

#upload csv

review=st.text_input("Enter your review")

if st.button("Analyze"):
    if review is not None:
        classifi=Classifier(review)
        sentiment=Sentiment_Analyzer(review)
        st.write(classifi,sentiment)



