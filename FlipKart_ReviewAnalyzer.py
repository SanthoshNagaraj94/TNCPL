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

st.title("FlipKart Review Analyzer")

#upload csv
uploaded_file = st.file_uploader("Choose a file")



#save csv
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data=df["Review"].to_list()

    Data=list(map(lambda x:Classifier(x),data))

    df["Category"]=Data

    st.dataframe(df)



