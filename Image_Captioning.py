from transformers import pipeline
import requests
from PIL import Image
import torch
import streamlit as st

st.title("Image Captioning Application")

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

url=st.text_input("Enter the url of the image")


Captioning=st.button("Captioning")
if Captioning and url is not None:
    try:
        st.image(url)
        image = Image.open(requests.get(url, stream=True).raw)
        res=image_to_text(url)[0]['generated_text']
        st.title(str(res).capitalize())
    except:
        st.alert("Please enter a valid url")  


