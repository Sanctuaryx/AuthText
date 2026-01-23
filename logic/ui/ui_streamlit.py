import os
import requests
import streamlit as st

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8001")

st.title("AuthText Detector")

models = requests.get(f"{BASE}/models").json()
model = st.selectbox("Model", models["available"], index=models["available"].index(models["default"]))

text = st.text_area("Text", height=200)

if st.button("Predict"):
    r = requests.post(f"{BASE}/predict", json={"text": text, "model": model})
    st.code(r.text, language="json")
