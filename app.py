import streamlit as st
import pandas as pd
import pickle # Changed from joblib to pickle
import numpy as np
# PAGE CONFIG---
st.set_page_config(page_title="Iris Classifier", page_icon="⭐")
# LOAD THE TRAINED MODEL
@st.cache_resource
def load_model():
    # Update the path to look for the pkl file
    model_path = "iris_model.pkl"
    if os.path.exists (model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"Model file '{model_path}' not found!")
        return None
model = load_model()
#UI INTERFACE
st.title("⭐ Iris Species Predictor")
st.markdown("""
This app uses a **Logistic Regression** model to predict the species of an Iris flower based on its physical mesurements
            """)

st.sidebar.header("Input floral Features")
