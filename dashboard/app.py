import sys
import os

# Dynamically add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd

from src.data_loader import load_and_preprocess_data
from src.model_trainer import load_models
from src.shap_explainer import explain_with_shap_logistic, explain_with_shap_rf, plot_shap_summary, plot_shap_waterfall
from src.lime_explainer import explain_instance_with_lime


# Streamlit page setup
st.set_page_config(page_title="AI Model Explainability Dashboard", layout="wide")

st.title("🧠 AI Model Explainability Dashboard")
st.markdown("""
Welcome to the **AI Model Explainability Dashboard**!  
Explore how machine learning models make decisions using **SHAP** and **LIME** visual explanations.

🔍 Dataset: Credit Card Default Prediction  
📊 Models: Logistic Regression | Random Forest  
🛠️ Explanations: Global (SHAP) & Local (SHAP, LIME)

---
""")

# File 
st.info("Using default credit card dataset.")

csv_path = "data/credit_card_default.csv"


# Load and preprocess data
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = load_and_preprocess_data(csv_path)

# Train models
logreg, rf = load_models()

# Model selection
model_choice = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Random Forest"))
explanation_type = st.sidebar.selectbox("Choose Explanation Method", ("SHAP", "LIME"))
sample_index = st.sidebar.slider("Select Test Sample to Explain", min_value=0, max_value=len(X_test) - 1, value=0)

# Run explanations based on selection
if st.sidebar.button("Generate Explanation"):
    if model_choice == "Logistic Regression":
        model = logreg
        X_test_input = X_test_scaled
    else:
        model = rf
        X_test_input = X_test

    if explanation_type == "SHAP":
        if model_choice == "Logistic Regression":
            explainer, shap_values = explain_with_shap_logistic(model, X_train_scaled, X_test_input, feature_names)
            st.subheader("SHAP Summary Plot (Global Explanation)")
            plot_shap_summary(shap_values, X_test_input, feature_names)


            st.subheader("SHAP Waterfall Plot (Local Explanation)")
            plot_shap_waterfall(shap_values, sample_index)

        else:
            explainer, shap_values = explain_with_shap_rf(model, X_train, X_test)
            st.subheader("SHAP Summary Plot (Global Explanation)")
            plot_shap_summary(shap_values[1], X_test[:100], feature_names)

            st.subheader("SHAP Waterfall Plot (Local Explanation)")
            plot_shap_waterfall(shap_values[1], sample_index)

    elif explanation_type == "LIME":
        st.subheader("LIME Explanation (Local Explanation)")
        lime_exp = explain_instance_with_lime(model, X_train_scaled, X_test_scaled, feature_names, sample_index)

        for feature, weight in lime_exp:
            st.write(f"{feature}: {weight:+.4f}")
