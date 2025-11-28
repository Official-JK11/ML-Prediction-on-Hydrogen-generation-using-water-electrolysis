import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load saved model
# -----------------------------
MODEL_PATH = "hydrogen_multioutput_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
targets = bundle["targets"]

st.set_page_config(page_title="Hydrogen Generation & Efficiency Predictor",
                   layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🔋 Hydrogen Generation & Efficiency Prediction Dashboard")
st.write("This dashboard predicts **H₂ Flow (Nm³/h)** and **Efficiency (%)** "
         "based on electrolysis operating conditions.")

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
st.header("📥 Input Operating Parameters")

input_values = {}

cols = st.columns(3)
for idx, feat in enumerate(features):
    with cols[idx % 3]:
        value = st.number_input(
            f"{feat}",
            value=0.0,
            format="%.4f"
        )
        input_values[feat] = value

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Hydrogen Output & Efficiency"):
    
    # Convert to model input
    row = np.array([input_values[f] for f in features]).reshape(1, -1)
    pred = model.predict(row)[0]

    # Display Results
    st.header("📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="💧 Hydrogen Generation (Nm³/h)", value=f"{pred[0]:.3f}")

    with col2:
        st.metric(label="⚡ Efficiency (%)", value=f"{pred[1]:.3f}")

    # -----------------------------
    # Feature Importance Plot
    # -----------------------------
    st.markdown("---")
    st.subheader("📌 Feature Importances (Random Forest)")
    
    rf = model.estimators_[0]
    importances = rf.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(features)[sorted_idx], importances[sorted_idx])
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.write("Built by **Jatin Gupta** for Minor Project on Hydrogen Generation Optimization.")
