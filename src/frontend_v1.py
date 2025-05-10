import os
import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import DATA_DIR
from inference import (
    get_model_predictions,
    load_batch_of_features_from_store,
    load_model_from_registry,
)

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(layout="wide")
st.title("🚲 Citi Bike Trip Demand Forecast")
current_date = pd.Timestamp.now(tz="UTC")
st.subheader(f"Prediction for {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# -----------------------------
# Progress bar setup
# -----------------------------
progress_bar = st.sidebar.header("Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4

# -----------------------------
# Step 1: Load features
# -----------------------------
with st.spinner("📦 Fetching batch of features..."):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.success("✅ Features loaded")
    progress_bar.progress(1 / N_STEPS)

# -----------------------------
# Step 2: Load model
# -----------------------------
with st.spinner("🧠 Loading model from Hopsworks..."):
    model = load_model_from_registry()
    st.sidebar.success("✅ Model loaded")
    progress_bar.progress(2 / N_STEPS)

# -----------------------------
# Step 3: Run inference
# -----------------------------
with st.spinner("🔮 Predicting demand..."):
    predictions = get_model_predictions(model, features)
    st.sidebar.success("✅ Predictions computed")
    progress_bar.progress(3 / N_STEPS)

# -----------------------------
# Step 4: Display predictions
# -----------------------------
with st.spinner("📊 Displaying results..."):
    st.subheader("🔢 Top 10 Stations by Predicted Demand")
    top10 = predictions.sort_values("predicted_demand", ascending=False).head(10)
    st.dataframe(top10)

    st.subheader("📈 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rides", f"{predictions['predicted_demand'].mean():.1f}")
    col2.metric("Max Rides", f"{predictions['predicted_demand'].max():.0f}")
    col3.metric("Min Rides", f"{predictions['predicted_demand'].min():.0f}")

    st.sidebar.success("✅ Visualization done")
    progress_bar.progress(4 / N_STEPS)