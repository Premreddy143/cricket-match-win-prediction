"""
app.py
------
Streamlit deployment app for the IPL Match Winner Prediction model.
Run with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

st.set_page_config(page_title="IPL Match Winner Predictor", page_icon="🏏", layout="centered")

st.title("🏏 IPL Match Winner Prediction")
st.write("Predict the winner between two IPL teams based on venue and toss conditions.")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)):
    st.error(
        "No trained model found. Run the training pipeline first:\n\n"
        "```\npython data/generate_data.py\ncd src && python train_models.py\n```"
    )
    st.stop()

model, encoders = load_artifacts()
team_options = sorted(encoders["team1"].classes_.tolist())
venue_options = sorted(encoders["venue"].classes_.tolist())
city_options = sorted(encoders["city"].classes_.tolist())

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", team_options, index=0)
with col2:
    team2_choices = [t for t in team_options if t != team1]
    team2 = st.selectbox("Team 2", team2_choices, index=0)

venue = st.selectbox("Venue", venue_options)
city = st.selectbox("City", city_options)
toss_winner = st.radio("Toss Winner", [team1, team2], horizontal=True)
toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

if st.button("Predict Winner", type="primary"):
    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0  # fallback for unseen category

    features = pd.DataFrame([{
        "team1_enc": safe_encode(encoders["team1"], team1),
        "team2_enc": safe_encode(encoders["team2"], team2),
        "venue_enc": safe_encode(encoders["venue"], venue),
        "city_enc": safe_encode(encoders["city"], city),
        "toss_winner_is_team1": int(toss_winner == team1),
        "toss_decision_field": int(toss_decision == "field"),
    }])

    proba = model.predict_proba(features)[0]
    pred = model.predict(features)[0]
    winner = team1 if pred == 1 else team2
    confidence = max(proba)

    st.success(f"Predicted Winner: **{winner}**")
    st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.progress(float(proba[1]))
    st.caption(f"{team1}: {proba[1]*100:.1f}%  |  {team2}: {proba[0]*100:.1f}%")

st.divider()
st.caption(
    "Model trained on historical match data using Logistic Regression, "
    "Decision Tree, Random Forest, and XGBoost — best performer selected automatically."
)
