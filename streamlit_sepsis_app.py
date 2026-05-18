"""Streamlit app for sepsis risk prediction with Neural ODE + Groq LLM explanations.

Prerequisites:
- Python 3.10+
- pip install streamlit groq torch torchdiffeq pandas numpy python-dotenv
- Place sepsis_inference.py and neural_ode_sepsis.pt in the same directory, or adjust paths.
- Create a .env file with GROQ_API_KEY set, e.g.:
    GROQ_API_KEY=your_groq_api_key_here

Run locally:
    streamlit run streamlit_sepsis_app.py
"""

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from groq import Groq
from dotenv import load_dotenv

from sepsis_inference import (
    load_neural_ode_v1_checkpoint,
    build_window_from_patient_df,
    build_patient_summary,
)

# Load environment variables from .env if present
load_dotenv(override=True)

# -----------------------------
# Cached model loading
# -----------------------------


@st.cache_resource
def load_model_and_meta(ckpt_path: str, device: str = "cpu"):
    device_obj = torch.device(device)
    model, meta = load_neural_ode_v1_checkpoint(ckpt_path=ckpt_path, device=device_obj)
    return model, meta, device_obj


def predict_sepsis_streamlit(
    model: torch.nn.Module,
    meta: dict,
    df_patient: pd.DataFrame,
    device: torch.device,
    window_size: int = 24,
) -> float:
    t_tensor, x_tensor = build_window_from_patient_df(
        df_patient,
        meta,
        window_size=window_size,
        device=device,
    )
    with torch.no_grad():
        logits = model(t_tensor, x_tensor)
        prob = torch.sigmoid(logits).item()
    return prob


def get_groq_client_from_env() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        st.warning("GROQ_API_KEY not found in environment/.env; LLM explanations will be disabled.")
        return None
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None


def generate_llm_explanation(client: Groq, patient_summary: dict) -> str:
    """Call Groq chat completions API to generate an explanation."""
    prompt = f"""
You are an ICU decision support assistant.

Given the following model output and 24-hour summary of a patient's vitals and labs,
explain in 3–5 sentences why the model estimates this sepsis risk for the next {patient_summary.get('horizon_hours', 6)} hours.
Do not invent data; only use what is given. Speak in clear, clinician-friendly language.

JSON:
{json.dumps(patient_summary, indent=2)}
"""
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a careful, conservative ICU decision support assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return "(Error calling Groq API. See logs.)"


# -----------------------------
# Streamlit UI
# -----------------------------


def main():
    st.set_page_config(page_title="Sepsis Risk (Neural ODE + Groq)", layout="wide")
    st.title("Sepsis Risk Prediction (Neural ODE + LLM Explanations)")

    st.markdown(
        "This app uses a Neural ODE model trained on PhysioNet-style sepsis data to "
        "estimate sepsis risk in the next few hours, and a Groq-hosted LLM to generate "
        "clinician-friendly explanations."
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    default_model_path = "./neural_ode_sepsis.pt"
    model_path = st.sidebar.text_input(
        "Neural ODE checkpoint path",
        value=default_model_path,
        help="Path to neural_ode_sepsis.pt",
    )

    device_choice = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)

    # Load model
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint not found at: {model_path}")
        st.stop()

    with st.spinner("Loading model..."):
        model, meta, device = load_model_and_meta(model_path, device_choice)

    st.sidebar.success("Model loaded.")

    # Input mode selection
    st.header("Input patient data")
    mode = st.radio("Choose input mode", ["Upload CSV", "Manual input (single snapshot)"])

    df_patient = None

    if mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload Dataset.csv or a single-patient CSV", type=["csv"], accept_multiple_files=False
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            patient_col = meta["patient_col"]
            time_col = meta["time_col"]

            if patient_col in df.columns:
                patient_ids = sorted(df[patient_col].astype(str).unique())
                selected_pid = st.selectbox("Select patient ID", patient_ids)
                df_patient = df[df[patient_col].astype(str) == selected_pid].copy()
            else:
                st.info(
                    "Patient_ID column not found; assuming this CSV contains a single patient's time series."
                )
                df_patient = df.copy()

    else:  # Manual input
        st.info(
            "Manual input mode: you enter a single time snapshot for each feature. "
            "The model will internally create a 24-hour window by repeating these values."
        )

        feature_cols = meta["feature_cols"]
        time_col = meta["time_col"]
        patient_col = meta["patient_col"]

        with st.form("manual_input_form"):
            st.subheader("Enter latest measurements for each feature")
            values = {}
            cols = st.columns(3)
            for i, feat in enumerate(feature_cols):
                with cols[i % 3]:
                    default_val = 0.0
                    val = st.number_input(feat, value=float(default_val))
                    values[feat] = val

            submitted = st.form_submit_button("Use these values")

        if submitted:
            # Build a synthetic DataFrame with a single row and required columns
            row = {feat: values[feat] for feat in feature_cols}
            row[time_col] = 0.0
            row[patient_col] = "manual_1"
            df_patient = pd.DataFrame([row])
            st.write("Constructed patient snapshot:")
            st.dataframe(df_patient)

    # Once df_patient is available, run prediction
    if df_patient is not None and not df_patient.empty:
        st.header("Prediction & Explanation")

        if st.button("Run sepsis risk prediction"):
            with st.spinner("Running Neural ODE model..."):
                risk_score = predict_sepsis_streamlit(model, meta, df_patient, device=device, window_size=24)

            st.metric("Sepsis risk (next 6 hours)", f"{risk_score:.4f}")

            # Build summary for display and LLM
            patient_summary = build_patient_summary(
                df_patient,
                risk_score=risk_score,
                meta=meta,
                horizon_hours=6,
                window_hours=24,
            )

            st.subheader("Patient Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Patient ID", str(patient_summary.get("patient_id", "N/A")))
            with col2:
                age = patient_summary.get("demographics", {}).get("age")
                st.metric("Age", f"{age:.0f}" if pd.notnull(age) else "N/A")
            with col3:
                sex = patient_summary.get("demographics", {}).get("sex")
                if pd.notnull(sex):
                    sex = "Male" if sex == 1 else "Female"
                st.metric("Sex", str(sex) if pd.notnull(sex) else "N/A")
            with col4:
                st.metric("Observation Window", f"{patient_summary.get('time_window_hours', 24)}h")
                
            st.markdown("**Vitals & Labs Summary**")
            vitals = patient_summary.get("vitals_summary", {})
            if vitals:
                vitals_df = pd.DataFrame.from_dict(vitals, orient='index')
                if not vitals_df.empty:
                    vitals_df = vitals_df.rename(columns={
                        "last": "Latest Value", 
                        "min": "Minimum", 
                        "max": "Maximum", 
                        "trend": "Trend"
                    })
                    for c in ["Latest Value", "Minimum", "Maximum"]:
                        if c in vitals_df.columns:
                            vitals_df[c] = vitals_df[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(vitals_df, use_container_width=True)
            else:
                st.info("No vitals summary available.")

            # LLM explanation via Groq
            client = get_groq_client_from_env()
            if client is None:
                st.info("GROQ_API_KEY not set; skipping LLM explanation.")
            else:
                with st.spinner("Requesting explanation from Groq LLM..."):
                    explanation = generate_llm_explanation(client, patient_summary)
                st.subheader("LLM-generated explanation")
                st.write(explanation)


if __name__ == "__main__":
    main()