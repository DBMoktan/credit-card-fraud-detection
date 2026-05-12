import streamlit as st
import pandas as pd
import requests
import joblib
import os
import random

# Constants
# When running in Docker, we can point to localhost since both apps share the same container
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
PROCESSED_DATA_DIR = "data/processed/"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="💳")

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("This dashboard connects to a FastAPI backend running an XGBoost model to predict fraudulent credit card transactions in real-time.")

with st.expander("ℹ️ Understanding the Data (Why 'V1-V28'?)", expanded=False):
    st.markdown("""
    Due to confidentiality issues, the original features of the credit card transactions cannot be shown. 
    Instead, the data has gone through a mathematical transformation called **PCA (Principal Component Analysis)**.
    
    * **V1 to V28**: These are the anonymized PCA features. Their exact meaning is hidden to protect user privacy, but they typically contain values ranging between -5 and 5.
    * **Scaled Amount**: The transaction amount (e.g., dollars). It has been scaled (normalized) so extreme values don't confuse the model.
    * **Scaled Time**: The seconds elapsed since the first recorded transaction, also scaled.
    
    *Since manually typing 30 abstract numbers is difficult, we recommend using the **'Load Sample'** buttons on the sidebar to test the model with real data!*
    """)

# Attempt to load test data for sample generation
@st.cache_data
def load_test_data():
    try:
        X_test = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.pkl'))
        y_test = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.pkl'))
        return X_test, y_test
    except Exception as e:
        st.warning(f"Could not load test data for samples: {e}")
        return None, None

X_test, y_test = load_test_data()

st.sidebar.header("Settings & Inputs")

# Helper function to generate a sample
def set_sample(fraud_type):
    if X_test is not None and y_test is not None:
        if fraud_type == "Normal":
            indices = y_test[y_test == 0].index
        else:
            indices = y_test[y_test == 1].index
            
        if len(indices) > 0:
            idx = random.choice(indices)
            sample = X_test.loc[idx]
            # Update session state with the selected sample
            st.session_state['scaled_amount'] = float(sample['scaled_amount'])
            st.session_state['scaled_time'] = float(sample['scaled_time'])
            for i in range(1, 29):
                st.session_state[f'V{i}'] = float(sample[f'V{i}'])
            st.session_state['sample_loaded'] = True
            st.session_state['true_class'] = "Fraud" if fraud_type == "Fraud" else "Normal"

st.sidebar.subheader("Quick Actions")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Load Normal Sample"):
        set_sample("Normal")
with col2:
    if st.button("Load Fraud Sample"):
        set_sample("Fraud")

if 'sample_loaded' in st.session_state and st.session_state['sample_loaded']:
    st.sidebar.success(f"Loaded a real {st.session_state['true_class']} transaction from the test set.")

st.sidebar.markdown("---")
st.sidebar.subheader("Transaction Features")
st.sidebar.caption("Hover over the ❓ icons for details on what values to input.")

# Create default values in session state if not present
if 'scaled_amount' not in st.session_state:
    st.session_state['scaled_amount'] = 0.0
if 'scaled_time' not in st.session_state:
    st.session_state['scaled_time'] = 0.0
for i in range(1, 29):
    if f'V{i}' not in st.session_state:
        st.session_state[f'V{i}'] = 0.0

# Input fields
scaled_amount = st.sidebar.number_input(
    "Scaled Amount", 
    value=st.session_state['scaled_amount'], 
    format="%.4f",
    help="Normalized transaction amount. Original values (like $150) are scaled to center around 0. Typically ranges from -1.0 to 10.0."
)
scaled_time = st.sidebar.number_input(
    "Scaled Time", 
    value=st.session_state['scaled_time'], 
    format="%.4f",
    help="Normalized elapsed time. Scaled to center around 0. Typically ranges from -1.0 to 1.0."
)

# Update session state
st.session_state['scaled_amount'] = scaled_amount
st.session_state['scaled_time'] = scaled_time

with st.sidebar.expander("Anonymized Features (V1 - V28)"):
    st.caption("Principal Components. Most values range between -3.0 and +3.0.")
    for i in range(1, 29):
        val = st.number_input(
            f"V{i}", 
            value=st.session_state[f'V{i}'], 
            format="%.4f",
            help=f"Anonymized PCA feature {i} representing hidden transaction behavior."
        )
        st.session_state[f'V{i}'] = val

# Main content area
st.subheader("Transaction Analysis")

# Build the payload
payload = {
    "scaled_amount": st.session_state['scaled_amount'],
    "scaled_time": st.session_state['scaled_time']
}
for i in range(1, 29):
    payload[f'V{i}'] = st.session_state[f'V{i}']

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Input Data Overview")
    st.dataframe(pd.DataFrame([payload]).T.rename(columns={0: "Value"}))

with col2:
    st.markdown("### Prediction")
    if st.button("Predict Fraud", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction via API..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['prediction'] == 1:
                        st.error(f"🚨 **FRAUD DETECTED!** 🚨")
                        st.markdown(f"**Confidence:** {result['probability']*100:.2f}%")
                        st.progress(result['probability'])
                    else:
                        st.success(f"✅ **TRANSACTION NORMAL** ✅")
                        st.markdown(f"**Fraud Probability:** {result['probability']*100:.2f}%")
                        st.progress(result['probability'])
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ **Could not connect to the API.** Please ensure the FastAPI backend is running.")

st.markdown("---")
st.markdown("### Instructions")
st.markdown("""
1. Make sure your FastAPI backend is running in the terminal (`python -m uvicorn app.main:app --reload`).
2. Use the 'Load Sample' buttons on the sidebar to quickly populate realistic transaction data.
3. Click **Predict Fraud** to send the data to the API and receive a real-time prediction.
""")
