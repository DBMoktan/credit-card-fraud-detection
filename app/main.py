import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API to predict whether a credit card transaction is fraudulent using an XGBoost model.",
    version="1.0.0"
)

# Load the trained model at startup
MODEL_PATH = "models/xgboost_fraud_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"[SUCCESS] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None

# Pydantic schema for input validation
class Transaction(BaseModel):
    scaled_amount: float
    scaled_time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API. Use the /predict endpoint to make predictions."}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please check server logs.")
    
    # Convert input to DataFrame for prediction
    # Maintain exact column order as seen in preprocessing: scaled_amount, scaled_time, V1-V28
    input_df = pd.DataFrame([transaction.dict()])
    
    try:
        # Predict probability
        proba = model.predict_proba(input_df)[0, 1]
        
        # Predict class (0: Normal, 1: Fraud)
        prediction = int(model.predict(input_df)[0])
        
        return {
            "prediction": prediction,
            "probability": float(proba),
            "status": "Fraud" if prediction == 1 else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
