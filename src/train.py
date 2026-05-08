import pandas as pd
import joblib
import os
import time
from xgboost import XGBClassifier

# Define paths
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"

def train_model():
    """Loads preprocessed data, trains the champion XGBoost model, and saves it."""
    print("Loading SMOTE-balanced training data...")
    X_train_smote = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_smote.pkl'))
    y_train_smote = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_smote.pkl'))
    
    # We use the optimal hyperparameters found during Notebook 03 tuning
    print("Initializing Champion XGBoost Model...")
    xgb_model = XGBClassifier(
        n_estimators=200,          # Optimal number of trees
        learning_rate=0.1,         # Step size
        max_depth=5,               # Depth of trees
        subsample=0.8,             # Prevent overfitting
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1                  # Use all CPU cores
    )
    
    print("Training model (this may take ~30 seconds)...")
    start_time = time.time()
    xgb_model.fit(X_train_smote, y_train_smote)
    train_time = time.time() - start_time
    print(f"✅ Training complete in {train_time:.2f} seconds!")
    
    # Save the trained model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'xgboost_fraud_model.pkl')
    
    print(f"Saving serialized model to {model_path}...")
    joblib.dump(xgb_model, model_path)
    print("🚀 Model successfully saved and ready for Deployment!")

if __name__ == "__main__":
    train_model()
