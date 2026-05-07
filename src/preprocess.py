import pandas as pd
import os
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Define paths
RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DATA_DIR = "data/processed/"

def load_data(file_path):
    """Loads the raw dataset and drops duplicates."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicates. New shape: {df.shape}")
    return df

def scale_features(df):
    """Scales Amount and Time using RobustScaler."""
    print("Scaling 'Amount' and 'Time' features...")
    robust_scaler = RobustScaler()
    
    df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Reorder columns
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    return df

def process_data():
    """Main pipeline to process data, split, apply SMOTE, and save."""
    # 1. Load Data
    df = load_data(RAW_DATA_PATH)
    
    # 2. Scale Features
    df = scale_features(df)
    
    # 3. Split Data (Before SMOTE)
    print("Splitting data into Train and Test sets...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Apply SMOTE to Training Data ONLY
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Original training fraud count: {sum(y_train == 1)}")
    print(f"SMOTE training fraud count: {sum(y_train_smote == 1)}")
    
    # 5. Save Processed Data
    print("Saving processed data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    joblib.dump(X_train_smote, os.path.join(PROCESSED_DATA_DIR, 'X_train_smote.pkl'))
    joblib.dump(X_test, os.path.join(PROCESSED_DATA_DIR, 'X_test.pkl'))
    joblib.dump(y_train_smote, os.path.join(PROCESSED_DATA_DIR, 'y_train_smote.pkl'))
    joblib.dump(y_test, os.path.join(PROCESSED_DATA_DIR, 'y_test.pkl'))
    
    print("✅ Preprocessing pipeline complete! Data saved to data/processed/")

if __name__ == "__main__":
    process_data()
