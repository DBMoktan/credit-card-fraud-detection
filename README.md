# Credit Card Fraud Detection System

An end-to-end Machine Learning pipeline to detect fraudulent credit card transactions. This project encompasses data engineering (handling extreme class imbalance), model training (XGBoost/Random Forest), a real-time REST API built with FastAPI, and an interactive Streamlit dashboard. The entire application is containerized using Docker and designed for cloud deployment.

## 📊 Dataset Setup

Due to GitHub's file size limits and best practices for data versioning, the raw dataset and trained models are **not** included in this repository. 

To run this project locally, you must download the dataset manually:

1. Visit the [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Download the archive and extract the `creditcard.csv` file.
3. Place the `creditcard.csv` file inside the `data/` folder in the root of this project.

```text
credit-card-fraud-detection/
└── data/
    └── creditcard.csv  <-- Place the downloaded file here
```

## 🧠 Models

Trained machine learning models (e.g., `.pkl` or `.joblib` files) are generated during the training phase. These files are saved in the `models/` directory. Like the dataset, these binary files are excluded from version control to prevent repository bloat. If you clone this repository, you will need to run the training scripts to generate your own local model files.

## 📂 Project Structure

This project follows a modular, professional architecture separating data, core logic, and application deployment.

```text
credit-card-fraud-detection/
├── app/                  # Deployment files (Frontend & Backend)
│   ├── dashboard.py      # Streamlit user interface
│   └── main.py           # FastAPI application
├── data/                 # Raw and processed datasets (Ignored by Git)
├── docker/               # Dockerfiles and container configurations
├── models/               # Saved model artifacts (.pkl, .joblib) (Ignored by Git)
├── notebooks/            # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                  # Core machine learning logic
│   ├── __init__.py
│   ├── preprocess.py     # Data cleaning and SMOTE implementation
│   └── train.py          # Model training and evaluation scripts
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DBMoktan/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # source venv/bin/activate    # On Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Data:** Follow the instructions in the *Dataset Setup* section above.
