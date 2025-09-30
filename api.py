import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

# IMPORTANT: We need to import the custom KlarnaDataProcessor class 
# from your src directory so that the model pipeline can load correctly.
sys.path.insert(0, './src') 
from preprocess import KlarnaDataProcessor


# --- Pydantic Model for Input Data ---
# This defines the expected structure of a single loan application payload.
# You need to adjust these fields to match the *raw* features needed 
# by your KlarnaDataProcessor class.

# --- Pydantic Model for Input Data ---
class LoanData(BaseModel):
    """
    Defines the data structure for a single loan application.
    FastAPI uses this to automatically validate incoming JSON data.
    """
    loan_issue_date: str
    amount_outstanding_14d: float
    amount_outstanding_21d: float
    loan_id: str
    loan_amount: float
    card_expiry_month: Union[int, None] = None
    card_expiry_year: Union[int, None] = None
    existing_klarna_debt: Union[float, int, None] = None
    num_active_loans: int
    days_since_first_loan: Union[float, int, None] = None 
    new_exposure_7d: float
    new_exposure_14d: float
    num_confirmed_payments_3m: int
    num_confirmed_payments_6m: int
    num_failed_payments_3m: int
    num_failed_payments_6m: int
    num_failed_payments_1y: int
    amount_repaid_14d: float
    amount_repaid_1m: float
    amount_repaid_3m: float
    amount_repaid_6m: float
    amount_repaid_1y: float
    merchant_group: str 
    merchant_category: str
    

# --- Configuration & Initialization ---
app = FastAPI(
    title="Klarna Default Prediction API",
    description="Predicts the probability of default for a loan application.",
)



# Directory where the saved assets are located
MODEL_DIR = './01-notebooks'

# Asset paths
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# --- Global Variables for Assets ---
model: Any = None
scaler: Any = None
feature_names: List[str] = []

"""
# Load model artifacts
model = joblib.load('./01-notebooks/model.pkl')
scaler = joblib.load('./01-notebooks/scaler.pkl')
feature_names = joblib.load('./01-notebooks/feature_names.pkl')
"""

# --- Helper Functions and Startup Event ---
def load_assets():
    """
    Loads the trained model, scaler, and feature names.
    """
    global model, scaler, feature_names
    
    try:
        # Load the list of expected features
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = joblib.load(f)
        print(f"Loaded {len(feature_names)} features: {feature_names[:10]}...")
            
        # Load the fitted scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = joblib.load(f)
        print("Scaler loaded successfully.")

        # Load the trained model
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
        print("Model loaded successfully.")
            
    except Exception as e:
        print(f"FATAL ERROR: Failed to load necessary model assets from {MODEL_DIR}. Error: {e}")
        raise RuntimeError(f"Failed to load model assets: {e}")

# Use FastAPI's startup event to load assets once.
@app.on_event("startup")
async def startup_event():
    """Load model assets when the application starts."""
    load_assets()


#  --- API Endpoint Definition ---

@app.get("/", summary="API Status Check")
def home():
    """Simple check to ensure the API is running."""
    return {"message": "Klarna Default Prediction API", "status": "active"}

@app.post('/predict', 
          #response_model=List[float],
          response_model=List[Dict[str, Union[str, float]]],
          summary="Predict Default Probability")

async def predict(data: Union[LoanData, List[LoanData]]):
    """
    Receives loan data (single object or list) and returns the predicted 
    probability of default for each application.
    """
    
    # 1. Input Data Handling
    if not isinstance(data, list):
        data = [data]
        
    # Convert list of Pydantic models to a list of dicts for DataFrame
    input_records = [item.model_dump() for item in data]
    input_df = pd.DataFrame(input_records)
    
    print(f"Received {len(input_df)} loan application(s)")
    
    # save loan_ids
    loan_ids = input_df['loan_id'].copy() 


    # 2. Preprocessing & Feature Engineering
    try:
        # Initialize the custom preprocessing step
        preprocessor = KlarnaDataProcessor(input_df)
        
        # Define target and clean data
        preprocessor.define_target_variable_and_clean_data()
        
        # Perform feature engineering
        preprocessor.feature_engineering()

        # Get preprocessed dataframe
        X_preprocessed = preprocessor.df
        
        # Drop the target column if it exists
        if 'default' in X_preprocessed.columns:
            X_preprocessed = X_preprocessed.drop(columns=['default'])
        
        print(f"After preprocessing, shape: {X_preprocessed.shape}")
        print(f"Features after preprocessing: {X_preprocessed.columns.tolist()[:10]}...")
        
        # CRITICAL: Align features with what the model expects
        # Add missing columns with 0s and remove extra columns
        missing_features = set(feature_names) - set(X_preprocessed.columns)
        extra_features = set(X_preprocessed.columns) - set(feature_names)
        
        if missing_features:
            print(f"Adding {len(missing_features)} missing features: {list(missing_features)[:5]}...")
            for feature in missing_features:
                X_preprocessed[feature] = 0
        
        if extra_features:
            print(f"Removing {len(extra_features)} extra features: {list(extra_features)[:5]}...")
        
        # Reorder columns to match training feature order
        X_preprocessed = X_preprocessed[feature_names]
        
        print(f"Final feature alignment: {X_preprocessed.shape}")
        
    except KeyError as e:
        print(f"KeyError during preprocessing: {e}")
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: Missing required feature: {e}")
    except Exception as e:
        print(f"Exception during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")
        
    # 3. Scaling & Prediction
    try:
        # Scale the features
        X_scaled = scaler.transform(X_preprocessed)
        print(f"Data scaled successfully, shape: {X_scaled.shape}")
        
        # Predict probabilities
        probabilities = model.predict_proba(X_scaled)[:, 1]
        print(f"Predictions generated: {probabilities}")
    
        
    except Exception as e:
        print(f"Exception during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    # 4. Output Results
    # Combine the saved loan_ids with the predicted probabilities
    results = []
    # loan_ids is a Pandas Series, probabilities is a NumPy array. They are aligned by index.
    for loan_id, prob in zip(loan_ids.values, probabilities.tolist()):
        results.append({
            "loan_id": loan_id,
            "probability": prob
        })

    return results