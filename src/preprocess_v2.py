import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from datetime import datetime

# Define a fixed, known date to normalize time features against, 
# since the API cannot see the min date of the training set.
# Using the Unix epoch start date (2020-01-01).
EPOCH_START = pd.to_datetime('2020-01-01')

class KlarnaDataProcessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean, engineer features, and align/select final columns
    for the Klarna credit default prediction model.
    """
    def __init__(self, feature_names=None):
        #self.feature_names = feature_names 
        #self.categorical_cols = ['merchant_group', 'merchant_category'] 
        # Add any other categorical columns you used for OHE here!
        self.categorical_cols = ['merchant_group', 'merchant_category'] 
        
        if feature_names is None:
            # When feature_names is None, we are in the training phase 
            # and need to generate the list of features. Do NOT load a file.
            self.feature_names = None 
        elif isinstance(feature_names, list):
            # When feature_names is a list, we are in the API/testing phase.
            self.feature_names = feature_names
        else:
            # If feature_names is not provided (None) or not a list, raise an error.
            # This is where your previous code likely tried to load the file and failed.
            raise ValueError("Feature names must be a list or None during training initialization.")

        # Set the constant for date calculation
        # This is the date we normalize all issue dates against (1st date in your dataset)
        self.EPOCH_START = datetime(2022, 12, 1) 
        print("KlarnaDataProcessor initialized.")

        


    def fit(self, X, y=None):
        # We don't fit anything here, as the scaler/model handles the scaling/training.
        return self

    def transform(self, X):
        """
        Applies cleaning, feature engineering, OHE, and final column alignment 
        on the input DataFrame X (which may be a single row).
        """
        # Suppress warnings that might clutter the API log
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            X = X.copy()
            
            # --- 0. Data Cleaning for API Input ---
            
            # The API assumes the input is clean enough from Pydantic validation, 
            # but we must handle explicit nulls/types required for your logic.
            
            # Fill missing existing_klarna_debt with 0 (as per your training logic)
            X["existing_klarna_debt"].fillna(0, inplace=True)
            
            # --- 1. Drop Leakage/ID Columns (Before Engineering) ---
            
            # Drop columns used for target definition or IDs, as they leak information.
            cols_to_drop = [
                'amount_outstanding_14d', 
                "amount_outstanding_21d", 
                'loan_id'
            ]
            X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')
            
            # --- 2. Feature Engineering Logic (Replicating your notebook steps) ---

            # 2a. Days until expiration (Requires: card_expiry_year, card_expiry_month, loan_issue_date)
            if all(col in X.columns for col in ['card_expiry_year', 'card_expiry_month', 'loan_issue_date']):
                X['expiration_date'] = pd.to_datetime(
                    X['card_expiry_year'].astype('Int64').astype(str) + '-' + 
                    X['card_expiry_month'].astype('Int64').astype(str) + '-01',
                    errors='coerce'
                )
                X['days_until_expiration'] = (X['expiration_date'] - pd.to_datetime(X['loan_issue_date'])).dt.days
                
                X = X.drop(columns=['card_expiry_year', 'card_expiry_month', 'expiration_date'])

            # 2b. Loan Issue Date Numeric (Normalized by a fixed epoch start date)
            if "loan_issue_date" in X.columns:
                X["loan_issue_date"] = pd.to_datetime(X["loan_issue_date"], format="%Y-%m-%d", errors='coerce')
                # Use a fixed date reference (epoch start) instead of the training set's min date
                X["loan_issue_date_numeric"] = (X['loan_issue_date'] - EPOCH_START).dt.days

                X = X.drop(columns="loan_issue_date")
            
            # 2c. Ratio failed to confirmed payments in last 3 months
            if all(col in X.columns for col in ['num_failed_payments_3m', 'num_confirmed_payments_3m']):
                denominator_3m = X['num_confirmed_payments_3m'] + X["num_failed_payments_3m"]
                X['ratio_failed_to_confirmed_3m'] = X['num_failed_payments_3m'] / denominator_3m
                
                # Handling 0/0 case by setting ratio to 0 (as a simple API strategy)
                X['ratio_failed_to_confirmed_3m'].fillna(0, inplace=True) 
                X = X.drop(columns=['num_failed_payments_3m'])

            # 2d. Ratio failed to confirmed payments in last 6 months
            if all(col in X.columns for col in ['num_failed_payments_6m', 'num_confirmed_payments_6m']):
                denominator_6m = X['num_confirmed_payments_6m'] + X["num_failed_payments_6m"]
                X['ratio_failed_to_confirmed_6m'] = X['num_failed_payments_6m'] / denominator_6m
                
                # Handling 0/0 case by setting ratio to 0
                X['ratio_failed_to_confirmed_6m'].fillna(0, inplace=True)
                X = X.drop(columns=['num_failed_payments_6m'])

            # 2e. New customer flag
            if 'days_since_first_loan' in X.columns:
                X['is_new_customer'] = (X['days_since_first_loan'] <= 5).astype(int)
                # Keep 'days_since_first_loan' as it is a base feature

            # 2f. Remove new_exposure_7d
            if "new_exposure_7d" in X.columns:
                X = X.drop(columns=['new_exposure_7d'])

            
            # --- 3. Perform One-Hot Encoding and Column Alignment (CRITICAL) ---

            # A. One-Hot Encoding on the intermediate data
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
            
            # B. Final Feature Selection and Alignment
            if self.feature_names is None:
                 raise ValueError("Feature names list must be loaded from feature_names.pkl.")
            
            # Create a new DataFrame with the correct index and the 92 expected column headers
            X_aligned = pd.DataFrame(index=X.index, columns=self.feature_names)
            
            # Copy values from X into X_aligned (matching columns get values, others are NaN)
            X_aligned.update(X)
            
            # Fill all NaN values (which represent absent categories) with 0
            X_final = X_aligned.fillna(0)
            
            # Ensure the output is a standard DataFrame for the scaler


            # --- Update in your local src/preprocess.py file ---




            return X_final
