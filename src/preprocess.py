import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

EPOCH_START = pd.to_datetime('2020-01-01')

class KlarnaDataProcessor:
    """
    Class to handle target definition, feature engineering, and data quality checks
    """

    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.engineered_features = []
        print(f"Initialized KlarnaDataProcessor with {self.df.shape[0]:,} rows and {self.df.shape[1]} columns")
    
    def define_target_variable_and_clean_data(self):
        """
        Define target variable 'default' based on outstanding amounts.
        Logic:
        
        """
        # Check if we have the required columns
        if 'amount_outstanding_21d' not in self.df.columns:
            print("Error: 'amount_outstanding_21d' column not found")
            return
        
        # True 1 if defaulted, False 0 if the loan is paid off  
        self.df["default"] = (self.df['amount_outstanding_21d'] > 0).astype(int) 
        
        # Calculate default rate
        default_rate = self.df['default'].mean()
        n_defaults = self.df['default'].sum()
        n_total = len(self.df)

        print(f"Default rate: {default_rate:.3f} ({default_rate*100:.1f}%)")
        print(f"Defaults: {n_defaults:,} out of {n_total:,} loans.")

        # clean data by removing rows with missing target
        # Drop loan entries where "card_expiry_month" and "card_expiry_year" is missing.
        self.df = self.df.dropna(subset=["card_expiry_month", "card_expiry_year"])

        # Fill missing existing_klarna_debt with 0, assuming no debt if not reported
        self.df["existing_klarna_debt"].fillna(0, inplace=True)

        # check if there are duplicate loan_id entries
        if self.df['loan_id'].duplicated().any():
            n_duplicates = self.df['loan_id'].duplicated().sum()
            print(f"Warning: Found {n_duplicates} duplicate loan_id entries. Keeping the first occurrence.")
            self.df = self.df.drop_duplicates(subset=['loan_id'], keep='first')

        return 



    def feature_engineering(self):
        """
        Feature Engineering
        Create derived features that I think might improve model performance
        """
        # 1. remove columns to avoid leakage
        self.df = self.df.drop(columns=['amount_outstanding_14d', "amount_outstanding_21d"]) 
        self.df = self.df.drop(columns=['loan_id']) # not useful for prediction

        # 2. Change card expiry year to number of days until expiry

        if all(col in self.df.columns for col in ['card_expiry_year', 'card_expiry_month', 'loan_issue_date']):
            self.df['expiration_date'] = pd.to_datetime(
                self.df['card_expiry_year'].astype(int).astype(str) + '-' + self.df['card_expiry_month'].astype(int).astype(str) + '-01'
            ) 
            # Compute days until expiration
            self.df['days_until_expiration'] = (self.df['expiration_date'] - pd.to_datetime(self.df['loan_issue_date'])).dt.days

            self.df = self.df.drop(columns=['card_expiry_year', 'card_expiry_month', 'expiration_date'])
            self.engineered_features.append('days_until_expiration')

        # 3. Change issue date of loan to days since the earliest date in the dataset
        if "loan_issue_date" in self.df.columns:
            self.df["loan_issue_date"]  = pd.to_datetime(self.df["loan_issue_date"], format="%Y-%m-%d")
            self.df["loan_issue_date_numeric"] = (self.df['loan_issue_date'] - EPOCH_START).dt.days
            # X["loan_issue_date_numeric"] = (X['loan_issue_date'] - EPOCH_START).dt.days
            self.df = self.df.drop(columns="loan_issue_date")
            self.engineered_features.append("loan_issue_date_numeric")

            
        
        """
        # 4. Debt to loan ratio
        if self.df['existing_klarna_debt'] and self.df['loan_amount']:
            self.df['debt_to_loan_ratio'] = self.df['existing_klarna_debt'] / (self.df['loan_amount'])
            self.engineered_features.append('debt_to_loan_ratio')       
        """

        # 5. Ratio failed to confirmed payments in last 3 and 6 months
        if all(col in self.df.columns for col in ['num_failed_payments_3m', 'num_confirmed_payments_3m']):
            self.df['ratio_failed_to_confirmed_3m'] = self.df['num_failed_payments_3m'] / (self.df['num_confirmed_payments_3m'] + self.df["num_failed_payments_3m"] ) 
            # fill NaN values (from 0/0) with mean ratio, assuming the customer has no failed payments
            self.df['ratio_failed_to_confirmed_3m'].fillna(0, inplace=True)
            self.df = self.df.drop(columns=['num_failed_payments_3m'])
            self.engineered_features.append('ratio_failed_to_confirmed_3m')


        if all(col in self.df.columns for col in ['num_failed_payments_6m', 'num_confirmed_payments_6m']):
            self.df['ratio_failed_to_confirmed_6m'] = self.df['num_failed_payments_6m'] / (self.df['num_confirmed_payments_6m'] + self.df["num_failed_payments_6m"] ) 
            # fill NaN values (from 0/0) with mean ratio, assuming the customer has no failed payments
            self.df['ratio_failed_to_confirmed_6m'].fillna(0, inplace=True)
            self.df = self.df.drop(columns=['num_failed_payments_6m'])
            self.engineered_features.append('ratio_failed_to_confirmed_6m')

        # 6. High risk customer flag: more than 3 active loans and existing debt above median
        """
        df['high_risk_customer'] = ((df['num_active_loans'] >= 3) & # 3 is set arbitrarily, more studies needed to set something meaningful
                            (df['existing_klarna_debt'] > df['existing_klarna_debt'].median())
                            ).astype(int)
        """
        
        # 7. New customer
        if 'days_since_first_loan' in self.df.columns:
            self.df['is_new_customer'] = (self.df['days_since_first_loan'] <= 5).astype(int)
            self.engineered_features.append('is_new_customer') 
        
        # simplify by removing categorical variables for now
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.df = self.df.drop(columns=categorical_cols)

        """
        #numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        
        # Handle categorical variables with one-hot encoding
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        """
        print("Feature engineered:", self.engineered_features)
        
        return 
        




