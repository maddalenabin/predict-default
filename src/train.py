import pandas as pd
import pickle
import sys
from datetime import datetime


def load_model(model_path='model.pkl'):
    """Load the trained model and associated objects."""
    print("Loading trained model...")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    print("✓ Model loaded successfully")
    
    # Display training info if available
    if 'training_info' in model_package:
        print("\n" + "="*50)
        print("MODEL TRAINING INFO")
        print("="*50)
        info = model_package['training_info']
        for key, value in info.items():
            print(f"{key}: {value:.4f}")
    
    return model_package

def preprocess_new_data(data, feature_columns):
    """Preprocess new data to match the training data format."""
    print("\nPreprocessing new data...")
    
    # Check if 'loan_id' exists (keep it for output)
    if 'loan_id' in data.columns:
        loan_ids = data['loan_id']
        X = data.drop(['loan_id'], axis=1)
    else:
        loan_ids = None
        X = data.copy()
    
    # Remove target column if it exists in new data
    if 'default' in X.columns:
        print("Note: 'default' column found in new data - removing it for prediction")
        X = X.drop(['default'], axis=1)
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=['merchant_group', 'merchant_category'], drop_first=True)
    
    # Ensure columns match training data (add missing columns with 0s)
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    
    # Remove extra columns that weren't in training data
    extra_cols = set(X.columns) - set(feature_columns)
    if extra_cols:
        print(f"Warning: Removing {len(extra_cols)} columns not present in training data")
        X = X.drop(columns=list(extra_cols))
    
    # Reorder columns to match training data
    X = X[feature_columns]
    
    print(f"✓ Preprocessing complete. Shape: {X.shape}")
    
    return X, loan_ids


def make_predictions(model_package, new_data_path, output_path='predictions.csv'):
    """Make predictions on new data and save results."""
    
    # Load new data
    print(f"\nLoading new data from '{new_data_path}'...")
    try:
        new_data = pd.read_csv(new_data_path)
        print(f"✓ Loaded {len(new_data)} records")
    except FileNotFoundError:
        print(f"Error: File '{new_data_path}' not found!")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract model components
    model = model_package['model']
    scaler = model_package['scaler']
    feature_columns = model_package['feature_columns']
    
    # Preprocess new data
    X_new, loan_ids = preprocess_new_data(new_data, feature_columns)
    
    # Scale features
    print("Scaling features...")
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_new_scaled)
    prediction_probabilities = model.predict_proba(X_new_scaled)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame()
    if loan_ids is not None:
        results['loan_id'] = loan_ids
    results['predicted_default'] = predictions
    results['default_probability'] = prediction_probabilities
    
    # Add risk category based on probability
    results['risk_category'] = pd.cut(
        results['default_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Save predictions
    results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to '{output_path}'")
    
    # Display summary statistics
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted defaults (1): {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
    print(f"Predicted no default (0): {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"\nAverage default probability: {prediction_probabilities.mean():.4f}")
    print(f"Median default probability: {prediction_probabilities.median():.4f}")
    print(f"Min default probability: {prediction_probabilities.min():.4f}")
    print(f"Max default probability: {prediction_probabilities.max():.4f}")
    
    print("\nRisk Distribution:")
    print(results['risk_category'].value_counts().sort_index())
    
    return results


if __name__ == "__main__":
    # Check if data path is provided as command line argument
    if len(sys.argv) > 1:
        new_data_path = sys.argv[1]
    else:
        # Default path - you can change this
        new_data_path = "./02-data/new_data.csv"
        print(f"No data path provided. Using default: {new_data_path}")
    
    # Output path (optional second argument)
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    print("="*50)
    print("KLARNA DEFAULT PREDICTION")
    print("="*50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model_package = load_model('model.pkl')
    
    # Make predictions
    results = make_predictions(model_package, new_data_path, output_path)
    
    print("\n" + "="*50)
    print("PREDICTION COMPLETE")
    print("="*50)