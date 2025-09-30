# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

from preprocess import build_preprocessor

# 1. Load your dataset
df = pd.read_csv("data/loans.csv")  # replace with real filename

# 2. Define target and features
target = "default"  # replace with actual target column
y = df[target]
X = df.drop(columns=[target])

# drop obvious leakage features for baseline
leakage_features = ["amount_outstanding_14d", "amount_outstanding_21d"]
X = X.drop(columns=[col for col in leakage_features if col in X])

# 3. Define numeric/categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 4. Build preprocessor
preprocessor = build_preprocessor(numeric_features, categorical_features)

# 5. Train/val split (for a proper solution, do time-based split!)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Fit preprocessor
preprocessor.fit(X_train)

X_train_prep = preprocessor.transform(X_train)
X_val_prep = preprocessor.transform(X_val)

# 7. Train model (LightGBM)
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_prep, y_train)

# 8. Evaluate
y_val_pred = model.predict_proba(X_val_prep)[:, 1]
roc = roc_auc_score(y_val, y_val_pred)
ap = average_precision_score(y_val, y_val_pred)

print(f"Validation ROC-AUC: {roc:.3f}")
print(f"Validation PR-AUC: {ap:.3f}")

# 9. Save artifacts
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "model.pkl")
print("Artifacts saved: preprocessor.pkl, model.pkl")
