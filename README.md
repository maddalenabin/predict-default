# Klarna Case Study

**Objective**
Develop a model that predicts the probability of default of a customer on a purchase made with Klarna’s Pay Later payment method.

**The solution should contain**
- Code to host an API, which a reviewer should be able to host on their local system and
make requests to.
- A 1-pager that summarizes how you chose a target definition, trained your model, and
evaluated performance.
- Any code used to explore the data or develop the model, which a reviewer should be able
to follow.

# Note
More details on the observations made, plots and so on are included in the notebooks in `01-notebooks` folder. In particualr, for plots about the evaluation (metrics) look at `01-notebooks/03-modeling.ipynb`. In general, the last cell in each notebook collect the observations made throughout the analysis performed. 

# Structure of the project
```
klarna-case/
├─ 01-notebooks/
│  ├─ 01_explore.ipynb 
│  ├─ 02_features.ipynb
│  └─ 03_modeling.ipynb
├─ 02-data/
│  ├─ mlcasestudy.csv
│  ├─ mlcasestudy_cleaned.csv       # for my data exploration
│  └─ mlcasestudy_to_train.csv      # for my data exploration
├─ src/
│  ├─ preprocess.py                 # for API
│  └─ preprocess_mine.py            # includes one-hot encoding of categorical features
├─ api.py                           # FastAPI
├─ test_loan.json
├─ requirements.txt
└─ README.md
```


# 1. Target definition
* Predict probability of **default**: customer enters debt collection / eventually written off for Pay Later loans. I used 
```python
df["default"] = (df['amount_outstanding_21d'] > 0).astype(int)
```
* `amount_outstanding_14d` and `amount_outstanding_21d` are data concerning the loan after it has been issued. So we need to remove them to avoid leakage. Additionally, `amount_outstanding_21d` is used for the target definition.
* The data is **imbalanced**: the default rate is ~5%.


# 2. Preprocessing
## Cleaning
* Remove rows with missing valued in `card_expiry_month`and `card_expiry_year`. 
* Fill missin values in `existing_klarna_debt` with 0, assuming no debt if not reported.
* Check for duplicates using `loan_id`. No duplicates were found.

## Feature engineering
* As previously stated, drop `amount_outstanding_14d` and `amount_outstanding_21d` to avoid leakage, as well as `loan_id`, because not used for predictions.
* Create `days_until_expiration` as days from `loan_issue_date` to card expiration day (used `card_expiry_month` and `card_expiry_year`)
* Parse `loan_issue_date` and calculate `loan_issue_date_numeric`as number of days from a certain day. Here I used `2020-01-01`.
* Ration of failed to total number of loans issued, 3 and 6 months before the loan issue date. If the result is NaN, substitute it with 0, assuming the customer has no failed payments. Could use median or mean instead. This has to be discussed based on the needs of the business.
* New customer feature `is_new_customer`: if `days_since_first_loan`is smaller or equal to 5. This is arbitrary, can be tuned.
* In the model used to run locally with api I also dropped the categorical columns, to simplify. However, in the model run in my notebook, I did keep them and transformed to one-hot encoding. 

### Next steps to try:
* loan-to-debt ratio: `loan_amount / existing_klarna_debt`.
* rolling exposures: `new_exposure_7d / loan_amount`.
* Encode categorical (merchant\_category, merchant\_group) with frequency encoding or target encoding (with out-of-fold to avoid leakage).
* Define risky customer: `df['num_active_loans'] >= 3` and `(df['existing_klarna_debt'] > df['existing_klarna_debt'].median())).astype(int)`
* Think of other features which may have good predictive power: 
    * temporal features (month, weekday vs weekend)
    * repayment trend:  `df['repayment_trend_recent'] = df['amount_repaid_14d'] / 14 - df['amount_repaid_1m'] / 30`and other features `amount_repaid_xx`



# Training the Model
* Split the data in train, validation, test (60/20/20).
* Transform the features with `StandardScaler()`.
* Model: Logistic Regression with class weight.
* Binary classification: 0 no default, 1 default. I also got the output probability of course. 
* Tuned decision threshold based on the best F1-score. I used F1-score because it takes into account both precision and recall. However, in a real-world scenario, you might want to consider the business implications of false positives vs false negatives.
  
### Next steps to try:
* Logistic Regression balanced sampling, or ensenlbe methods XGBoost, LightGBM.
* Add cross-validation with time folds (time-series CV).
* Use target encoding with K-fold out-of-fold scheme for merchant features.
* Data balance: Oversample defaults (SMOTE) or use more aggressive class weights.


# Evaluation
* ROC-AUC and Average Precision (PR-AUC).
* As the class in imbalanced, ROC-AUC can be misleading. So better to use Precision-Recall curve.
    * Precision = how many loans you decline were actually bad (avoid losing good customers)
    * Recall = how many real defaults you catch (avoid approving bad loans, or having bad customers)
    * So you may prefer high recall if the cost of approving a bad loan is large, or tune threshold for high precision if declining good loans is more expensive.
* Confusion matric and classification report are also inspected (see `01-notebooks/03-modeling.ipynb` for more details and observations).
* I also checked for overfitting using train/val data sets.
* Which is the best metric for the business case? Is it more costly to approve bad loans or to decline good loans?

The summary of the metrics is here:
```bash
Confusion Matrix (Test):
[[17245  3891]
 [  765   449]]

Classification Report (Test):
              precision    recall  f1-score   support

           0       0.96      0.82      0.88     21136
           1       0.10      0.37      0.16      1214

    accuracy                           0.79     22350
   macro avg       0.53      0.59      0.52     22350
weighted avg       0.91      0.79      0.84     22350
````

---

# How to run the model locally
1. Install required libraries
```bash
pip install -r requirements.txt
```

2. Run the API using **Uvicorn**, the ASGI server recommended for FastAPI:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8080
```

3. Test the api as
```bash
curl -X POST -H "Content-Type: application/json" -d @test_loan.json http://127.0.0.1:8080/predict
```

# with high_risk_customer and OHE
Confusion Matrix (Test):
[[18130  3006]
 [  833   381]]

Classification Report (Test):
              precision    recall  f1-score   support

           0       0.96      0.86      0.90     21136
           1       0.11      0.31      0.17      1214

    accuracy                           0.83     22350
   macro avg       0.53      0.59      0.53     22350
weighted avg       0.91      0.83      0.86     22350


