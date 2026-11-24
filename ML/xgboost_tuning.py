
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load training and testing data
X_train = pd.read_csv("ML/X_train_Top100.csv")
X_test = pd.read_csv("ML/X_test_Top100.csv")

y_train = pd.read_csv("ML/y_train.csv")["condition"]
y_test = pd.read_csv("ML/y_test.csv")["condition"]

# Define XGBoost model
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# Hyperparameter combinations
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Starting XGBoost Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

# Best model
best_xgb = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Final test evaluation
predictions = best_xgb.predict(X_test)
probabilities = best_xgb.predict_proba(X_test)[:, 1]

results = {
    "Model": "Tuned XGBoost",
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions),
    "Recall": recall_score(y_test, predictions),
    "F1_Score": f1_score(y_test, predictions),
    "ROC_AUC": roc_auc_score(y_test, probabilities)
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    "ML/XGBoost_Tuning_Results.csv",
    index=False
)

print("\n===== TUNED XGBOOST RESULTS =====")
print(results_df.round(4).to_string(index=False))

print("\nResults saved: ML/XGBoost_Tuning_Results.csv")
