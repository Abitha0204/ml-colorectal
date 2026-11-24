
import os
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train_Top100.csv"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test_Top100.csv"))

y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv")).iloc[:, 0]
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv")).iloc[:, 0]

# Remove unwanted index columns
X_train = X_train.loc[:, ~X_train.columns.str.contains("^Unnamed")]
X_test = X_test.loc[:, ~X_test.columns.str.contains("^Unnamed")]

print("Starting LightGBM Hyperparameter Tuning...")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Features:", X_train.shape[1])

# Base model
model = LGBMClassifier(
    random_state=42,
    verbosity=-1,
    n_jobs=1
)

# Parameters to test
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [7, 15, 31],
    "max_depth": [-1, 5, 10],
    "min_child_samples": [10, 20]
}

# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Tune using training data only
grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(f"{grid_search.best_score_ * 100:.2f}%")

# Evaluate best model on test data
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n========== TUNED LIGHTGBM RESULTS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")

# Save results
results = pd.DataFrame([{
    "Model": "Tuned LightGBM",
    "Best_CV_Accuracy": grid_search.best_score_ * 100,
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_Score": f1 * 100,
    "ROC_AUC": auc
}])

results.to_csv(
    os.path.join(BASE_DIR, "LightGBM_Tuning_Results.csv"),
    index=False
)

print("\nResults saved successfully!")
