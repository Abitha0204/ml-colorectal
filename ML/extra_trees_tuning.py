
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("\n========== EXTRA TREES TUNING ==========\n")

# Load training and testing data
X_train = pd.read_csv("ML/X_train_Top100.csv")
X_test = pd.read_csv("ML/X_test_Top100.csv")

y_train = pd.read_csv("ML/y_train.csv")["condition"]
y_test = pd.read_csv("ML/y_test.csv")["condition"]

# Encode labels
label_map = {
    "control": 0,
    "healthy": 0,
    "CRC": 1,
    "cancer": 1
}

y_train = y_train.replace(label_map).astype(int)
y_test = y_test.replace(label_map).astype(int)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Features:", X_train.shape[1])

# Extra Trees model
model = ExtraTreesClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

# Cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search
print("\nStarting Hyperparameter Tuning...")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Testing
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== TUNED EXTRA TREES RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Save results
os.makedirs("ML", exist_ok=True)

results = pd.DataFrame([{
    "Model": "Extra Trees",
    "Best_CV_Accuracy": grid_search.best_score_,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "Best_Parameters": str(grid_search.best_params_)
}])

results.to_csv(
    "ML/Extra_Trees_Tuning_Results.csv",
    index=False
)

# Save CV results
pd.DataFrame(grid_search.cv_results_).to_csv(
    "ML/Extra_Trees_CV_Results.csv",
    index=False
)

print("\nResults saved to: ML/Extra_Trees_Tuning_Results.csv")
print("\n========== COMPLETED ==========\n")
