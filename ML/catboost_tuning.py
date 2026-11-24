
import os
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================
# 1. Load Dataset
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(
    os.path.join(BASE_DIR, "X_train_Top100.csv")
)

X_test = pd.read_csv(
    os.path.join(BASE_DIR, "X_test_Top100.csv")
)

y_train = pd.read_csv(
    os.path.join(BASE_DIR, "y_train.csv")
).iloc[:, 0]

y_test = pd.read_csv(
    os.path.join(BASE_DIR, "y_test.csv")
).iloc[:, 0]

# Remove unwanted index columns
X_train = X_train.loc[
    :, ~X_train.columns.str.contains("^Unnamed")
]

X_test = X_test.loc[
    :, ~X_test.columns.str.contains("^Unnamed")
]

print("\n========== CATBOOST TUNING ==========")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Features:", X_train.shape[1])

# =====================================
# 2. Initialize Model
# =====================================

model = CatBoostClassifier(
    loss_function="Logloss",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=1
)

# =====================================
# 3. Hyperparameter Grid
# =====================================

param_grid = {
    "iterations": [200, 300, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1, 3, 5]
}

# =====================================
# 4. 5-Fold Cross Validation
# =====================================

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
print("\nStarting Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

# =====================================
# 5. Best Parameters
# =====================================

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(f"{grid_search.best_score_ * 100:.2f}%")

# =====================================
# 6. Test Evaluation
# =====================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test).ravel()
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)
auc = roc_auc_score(y_test, y_prob)

print("\n========== TUNED CATBOOST RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")

# =====================================
# 7. Save Results
# =====================================

results = pd.DataFrame([{
    "Model": "Tuned CatBoost",
    "Best_CV_Accuracy": grid_search.best_score_ * 100,
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_Score": f1 * 100,
    "ROC_AUC": auc
}])

output_path = os.path.join(
    BASE_DIR, "CatBoost_Tuning_Results.csv"
)

results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("\n========== COMPLETED ==========")
