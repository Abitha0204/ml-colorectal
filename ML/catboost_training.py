
import os
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =====================================
# 1. Load Dataset
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train_Top100.csv"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test_Top100.csv"))

y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv"))
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv"))

# Remove unwanted index columns
X_train = X_train.loc[:, ~X_train.columns.str.contains("^Unnamed")]
X_test = X_test.loc[:, ~X_test.columns.str.contains("^Unnamed")]

y_train = y_train.loc[:, ~y_train.columns.str.contains("^Unnamed")].iloc[:, 0]
y_test = y_test.loc[:, ~y_test.columns.str.contains("^Unnamed")].iloc[:, 0]

print("\n========== CATBOOST TRAINING ==========")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Number of Features:", X_train.shape[1])

print("\nTraining Class Counts:")
print(y_train.value_counts())

# =====================================
# 2. Initialize CatBoost
# =====================================

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1
)

# =====================================
# 3. Train Model
# =====================================

print("\nTraining CatBoost Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================
# 4. Predictions
# =====================================

y_pred = model.predict(X_test).ravel()
y_prob = model.predict_proba(X_test)[:, 1]

# =====================================
# 5. Evaluation
# =====================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n========== CATBOOST RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================================
# 6. Save Results
# =====================================

results = pd.DataFrame([{
    "Model": "CatBoost",
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_Score": f1 * 100,
    "ROC_AUC": auc
}])

output_path = os.path.join(BASE_DIR, "CatBoost_Results.csv")
results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("\n========== COMPLETED ==========")
