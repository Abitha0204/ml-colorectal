
import os
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. Load Existing Top 100 Dataset
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train_Top100.csv"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test_Top100.csv"))

y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv"))
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv"))

# Remove unwanted index columns
X_train = X_train.loc[:, ~X_train.columns.str.contains("^Unnamed")]
X_test = X_test.loc[:, ~X_test.columns.str.contains("^Unnamed")]

y_train = y_train.loc[:, ~y_train.columns.str.contains("^Unnamed")]
y_test = y_test.loc[:, ~y_test.columns.str.contains("^Unnamed")]

# Extract target labels
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

# ==========================================
# 2. Dataset Information
# ==========================================

print("\n========== LIGHTGBM TRAINING ==========")

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
print("Number of Features:", X_train.shape[1])

print("\nTraining Class Counts:")
print(y_train.value_counts())

# ==========================================
# 3. Initialize LightGBM Model
# ==========================================

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

# ==========================================
# 4. Train Model
# ==========================================

print("\nTraining LightGBM Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================
# 5. Make Predictions
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 6. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== LIGHTGBM RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 7. Save Results
# ==========================================

results = pd.DataFrame([{
    "Model": "LightGBM",
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_Score": f1 * 100,
    "ROC_AUC": roc_auc
}])

output_path = os.path.join(BASE_DIR, "LightGBM_Results.csv")
results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("\n========== COMPLETED ==========")
