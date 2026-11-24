
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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

y_train = pd.read_csv("ML/y_train.csv").values.ravel()
y_test = pd.read_csv("ML/y_test.csv").values.ravel()

# Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

# Parameters to test
param_grid = {
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Starting Decision Tree Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    "\nBest CV Accuracy:",
    round(grid_search.best_score_ * 100, 2),
    "%"
)

# Evaluate best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

results = {
    "Model": "Tuned Decision Tree",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1_Score": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, y_prob)
}

print("\n===== DECISION TREE TUNING RESULTS =====")

for metric, value in results.items():
    if metric != "Model":
        print(f"{metric}: {value:.4f}")

# Save results
pd.DataFrame([results]).to_csv(
    "ML/Decision_Tree_Tuning_Results.csv",
    index=False
)

print("\nResults saved: ML/Decision_Tree_Tuning_Results.csv")
