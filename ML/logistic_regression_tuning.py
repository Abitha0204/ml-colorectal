
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Define Logistic Regression pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        random_state=42
    ))
])

# Hyperparameter combinations
param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
    "model__class_weight": [None, "balanced"]
}

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Starting Logistic Regression Tuning...")

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

# Final test evaluation
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

results = {
    "Model": "Tuned Logistic Regression",
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions),
    "Recall": recall_score(y_test, predictions),
    "F1_Score": f1_score(y_test, predictions),
    "ROC_AUC": roc_auc_score(y_test, probabilities)
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    "ML/Logistic_Regression_Tuning_Results.csv",
    index=False
)

print("\n===== TUNED LOGISTIC REGRESSION RESULTS =====")
print(results_df.round(4).to_string(index=False))

print("\nResults saved: ML/Logistic_Regression_Tuning_Results.csv")
