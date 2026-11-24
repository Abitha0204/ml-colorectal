
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================
# 1. LOAD TOP 100 FEATURES AND LABELS
# =====================================

X_train = pd.read_csv("ML/X_train_Top100.csv")
X_test = pd.read_csv("ML/X_test_Top100.csv")

y_train = pd.read_csv("ML/y_train.csv")["condition"]
y_test = pd.read_csv("ML/y_test.csv")["condition"]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Bacterial features:", X_train.shape[1])

# =====================================
# 2. DEFINE SIX ML MODELS
# =====================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            probability=True,
            random_state=42
        )
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
}

# =====================================
# 3. TRAIN AND EVALUATE MODELS
# =====================================

results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),

        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_test, probabilities
        )
    })

# =====================================
# 4. SAVE MODEL COMPARISON RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "ML/ML_Model_Comparison.csv",
    index=False
)

print("\n===== MODEL COMPARISON RESULTS =====")

print(
    results_df.round(4).to_string(index=False)
)

print("\nResults saved: ML/ML_Model_Comparison.csv")
