
import os
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================
# 1. LOAD COMBINED DATASET
# ==========================================

DATA_PATH = "Dataset/Processed/CRC_Merged_Initial.csv"

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)

# Encode labels
label_mapping = {
    "control": 0,
    "CRC": 1
}

df["condition"] = (
    df["condition"].astype(str).str.strip().map(label_mapping)
)

if df["condition"].isnull().any():
    raise ValueError("Unknown condition labels found!")

df["condition"] = df["condition"].astype(int)

# Metadata columns are not ML features
metadata_columns = ["condition", "Country", "Dataset"]

feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

datasets = sorted(df["Dataset"].dropna().unique())

print("Total Samples:", len(df))
print("Total Bacterial Features:", len(feature_columns))
print("Total Datasets:", len(datasets))

# Output folder
os.makedirs("Validation/LODO_Feature_Rankings", exist_ok=True)

fold_results = []
all_predictions = []

# ==========================================
# 2. LODO LOOP
# ==========================================

for held_out_dataset in datasets:

    print("\n" + "=" * 55)
    print("Testing Dataset:", held_out_dataset)
    print("=" * 55)

    # Separate training and testing datasets
    train_df = df[df["Dataset"] != held_out_dataset].copy()
    test_df = df[df["Dataset"] == held_out_dataset].copy()

    X_train = train_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    X_test = test_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    y_train = train_df["condition"]
    y_test = test_df["condition"]

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # ======================================
    # 3. PREVALENCE FILTERING (TRAIN ONLY)
    # ======================================

    prevalence = (X_train > 0).mean(axis=0)

    retained_features = prevalence[
        prevalence >= 0.10
    ].index.tolist()

    X_train = X_train[retained_features]
    X_test = X_test[retained_features]

    print(
        "Features after prevalence filtering:",
        len(retained_features)
    )

    if len(retained_features) < 100:
        raise ValueError(
            f"Only {len(retained_features)} features available "
            f"for {held_out_dataset}. Cannot select Top 100."
        )

    # ======================================
    # 4. EFS FEATURE SELECTION (TRAIN ONLY)
    # ======================================

    control = X_train[y_train == 0]
    crc = X_train[y_train == 1]

    ranking = pd.DataFrame(index=retained_features)

    # Mann-Whitney U test
    mw_scores = {}

    for col in retained_features:
        try:
            mw_scores[col] = mannwhitneyu(
                control[col],
                crc[col],
                alternative="two-sided"
            ).pvalue
        except ValueError:
            mw_scores[col] = 1.0

    ranking["Mann_Whitney"] = pd.Series(mw_scores)
    ranking["Mann_Whitney_Rank"] = (
        ranking["Mann_Whitney"].rank(ascending=True)
    )

    # Pearson correlation
    pearson_scores = {}

    for col in retained_features:
        try:
            value = pearsonr(
                X_train[col],
                y_train
            )[0]

            pearson_scores[col] = (
                abs(value) if np.isfinite(value) else 0
            )

        except Exception:
            pearson_scores[col] = 0

    ranking["Pearson"] = pd.Series(pearson_scores)
    ranking["Pearson_Rank"] = (
        ranking["Pearson"].rank(ascending=False)
    )

    # Spearman correlation
    spearman_scores = {}

    for col in retained_features:
        try:
            value = spearmanr(
                X_train[col],
                y_train
            ).statistic

            spearman_scores[col] = (
                abs(value) if np.isfinite(value) else 0
            )

        except Exception:
            spearman_scores[col] = 0

    ranking["Spearman"] = pd.Series(spearman_scores)
    ranking["Spearman_Rank"] = (
        ranking["Spearman"].rank(ascending=False)
    )

    # Logistic Regression importance
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=42
        ))
    ])

    lr_pipeline.fit(X_train, y_train)

    lr_importance = np.abs(
        lr_pipeline.named_steps["model"].coef_[0]
    )

    ranking["Logistic_Regression"] = lr_importance
    ranking["Logistic_Rank"] = (
        ranking["Logistic_Regression"].rank(ascending=False)
    )

    # Random Forest importance
    selector_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    selector_rf.fit(X_train, y_train)

    ranking["Random_Forest"] = (
        selector_rf.feature_importances_
    )

    ranking["Random_Forest_Rank"] = (
        ranking["Random_Forest"].rank(ascending=False)
    )

    # Ensemble Feature Selection
    rank_columns = [
        "Mann_Whitney_Rank",
        "Pearson_Rank",
        "Spearman_Rank",
        "Logistic_Rank",
        "Random_Forest_Rank"
    ]

    ranking["Ensemble_Score"] = (
        ranking[rank_columns].mean(axis=1)
    )

    ranking = ranking.sort_values("Ensemble_Score")

    top100_features = ranking.head(100).index.tolist()

    # Save fold-wise feature ranking
    ranking.to_csv(
        f"Validation/LODO_Feature_Rankings/"
        f"{held_out_dataset}_ranking.csv"
    )

    print("Selected Top Features:", len(top100_features))

    # ======================================
    # 5. RANDOM FOREST TRAINING
    # ======================================

    X_train_top100 = X_train[top100_features]
    X_test_top100 = X_test[top100_features]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_top100, y_train)

    # ======================================
    # 6. TEST EVALUATION
    # ======================================

    y_pred = model.predict(X_test_top100)
    y_prob = model.predict_proba(X_test_top100)[:, 1]

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

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    fold_results.append({
        "Dataset": held_out_dataset,
        "Train_Samples": len(X_train),
        "Test_Samples": len(X_test),
        "Selected_Features": len(top100_features),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    for actual, predicted, probability in zip(
        y_test, y_pred, y_prob
    ):
        all_predictions.append({
            "Dataset": held_out_dataset,
            "Actual": actual,
            "Predicted": predicted,
            "Probability": probability
        })


# ==========================================
# 7. FINAL LODO SUMMARY
# ==========================================

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    "Validation/LODO_RF_Fold_Results.csv",
    index=False
)

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    "Validation/LODO_RF_Predictions.csv",
    index=False
)

# Mean accuracy across 8 datasets
mean_accuracy = results_df["Accuracy"].mean()

# Pooled accuracy across all held-out predictions
pooled_accuracy = accuracy_score(
    predictions_df["Actual"],
    predictions_df["Predicted"]
)

print("\n" + "=" * 55)
print("FINAL LODO RESULTS - RANDOM FOREST")
print("=" * 55)

print("\nDataset-wise Results:")
print(results_df.to_string(index=False))

print(
    "\nMean LODO Accuracy:",
    round(mean_accuracy * 100, 2),
    "%"
)

print(
    "Pooled LODO Accuracy:",
    round(pooled_accuracy * 100, 2),
    "%"
)

print("\nSaved Files:")
print("1. Validation/LODO_RF_Fold_Results.csv")
print("2. Validation/LODO_RF_Predictions.csv")
print("3. Validation/LODO_Feature_Rankings/")
