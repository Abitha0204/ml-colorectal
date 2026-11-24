
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load training data only
X_train = pd.read_csv("ML/X_train.csv")
y_train = pd.read_csv("ML/y_train.csv").squeeze()

print("===== ENSEMBLE FEATURE SELECTION =====")
print("Training samples:", X_train.shape[0])
print("Available bacterial features:", X_train.shape[1])

# 2. Separate CRC and control samples
crc = X_train[y_train == 1]
control = X_train[y_train == 0]

# 3. Mann-Whitney U test
mw_scores = []

for col in X_train.columns:
    score = mannwhitneyu(
        crc[col], control[col], alternative="two-sided"
    ).pvalue
    mw_scores.append(score)

mw_rank = pd.Series(mw_scores, index=X_train.columns).rank(
    ascending=True, method="average"
)

# 4. Pearson correlation
pearson_scores = []

for col in X_train.columns:
    result = pearsonr(X_train[col], y_train)
    pearson_scores.append(abs(result.statistic))

pearson_rank = pd.Series(pearson_scores, index=X_train.columns).rank(
    ascending=False, method="average"
)

# 5. Spearman correlation
spearman_scores = []

for col in X_train.columns:
    result = spearmanr(X_train[col], y_train)
    spearman_scores.append(abs(result.statistic))

spearman_rank = pd.Series(spearman_scores, index=X_train.columns).rank(
    ascending=False, method="average"
)

# 6. Logistic Regression importance
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)
lr_model.fit(X_train, y_train)

lr_importance = np.abs(
    lr_model.named_steps["logisticregression"].coef_[0]
)

lr_rank = pd.Series(lr_importance, index=X_train.columns).rank(
    ascending=False, method="average"
)

# 7. Random Forest importance
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_importance = rf_model.feature_importances_

rf_rank = pd.Series(rf_importance, index=X_train.columns).rank(
    ascending=False, method="average"
)

# 8. Ensemble ranking
results = pd.DataFrame({
    "Bacteria": X_train.columns,
    "Mann_Whitney_Rank": mw_rank.values,
    "Pearson_Rank": pearson_rank.values,
    "Spearman_Rank": spearman_rank.values,
    "Logistic_Regression_Rank": lr_rank.values,
    "Random_Forest_Rank": rf_rank.values
})

rank_columns = [
    "Mann_Whitney_Rank",
    "Pearson_Rank",
    "Spearman_Rank",
    "Logistic_Regression_Rank",
    "Random_Forest_Rank"
]

results["Ensemble_Score"] = results[rank_columns].mean(axis=1)

results = results.sort_values("Ensemble_Score").reset_index(drop=True)
results["Final_Rank"] = range(1, len(results) + 1)

# 9. Select Top 100
top100 = results.head(100)

# 10. Save results
results.to_csv(
    "Feature_Selection/EFS_All_Feature_Ranking.csv",
    index=False
)

top100.to_csv(
    "Feature_Selection/EFS_Top100_Bacteria.csv",
    index=False
)

print("\n===== EFS RESULTS =====")
print("Total bacteria ranked:", len(results))
print("Top bacteria selected:", len(top100))
print("\nTop 10 bacteria:")
print(top100[["Bacteria", "Ensemble_Score", "Final_Rank"]].head(10))

print("\nEFS completed successfully!")
