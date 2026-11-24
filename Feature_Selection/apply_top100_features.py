
import pandas as pd

# Load selected Top 100 bacteria
top100 = pd.read_csv(
    "Feature_Selection/EFS_Top100_Bacteria.csv"
)

selected_features = top100["Bacteria"].tolist()

# Load original train and test features
X_train = pd.read_csv("ML/X_train.csv")
X_test = pd.read_csv("ML/X_test.csv")

# Verify selected features exist
missing_train = set(selected_features) - set(X_train.columns)
missing_test = set(selected_features) - set(X_test.columns)

if missing_train or missing_test:
    raise ValueError("Some selected features are missing!")

# Extract the same Top 100 features
X_train_top100 = X_train[selected_features]
X_test_top100 = X_test[selected_features]

# Save datasets
X_train_top100.to_csv(
    "ML/X_train_Top100.csv", index=False
)

X_test_top100.to_csv(
    "ML/X_test_Top100.csv", index=False
)

print("===== TOP 100 FEATURE APPLICATION =====")
print("Training shape:", X_train_top100.shape)
print("Testing shape:", X_test_top100.shape)
print("Selected bacteria:", len(selected_features))
print("Missing features in training:", len(missing_train))
print("Missing features in testing:", len(missing_test))

print("\nTop 100 features applied successfully!")
