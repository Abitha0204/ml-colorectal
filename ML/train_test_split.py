
import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded dataset
df = pd.read_csv("Dataset/Processed/CRC_Encoded_Labels.csv")

# Separate bacterial features and target
metadata = df[["Country", "Dataset"]]
X = df.drop(columns=["condition", "Country", "Dataset"])
y = df["condition"]

# Split data (80:20)
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, metadata,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("===== TRAIN-TEST SPLIT RESULTS =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Bacterial features:", X.shape[1])

print("\nTraining class counts:")
print(y_train.value_counts())

print("\nTesting class counts:")
print(y_test.value_counts())

# Save split datasets
X_train.to_csv("ML/X_train.csv", index=False)
X_test.to_csv("ML/X_test.csv", index=False)
y_train.to_csv("ML/y_train.csv", index=False)
y_test.to_csv("ML/y_test.csv", index=False)

meta_train.to_csv("ML/meta_train.csv", index=False)
meta_test.to_csv("ML/meta_test.csv", index=False)

print("\nAll split files saved successfully!")
