
import pandas as pd

file_path = "Dataset/Processed/CRC_Prevalence_Filtered_10pct.csv"
df = pd.read_csv(file_path)

bacteria = df.drop(columns=["condition", "Country", "Dataset"])

print("===== FILTERED DATASET VERIFICATION =====")
print("Dataset shape:", df.shape)
print("Total samples:", len(df))
print("Bacterial features:", bacteria.shape[1])
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nClass distribution:")
print(df["condition"].value_counts())

print("\nDataset-wise sample count:")
print(df["Dataset"].value_counts())
