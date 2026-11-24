
import pandas as pd

input_file = "Dataset/Processed/CRC_Merged_Initial.csv"
output_file = "Dataset/Processed/CRC_Prevalence_Filtered_10pct.csv"

df = pd.read_csv(input_file)

metadata = ["condition", "Country", "Dataset"]
bacteria = df.drop(columns=metadata, errors="ignore")

# Calculate prevalence
prevalence = (bacteria > 0).mean()

# Retain features present in at least 10% of samples
selected_features = prevalence[prevalence >= 0.10].index.tolist()

filtered_df = df[metadata + selected_features]

filtered_df.to_csv(output_file, index=False)

print("===== PREVALENCE FILTERING RESULTS =====")
print("Original samples:", len(df))
print("Original bacterial features:", bacteria.shape[1])
print("Removed features:", bacteria.shape[1] - len(selected_features))
print("Retained features:", len(selected_features))
print("Final dataset shape:", filtered_df.shape)
print("Saved:", output_file)
