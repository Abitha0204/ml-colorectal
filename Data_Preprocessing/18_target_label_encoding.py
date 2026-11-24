
import pandas as pd

file_path = "Dataset/Processed/CRC_Prevalence_Filtered_10pct.csv"
df = pd.read_csv(file_path)

df["condition"] = df["condition"].map({
    "control": 0,
    "CRC": 1
})

print("===== TARGET LABEL ENCODING =====")
print(df["condition"].value_counts().sort_index())
print("\nLabel mapping:")
print("Control = 0")
print("CRC = 1")

print("\nMissing labels:", df["condition"].isnull().sum())

output_path = "Dataset/Processed/CRC_Encoded_Labels.csv"
df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
