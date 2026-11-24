
import pandas as pd
import numpy as np

file_path = "Dataset/Processed/CRC_Prevalence_Filtered_10pct.csv"
df = pd.read_csv(file_path)

bacteria = df.drop(columns=["condition", "Country", "Dataset"])

total_values = bacteria.size
zero_values = (bacteria == 0).sum().sum()

print("===== NORMALIZATION PREPARATION =====")
print("Samples:", len(df))
print("Bacterial features:", bacteria.shape[1])
print("Total abundance values:", total_values)
print("Zero abundance values:", zero_values)
print("Zero percentage:", round(zero_values / total_values * 100, 2), "%")
print("Minimum abundance:", bacteria.to_numpy().min())
print("Maximum abundance:", bacteria.to_numpy().max())
