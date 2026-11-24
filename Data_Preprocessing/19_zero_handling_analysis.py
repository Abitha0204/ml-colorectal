
import pandas as pd
import numpy as np

file_path = "Dataset/Processed/CRC_Prevalence_Filtered_10pct.csv"
df = pd.read_csv(file_path)

bacteria = df.drop(columns=["condition", "Country", "Dataset"])

values = bacteria.to_numpy()
positive_values = values[values > 0]

row_sums = bacteria.sum(axis=1)

print("===== ZERO HANDLING ANALYSIS =====")
print("Minimum positive abundance:", positive_values.min())
print("Maximum abundance:", positive_values.max())
print("Median positive abundance:", np.median(positive_values))
print("Total zero values:", np.sum(values == 0))
print("Samples with all-zero bacterial values:", np.sum(row_sums == 0))
print("Minimum sample abundance sum:", row_sums.min())
print("Maximum sample abundance sum:", row_sums.max())
