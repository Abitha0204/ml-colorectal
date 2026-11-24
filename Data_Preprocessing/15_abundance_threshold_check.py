
import pandas as pd
import numpy as np

file_path = "Dataset/Processed/CRC_Merged_Initial.csv"
df = pd.read_csv(file_path)

bacteria = df.drop(columns=["condition", "Country", "Dataset"])

threshold = 1e-15

positive_values = bacteria.to_numpy()
positive_values = positive_values[positive_values > 0]

below_threshold = np.sum(positive_values < threshold)

print("===== ABUNDANCE THRESHOLD CHECK =====")
print("Total bacterial features:", bacteria.shape[1])
print("Total positive abundance values:", len(positive_values))
print("Values below 1e-15:", below_threshold)

if len(positive_values) > 0:
    print("Minimum positive abundance:", positive_values.min())

print("No changes made to the original dataset.")
