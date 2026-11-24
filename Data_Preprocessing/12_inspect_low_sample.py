
import pandas as pd

file = "Dataset/Raw/P003.MArumugam.Data.IT-CRC_species.tsv"

df = pd.read_csv(file, sep="\t")

row = df.iloc[29]

bacteria = row.drop("condition").astype(float)

print("===== LOW ABUNDANCE SAMPLE DETAILS =====")
print("Sample index: 29")
print("Condition:", row["condition"])
print("Total abundance:", bacteria.sum())
print("Non-zero bacterial features:", (bacteria > 0).sum())
print("Zero bacterial features:", (bacteria == 0).sum())
print("Maximum abundance:", bacteria.max())
print("\nTop 10 bacterial abundances:")
print(bacteria.sort_values(ascending=False).head(10))
