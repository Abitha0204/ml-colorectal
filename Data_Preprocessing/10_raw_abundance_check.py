
import pandas as pd
from pathlib import Path

raw_folder = Path("Dataset/Raw")

print("\n===== RAW DATA ABUNDANCE CHECK =====")

for file in sorted(raw_folder.glob("*.tsv")):
    df = pd.read_csv(file, sep="\t")

    bacteria = df.drop(columns=["condition"], errors="ignore")
    bacteria = bacteria.apply(pd.to_numeric, errors="coerce")

    row_sums = bacteria.sum(axis=1)

    print(f"\nDataset: {file.name}")
    print("Samples:", len(df))
    print("Bacterial features:", bacteria.shape[1])
    print("First 5 sample sums:")
    print(row_sums.head().round(6).to_string(index=False))
    print("Average abundance sum:", round(row_sums.mean(), 6))
    print("Minimum abundance sum:", round(row_sums.min(), 6))
    print("Maximum abundance sum:", round(row_sums.max(), 6))
