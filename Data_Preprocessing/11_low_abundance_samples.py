
import pandas as pd
from pathlib import Path

raw_folder = Path("Dataset/Raw")

print("\n===== LOW ABUNDANCE SAMPLE INVESTIGATION =====")

for file in sorted(raw_folder.glob("*.tsv")):
    df = pd.read_csv(file, sep="\t")

    bacteria = df.drop(columns=["condition"], errors="ignore")
    bacteria = bacteria.apply(pd.to_numeric, errors="coerce")

    sums = bacteria.sum(axis=1)

    low = df.loc[sums < 0.5, ["condition"]].copy()
    low["Abundance_Sum"] = sums[sums < 0.5]

    if len(low) > 0:
        print(f"\nDataset: {file.name}")
        print(low.to_string())
    else:
        print(f"\n{file.name}: No samples below 0.5")
