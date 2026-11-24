
import pandas as pd
from pathlib import Path

for file in sorted(Path("Dataset/Raw").glob("*.tsv")):
    df = pd.read_csv(file, sep="\t")
    X = df.drop(columns=["condition"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    sums = X.sum(axis=1)

    print(f"\n{file.name}")
    print("25th percentile:", sums.quantile(0.25))
    print("Median:", sums.median())
    print("75th percentile:", sums.quantile(0.75))
    print("Samples below 0.1:", (sums < 0.1).sum())
