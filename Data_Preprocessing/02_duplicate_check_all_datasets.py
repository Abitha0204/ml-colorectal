
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

file_path = (
    ROOT / "Dataset" / "Processed" / "CRC_Merged_Initial.csv"
)

df = pd.read_csv(file_path)

# Separate bacterial features and metadata
metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

print("Total samples:", len(df))
print("Bacterial features:", len(bacteria))

# Find exact matching bacterial profiles
duplicate_mask = df.duplicated(
    subset=bacteria,
    keep=False
)

duplicates = df.loc[
    duplicate_mask,
    ["condition", "Country", "Dataset"]
].copy()

duplicates.insert(
    0, "Original_Row",
    df.index[duplicate_mask] + 2
)

print("\nTotal rows with repeated profiles:", len(duplicates))

if duplicates.empty:
    print("No exact repeated bacterial profiles found.")
else:
    print("\nDuplicate profile details:")
    print(duplicates.to_string(index=False))

    print("\nRepeated profiles by dataset:")
    print(duplicates.groupby("Dataset").size())

    print("\nRepeated profiles by country:")
    print(duplicates.groupby("Country").size())

# Save report
output = ROOT / "Dataset" / "Processed"
duplicates.to_csv(
    output / "Duplicate_Profile_Report.csv",
    index=False
)

print("\nDuplicate verification completed!")
print("Report saved successfully.")
