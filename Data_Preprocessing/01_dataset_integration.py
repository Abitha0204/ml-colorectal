
import pandas as pd
from pathlib import Path

# Project folders
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Dataset" / "Raw"
OUTPUT = ROOT / "Dataset" / "Processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(RAW.glob("P003.MArumugam.Data.*_species.tsv"))

if len(files) != 8:
    raise ValueError(f"Expected 8 files, found {len(files)}")

country_names = {
    "AT": "Austria",
    "CN": "China",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "JP": "Japan",
    "US": "USA"
}

dataframes = []

for file in files:
    df = pd.read_csv(file, sep="\t")

    dataset_id = file.stem.replace(
        "P003.MArumugam.Data.", ""
    ).replace("_species", "")

    country_code = dataset_id.split("-")[0]

    df["Country"] = country_names[country_code]
    df["Dataset"] = dataset_id

    dataframes.append(df)

# Check feature consistency
reference_columns = list(dataframes[0].columns[:-2])

for df in dataframes:
    if list(df.columns[:-2]) != reference_columns:
        raise ValueError("Column mismatch detected!")

# Combine datasets
merged_df = pd.concat(dataframes, ignore_index=True)

# Check duplicates without deleting anything
duplicate_count = merged_df.duplicated(
    subset=reference_columns
).sum()

print("Merged dataset shape:", merged_df.shape)
print("\nClass distribution:")
print(merged_df["condition"].value_counts())

print("\nCountry distribution:")
print(merged_df["Country"].value_counts())

print("\nExact duplicate rows:", duplicate_count)
print("Missing values:", merged_df.isna().sum().sum())

# Save merged dataset
merged_df.to_csv(
    OUTPUT / "CRC_Merged_Initial.csv",
    index=False
)

merged_df.to_excel(
    OUTPUT / "CRC_Merged_Initial.xlsx",
    index=False
)

print("\nDataset integration completed!")
