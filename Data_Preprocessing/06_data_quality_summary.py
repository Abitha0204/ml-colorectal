
import pandas as pd
from pathlib import Path

# Load merged dataset
file_path = Path("Dataset/Processed/CRC_Merged_Initial.csv")
df = pd.read_csv(file_path)

# Identify bacterial features
metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

# Calculate summary
total_samples = len(df)
total_features = len(bacteria)
missing_values = df.isnull().sum().sum()
duplicate_profiles = df.duplicated(subset=bacteria).sum()

class_counts = df["condition"].value_counts()
country_counts = df["Country"].value_counts()

# Display report
print("\n===== FINAL DATA QUALITY SUMMARY =====")
print("Total samples:", total_samples)
print("Total bacterial features:", total_features)
print("Total columns:", len(df.columns))
print("Missing values:", missing_values)
print("Exact duplicate bacterial profiles:", duplicate_profiles)

print("\nClass Distribution:")
print(class_counts)

print("\nCountry-wise Sample Distribution:")
print(country_counts)

# Save summary report
report = [
    ["Total Samples", total_samples],
    ["Bacterial Features", total_features],
    ["Total Columns", len(df.columns)],
    ["Missing Values", missing_values],
    ["Exact Duplicate Profiles", duplicate_profiles]
]

summary_df = pd.DataFrame(report, columns=["Metric", "Value"])

output_path = Path("Dataset/Processed/Final_Data_Quality_Summary.csv")
summary_df.to_csv(output_path, index=False)

print("\nSummary saved to:", output_path)
print("\nData quality summary completed!")
