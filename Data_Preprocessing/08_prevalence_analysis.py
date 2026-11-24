
import pandas as pd
from pathlib import Path

# Load dataset
df = pd.read_csv("Dataset/Processed/CRC_Merged_Initial.csv")

metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

# Calculate prevalence
present_counts = (df[bacteria] > 0).sum()
prevalence = (present_counts / len(df)) * 100

# Create summary
summary = pd.DataFrame({
    "Bacteria": bacteria,
    "Present_Count": present_counts.values,
    "Prevalence_Percentage": prevalence.values
})

summary = summary.sort_values(
    "Prevalence_Percentage"
)

# Identify features below 10% prevalence
low_prevalence = summary[
    summary["Prevalence_Percentage"] < 10
]

print("\n===== PREVALENCE ANALYSIS =====")
print("Total samples:", len(df))
print("Total bacterial features:", len(bacteria))
print("Features below 10% prevalence:", len(low_prevalence))
print("Features at or above 10% prevalence:",
      len(summary) - len(low_prevalence))

print("\nLowest 20 prevalence features:")
print(summary.head(20).to_string(index=False))

# Save report
summary.to_csv(
    "Dataset/Processed/Prevalence_Summary.csv",
    index=False
)

print("\nSaved: Dataset/Processed/Prevalence_Summary.csv")
print("Prevalence analysis completed!")
