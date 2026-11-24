
import pandas as pd
from pathlib import Path

# Load merged dataset
input_file = Path("Dataset/Processed/CRC_Merged_Initial.csv")
df = pd.read_csv(input_file)

# Identify bacterial features
metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

# Calculate prevalence for each dataset
results = []

for dataset_name, group in df.groupby("Dataset"):

    total_samples = len(group)

    present_counts = (group[bacteria] > 0).sum()

    prevalence = (present_counts / total_samples) * 100

    for bacterium in bacteria:
        results.append({
            "Dataset": dataset_name,
            "Total_Samples": total_samples,
            "Bacteria": bacterium,
            "Present_Count": present_counts[bacterium],
            "Prevalence_Percentage": prevalence[bacterium]
        })

# Create results dataframe
prevalence_df = pd.DataFrame(results)

# Identify features below 10% in each dataset
prevalence_df["Below_10_Percent"] = (
    prevalence_df["Prevalence_Percentage"] < 10
)

# Display summary
print("\n===== DATASET-WISE PREVALENCE ANALYSIS =====")

for dataset_name, group in prevalence_df.groupby("Dataset"):

    total_samples = group["Total_Samples"].iloc[0]

    below_10 = group["Below_10_Percent"].sum()

    above_10 = len(group) - below_10

    print("\nDataset:", dataset_name)
    print("Total samples:", total_samples)
    print("Total bacterial features:", len(bacteria))
    print("Below 10% prevalence:", below_10)
    print("10% or above prevalence:", above_10)

# Save complete results
output_file = Path(
    "Dataset/Processed/Dataset_Wise_Prevalence.csv"
)

prevalence_df.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Dataset-wise prevalence analysis completed!")
