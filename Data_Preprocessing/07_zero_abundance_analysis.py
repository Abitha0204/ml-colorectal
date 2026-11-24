
import pandas as pd
from pathlib import Path

# Load merged dataset
input_file = Path("Dataset/Processed/CRC_Merged_Initial.csv")
df = pd.read_csv(input_file)

# Identify bacterial features
metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

# Calculate zero counts and percentages
zero_counts = (df[bacteria] == 0).sum()
zero_percentages = (zero_counts / len(df)) * 100

# Create result table
zero_summary = pd.DataFrame({
    "Bacteria": bacteria,
    "Zero_Count": zero_counts.values,
    "Zero_Percentage": zero_percentages.values
})

zero_summary = zero_summary.sort_values(
    by="Zero_Percentage", ascending=False
)

# Overall statistics
print("\n===== ZERO ABUNDANCE ANALYSIS =====")
print("Total samples:", len(df))
print("Total bacterial features:", len(bacteria))
print("Total zero values:", (df[bacteria] == 0).sum().sum())
print("Average zero percentage:",
      round(zero_percentages.mean(), 2), "%")

print("\nTop 20 bacteria with highest zero percentage:")
print(zero_summary.head(20).to_string(index=False))

# Save results
output_file = Path(
    "Dataset/Processed/Zero_Abundance_Summary.csv"
)
zero_summary.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Zero-abundance analysis completed!")
