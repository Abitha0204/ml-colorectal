
import pandas as pd

# Load merged dataset
df = pd.read_csv("Dataset/Processed/CRC_Merged_Initial.csv")

# Select Italy samples
italy = df[df["Country"] == "Italy"].copy()

print("ITALY DATASET INSPECTION")
print("------------------------")

print("Total Italy samples:", len(italy))

print("\nDataset-wise sample count:")
print(italy["Dataset"].value_counts())

print("\nDataset-wise class distribution:")
print(pd.crosstab(italy["Dataset"], italy["condition"]))

print("\nMissing values:", italy.isnull().sum().sum())

print("\nItaly inspection completed!")


# Get bacterial feature columns
metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in italy.columns if col not in metadata]

# Separate the two Italy datasets
it1 = italy[italy["Dataset"] == "IT-CRC"]
it2 = italy[italy["Dataset"] == "IT-CRC-2"]

# Compare bacterial profiles between datasets
comparison = it1[bacteria].merge(
    it2[bacteria],
    how="inner",
    on=bacteria
)

print("\nCROSS-DATASET DUPLICATE CHECK")
print("-----------------------------")
print("IT-CRC samples:", len(it1))
print("IT-CRC-2 samples:", len(it2))
print("Matching bacterial profiles:", len(comparison))


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Convert bacterial abundance values into numeric format
X1 = it1[bacteria].to_numpy(dtype=float)
X2 = it2[bacteria].to_numpy(dtype=float)

# Calculate similarity between both datasets
similarity = cosine_similarity(X1, X2)

# Store all sample-pair comparisons
results = []

for i in range(len(it1)):
    for j in range(len(it2)):
        results.append({
            "IT_CRC_Index": it1.index[i],
            "IT_CRC2_Index": it2.index[j],
            "IT_CRC_Condition": it1.iloc[i]["condition"],
            "IT_CRC2_Condition": it2.iloc[j]["condition"],
            "Cosine_Similarity": similarity[i, j]
        })

# Sort from highest similarity to lowest
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(
    by="Cosine_Similarity",
    ascending=False
)

# Display top 20 similar pairs
print("\nTOP 20 MOST SIMILAR SAMPLE PAIRS")
print("--------------------------------")
print(result_df.head(20).to_string(index=False))

# Exploratory high-similarity candidates
candidates = result_df[
    result_df["Cosine_Similarity"] >= 0.999
]

print("\nHIGH SIMILARITY CANDIDATES (>= 0.999):")
print(len(candidates))

# Save comparison report
result_df.to_csv(
    "Dataset/Processed/Italy_Near_Duplicate_Report.csv",
    index=False
)

print("\nNear-duplicate comparison completed!")
