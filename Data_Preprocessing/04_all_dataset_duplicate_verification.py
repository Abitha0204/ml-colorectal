
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load merged dataset
df = pd.read_csv("Dataset/Processed/CRC_Merged_Initial.csv")

metadata = ["condition", "Country", "Dataset"]
bacteria = [col for col in df.columns if col not in metadata]

X = df[bacteria].to_numpy(dtype=float)

print("ALL DATASET DUPLICATE VERIFICATION")
print("---------------------------------")
print("Total samples:", len(df))
print("Bacterial features:", len(bacteria))

# 1. Exact duplicate check
exact_mask = df.duplicated(subset=bacteria, keep=False)
exact_duplicates = df.loc[exact_mask]

exact_duplicates.to_csv(
    "Dataset/Processed/All_Datasets_Exact_Duplicates.csv",
    index=False
)

print("\nExact duplicate rows:", len(exact_duplicates))

# 2. Fast cosine similarity calculation
print("\nCalculating cosine similarity...")

similarity = cosine_similarity(X)

# Get upper triangle pairs only (exclude self-comparisons)
i, j = np.triu_indices(len(df), k=1)

# Create results directly using NumPy arrays
result_df = pd.DataFrame({
    "Sample_1_Index": df.index.to_numpy()[i],
    "Sample_2_Index": df.index.to_numpy()[j],
    "Dataset_1": df["Dataset"].to_numpy()[i],
    "Dataset_2": df["Dataset"].to_numpy()[j],
    "Condition_1": df["condition"].to_numpy()[i],
    "Condition_2": df["condition"].to_numpy()[j],
    "Cosine_Similarity": similarity[i, j]
})

# Sort similarity scores
result_df = result_df.sort_values(
    "Cosine_Similarity",
    ascending=False
)

# 3. High similarity candidates
candidates = result_df[
    result_df["Cosine_Similarity"] >= 0.999
]

print("\nTotal pairs compared:", len(result_df))
print("High similarity candidates:", len(candidates))

print("\nTOP 20 SIMILAR PAIRS:")
print(result_df.head(20).to_string(index=False))

# Save reports
result_df.head(100).to_csv(
    "Dataset/Processed/All_Datasets_Top100_Similar_Pairs.csv",
    index=False
)

candidates.to_csv(
    "Dataset/Processed/All_Datasets_Near_Duplicate_Candidates.csv",
    index=False
)

print("\nVerification completed successfully!")
print("Reports saved in Dataset/Processed/")
