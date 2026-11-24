
import pandas as pd
from pathlib import Path

raw_folder = Path("Dataset/Raw")

id_names = {
    "id", "sample_id", "sampleid",
    "patient_id", "patientid",
    "subject_id", "subjectid",
    "sample", "patient", "subject"
}

for file in sorted(raw_folder.glob("*.tsv")):
    df = pd.read_csv(file, sep="\t")

    candidates = [
        col for col in df.columns
        if col.strip().lower().replace(" ", "_") in id_names
    ]

    non_bacterial = [
        col for col in df.columns
        if col != "condition" and not col.endswith("__")
    ]

    print("\nFile:", file.name)
    print("Samples:", len(df))
    print("Columns:", len(df.columns))
    print("Possible ID columns:", candidates)
    print("Other non-bacterial columns:", non_bacterial)
