import os
import pandas as pd

INPUT_FOLDER = "openaq_data"
OUTPUT_FILE = "combined_data.csv"

csv_files = [
    f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")
]


if not csv_files:
    print("No CSV files found in", INPUT_FOLDER)
    exit()

print(f"Found {len(csv_files)} files. Merging...")


df_list = []
for filename in csv_files:
    filepath = os.path.join(INPUT_FOLDER, filename)
    df = pd.read_csv(filepath)
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)


combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"Combined CSV saved as: {OUTPUT_FILE}")
