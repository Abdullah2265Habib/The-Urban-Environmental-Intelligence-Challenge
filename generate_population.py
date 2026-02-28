"""
Generate random population data for each location in the dataset
"""
import os
import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

# Load all datasets
folder_path = "dataset"
dataframes = []

for file in os.listdir(folder_path):
    if file.startswith("openaq") and file.endswith(".csv"):
        full_path = os.path.join(folder_path, file)
        dfm = pd.read_csv(full_path)
        dataframes.append(dfm)

df = pd.concat(dataframes, ignore_index=True)

# Get unique locations
locations = df[['location_id', 'location_name', 'country_iso', 'latitude', 'longitude']].drop_duplicates()

# Generate random population density for each location
# Population ranges from 50,000 to 5,000,000 with realistic distribution
# Lower population for rural/mixed areas, higher for urban areas
population_values = np.random.lognormal(mean=11, sigma=1.5, size=len(locations))
population_values = np.clip(population_values, 50000, 5000000).astype(int)

locations['population'] = population_values

# Save to CSV
output_path = "population.csv"
locations.to_csv(output_path, index=False)

print(f"Population data saved to {output_path}")
print(f"\nSummary Statistics:")
print(f"Total locations: {len(locations)}")
print(f"Population range: {locations['population'].min():,} to {locations['population'].max():,}")
print(f"Mean population: {locations['population'].mean():,.0f}")
print(f"Median population: {locations['population'].median():,.0f}")
print(f"\nFirst few rows:")
print(locations.head(10))
