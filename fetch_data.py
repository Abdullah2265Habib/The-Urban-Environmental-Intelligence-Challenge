import requests
import pandas as pd
import os
import time

#CONFIGURATION
API_KEY = ""
BASE_URL = "https://api.openaq.org/v3"
OUTPUT_FOLDER = "openaq_data"
YEAR = "2025"

DATE_FROM = f"{YEAR}-01-01T00:00:00Z"
DATE_TO = f"{YEAR}-12-31T23:59:59Z"

HEADERS = {"X-API-Key": API_KEY}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Fetching locations...")
locations_response = requests.get(
    f"{BASE_URL}/locations",
    headers=HEADERS,
    params={"limit": 200}
)

locations_response.raise_for_status()
locations = locations_response.json()["results"]

print(f"Fetched {len(locations)} locations")


for location in locations:

    location_id = location["id"]
    print(f"\nProcessing location {location_id}")

    # Get sensors for this location
    sensors_response = requests.get(
        f"{BASE_URL}/locations/{location_id}/sensors",
        headers=HEADERS
    )

    if sensors_response.status_code != 200:
        print("No sensors found.")
        continue

    sensors = sensors_response.json().get("results", [])

    all_data = []


    for sensor in sensors:
        sensor_id = sensor["id"]
        print(f"  Fetching sensor {sensor_id}")

        page = 1

        while True:
            response = requests.get(
                f"{BASE_URL}/sensors/{sensor_id}/measurements",
                headers=HEADERS,
                params={
                    "date_from": DATE_FROM,
                    "date_to": DATE_TO,
                    "limit": 1000,
                    "page": page
                }
            )

            if response.status_code != 200:
                break

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            all_data.extend(results)

            meta = data.get("meta", {})
            if page >= meta.get("totalPages", 1):
                break

            page += 1
            time.sleep(0.5)

    if all_data:
        df = pd.json_normalize(all_data)
        file_path = os.path.join(OUTPUT_FOLDER, f"station_{location_id}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}")
    else:
        print("No measurement data found.")

print("\nFinished successfully!")
