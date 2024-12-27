import pandas as pd
import json

# Load the spreadsheet
file_path = "venusdataset.xlsx"  # Replace with your file path
df = pd.read_excel(file_path, engine="openpyxl")

# Process profiles
def extract_profiles(df, num_profiles=5):
    profiles = []
    for _, row in df.iterrows():
        profile = {
            "age": row.get("age", "Unknown"),
            "gender": row.get("sex", "Unknown"),
            "orientation": row.get("orientation", "Unknown"),
            "body_type": row.get("body_type", "Unknown"),
            "location": row.get("location", "Unknown"),
            "description": row.get("essay0", "No description available.")
        }
        profiles.append(profile)
        if len(profiles) >= num_profiles:
            break  # Limit the number of profiles to prevent clutter in the Playground
    return profiles

# Extract the first 5 profiles for testing
profiles = extract_profiles(df, num_profiles=5)

# Format profiles as JSON for easy use in prompts
print(json.dumps(profiles, indent=4))
