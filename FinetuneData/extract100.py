import pandas as pd
import json
import random

# Load the spreadsheet
file_path = "venusdataset.xlsx"  # Replace with your local file path
df = pd.read_excel(file_path, engine="openpyxl")

# Process profiles
def extract_profiles_with_conversion(df, num_profiles=250):
    # Get all valid profiles first
    all_profiles = []
    for i, row in df.iterrows():
        profile = {}
        for key, value in row.items():
            if pd.notna(value) and value != "":
                # Convert datetime to string if necessary
                if isinstance(value, pd.Timestamp):
                    profile[key] = value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    profile[key] = value
        if profile:  # Only add non-empty profiles
            all_profiles.append(profile)
    
    # Randomly sample num_profiles from all valid profiles
    if len(all_profiles) > num_profiles:
        return random.sample(all_profiles, num_profiles)
    return all_profiles

print(f"Total rows in Excel: {len(df)}")

# Extract 500 random profiles
profiles_converted = extract_profiles_with_conversion(df, num_profiles=500)

# Save profiles to a JSON file
output_file = "extracted_500_random_profiles.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(profiles_converted, f, indent=2, ensure_ascii=False)

print(f"\nSuccessfully extracted {len(profiles_converted)} profiles to {output_file}")
