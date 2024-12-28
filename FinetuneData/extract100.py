import pandas as pd
import json

# Load the spreadsheet
file_path = "venusdataset.xlsx"  # Replace with your local file path
df = pd.read_excel(file_path, engine="openpyxl")

# Process profiles
def extract_profiles_with_conversion(df, num_profiles=100):
    profiles = []
    for i, row in df.iterrows():
        profile = {}
        for key, value in row.items():
            if pd.notna(value) and value != "":
                # Convert datetime to string if necessary
                if isinstance(value, pd.Timestamp):
                    profile[key] = value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    profile[key] = value
        profiles.append(profile)
        if len(profiles) >= num_profiles:
            break  # Stop after extracting the specified number of profiles
    return profiles

# Extract the first 100 profiles
profiles_converted = extract_profiles_with_conversion(df, num_profiles=100)

# Save profiles to a JSON file
output_file = "extracted_100_profiles.json"
with open(output_file, "w") as f:
    json.dump(profiles_converted, f, indent=4)

print(f"Profiles successfully saved to {output_file}")
