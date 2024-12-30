import pandas as pd
import json
import random

# Load the spreadsheet
file_path = "venusdataset.xlsx"  # Replace with your local file path
df = pd.read_excel(file_path, engine="openpyxl")
print(f"Total rows in Excel: {len(df)}")

def clean_value(value):
    if pd.isna(value) or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value).lower().strip()

def clean_age(age):
    try:
        if pd.isna(age):
            return None
        age_val = int(float(age))
        if 18 <= age_val <= 100:  # Basic age validation
            return age_val
        return None
    except (ValueError, TypeError):
        return None

def combine_essays(row):
    essays = []
    for i in range(10):  # essays 0-9
        essay = clean_value(row.get(f'essay{i}'))
        if essay:
            essays.append(essay)
    return ' '.join(essays) if essays else None

# Process profiles with better cleaning and validation
def extract_profiles_with_conversion(df, num_profiles=250):
    profiles = []
    invalid_count = 0
    for _, row in df.iterrows():
        age = clean_age(row.get('age'))
        if age is None:
            invalid_count += 1
            continue
            
        profile = {
            "age": age,
            "gender": clean_value(row.get('sex')),  # Changed from gender to sex
            "orientation": clean_value(row.get('orientation')),
            "ethnicity": clean_value(row.get('ethnicity')),
            "height": clean_value(row.get('height')),
            "body_type": clean_value(row.get('body_type')),
            "education": clean_value(row.get('education')),
            "occupation": clean_value(row.get('job')),  # Changed from occupation to job
            "about": combine_essays(row)  # Combine all essays into about field
        }
        
        # Only add profiles with valid required fields
        if (profile["age"] and 
            profile["gender"] and 
            profile["orientation"] and 
            all(profile[field] is not None for field in ["gender", "orientation"])):
            profiles.append(profile)
        else:
            invalid_count += 1
    
    print(f"Found {len(profiles)} valid profiles")
    print(f"Skipped {invalid_count} invalid profiles")
    
    # Randomly select num_profiles if we have more than that
    if len(profiles) > num_profiles:
        profiles = random.sample(profiles, num_profiles)
    
    return profiles

# Print the column names to verify we're using correct names
print("\nAvailable columns:")
print(df.columns.tolist())

# Extract profiles with conversion
profiles_converted = extract_profiles_with_conversion(df)

# Save profiles to a JSON file
output_file = "extracted_250_profiles.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(profiles_converted, f, indent=2, ensure_ascii=False)

print(f"\nSuccessfully extracted {len(profiles_converted)} profiles to {output_file}")
