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
        return value.strftime("%Y-%m-%d-%H-%M")
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

def clean_height(height):
    try:
        if pd.isna(height):
            return None
        height_val = int(float(height))
        if 48 <= height_val <= 96:  # Basic height validation (4ft to 8ft)
            return height_val
        return None
    except (ValueError, TypeError):
        return None

def clean_income(income):
    try:
        if pd.isna(income):
            return -1
        income_val = int(float(income))
        return income_val
    except (ValueError, TypeError):
        return -1

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
            "status": clean_value(row.get('status')),
            "sex": clean_value(row.get('sex')),
            "orientation": clean_value(row.get('orientation')),
            "body_type": clean_value(row.get('body_type')),
            "diet": clean_value(row.get('diet')),
            "drinks": clean_value(row.get('drinks')),
            "drugs": clean_value(row.get('drugs')),
            "education": clean_value(row.get('education')),
            "ethnicity": clean_value(row.get('ethnicity')),
            "height": clean_height(row.get('height')),
            "income": clean_income(row.get('income')),
            "job": clean_value(row.get('job')),
            "last_online": clean_value(row.get('last_online')),
            "location": clean_value(row.get('location')),
            "offspring": clean_value(row.get('offspring')),
            "pets": clean_value(row.get('pets')),
            "religion": clean_value(row.get('religion')),
            "sign": clean_value(row.get('sign')),
            "smokes": clean_value(row.get('smokes')),
            "speaks": clean_value(row.get('speaks'))
        }
        
        # Add all essays
        for i in range(10):
            essay_key = f'essay{i}'
            essay_value = clean_value(row.get(essay_key))
            if essay_value:  # Only add non-empty essays
                profile[essay_key] = essay_value
        
        # Only add profiles with valid required fields
        if (profile["age"] and 
            profile["sex"] and 
            profile["orientation"] and 
            all(profile[field] is not None for field in ["sex", "orientation"])):
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
