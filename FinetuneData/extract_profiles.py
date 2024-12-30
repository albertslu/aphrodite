import pandas as pd
import json
import random

def clean_age(age):
    try:
        return int(float(age))
    except (ValueError, TypeError):
        return None

def extract_profiles(excel_path, num_profiles=250):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Clean and prepare the data
    profiles = []
    for _, row in df.iterrows():
        age = clean_age(row.get('age'))
        if age is None:
            continue
            
        profile = {
            "age": age,
            "gender": str(row.get('gender', '')).lower(),
            "orientation": str(row.get('orientation', '')).lower(),
            "ethnicity": str(row.get('ethnicity', '')).lower(),
            "height": str(row.get('height', '')).lower(),
            "body_type": str(row.get('body_type', '')).lower(),
            "education": str(row.get('education', '')).lower(),
            "occupation": str(row.get('occupation', '')).lower(),
            "interests": str(row.get('interests', '')).lower(),
            "about": str(row.get('about', '')).lower()
        }
        
        # Only add profiles with valid required fields
        if (profile["age"] and 
            profile["gender"] and 
            profile["orientation"] and 
            all(profile[field] != 'nan' for field in ["gender", "orientation"])):
            profiles.append(profile)
    
    # Randomly select num_profiles if we have more than that
    if len(profiles) > num_profiles:
        profiles = random.sample(profiles, num_profiles)
    
    return profiles

def main():
    excel_path = "venusdataset.xlsx"  # Update this path if needed
    profiles = extract_profiles(excel_path)
    
    # Save to JSON file
    output_file = "extracted_250_profiles.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully extracted {len(profiles)} profiles to {output_file}")

if __name__ == "__main__":
    main()
