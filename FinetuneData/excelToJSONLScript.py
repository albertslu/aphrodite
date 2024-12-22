import pandas as pd
import json
import random

# Load the first 20 rows of the Excel file
file_path = "venusdataset.xlsx"  # Adjust the file name if necessary
df = pd.read_excel(file_path, engine='openpyxl', nrows=100)

# Normalize gender terms
gender_map = {
    "m": "male", "man": "male", "boy": "male", "men": "male", "male": "male",
    "f": "female", "woman": "female", "girl": "female", "women": "female", "female": "female"
}

# Define possible phrases for variation
phrases_start = [
    "Looking for",
    "I want to meet",
    "Seeking someone who is",
    "Hoping to find",
    "Dreaming of meeting",
    "Searching for"
]

# Define traits, hobbies, and other descriptors for variety
traits = [
    "fun-loving",
    "intellectual",
    "adventurous",
    "kind-hearted",
    "goal-oriented",
    "creative",
    "spontaneous",
    "outgoing",
    "introverted"
]

physical_features = [
    "tall",
    "short",
    "athletic",
    "curvy",
    "slim",
    "with dark hair",
    "with red hair",
    "with bright eyes"
]

# Random templates for prompts
prompt_templates = [
    "{start_phrase} a {age} year-old {gender} who is {trait} and {physical}.",
    "{start_phrase} someone {physical} and {trait}, aged {age}.",
    "{start_phrase} a partner who is {trait} and {physical}.",
    "{start_phrase} a {age} year-old, {gender}, {orientation} partner.",
    "{start_phrase} someone who is {physical} and loves {random_hobby}."
]

# Function to infer hobbies or add randomness
def infer_random_hobby():
    hobbies = ["hiking", "reading", "dancing", "cooking", "traveling", "exploring nature", "fitness"]
    return random.choice(hobbies)

# Define a function to normalize gender
def normalize_gender(gender):
    return gender_map.get(gender.lower(), "unknown")  # Map gender or use "unknown" if not in the map

# Define a function to format data into JSONL format
def format_data(row):
    # Randomly select attributes and phrases
    start_phrase = random.choice(phrases_start)
    trait = random.choice(traits)
    physical = random.choice(physical_features)
    random_hobby = infer_random_hobby()
    template = random.choice(prompt_templates)

    # Normalize gender
    normalized_gender = normalize_gender(row["sex"])

    # Fill in the template with attributes
    prompt = template.format(
        start_phrase=start_phrase,
        age=row["age"],
        gender=normalized_gender,
        orientation=row["orientation"],
        trait=trait,
        physical=physical,
        random_hobby=random_hobby
    )
    
    # Include the entire row (multiple fields) in the completion
    completion = {
        "age": row["age"],
        "sex": normalized_gender,
        "orientation": row["orientation"],
        "body_type": row["body_type"],
        "location": row["location"],
        "essay0": row["essay0"] if pd.notna(row["essay0"]) else "",
        "essay1": row["essay1"] if pd.notna(row["essay1"]) else ""
    }
    
    return {"prompt": prompt, "completion": json.dumps(completion)}

# Drop rows where critical columns are missing, and log the dropped rows
initial_count = len(df)
df = df.dropna(subset=['age', 'sex', 'orientation', 'body_type', 'location', 'essay0'])
final_count = len(df)

print(f"Rows dropped due to missing values: {initial_count - final_count}")

# Generate JSONL data for the first 20 entries
jsonl_data = [format_data(row) for _, row in df.iterrows()]

# Save to a JSONL file
output_file = "formatted_profiles_100_normalized.jsonl"
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"Data successfully saved to {output_file}")
