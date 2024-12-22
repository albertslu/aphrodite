import pandas as pd
import json
import random

# Ensure you have the necessary library installed
# pip install openpyxl

# Load the first 100 rows of the Excel file
file_path = "venusdataset.xlsx"  # Adjust the file name if necessary
df = pd.read_excel(file_path, engine='openpyxl', nrows=100)

# Define possible phrases for variation
phrases_start = [
    "Looking for",
    "I want to meet",
    "Seeking someone who is",
    "Hoping to find",
    "Would love to meet",
    "Searching for",
    "Dreaming of meeting"
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

# Define a function to format data into JSONL format
def format_data(row):
    # Randomly select attributes and phrases
    start_phrase = random.choice(phrases_start)
    trait = random.choice(traits)
    physical = random.choice(physical_features)
    random_hobby = infer_random_hobby()
    template = random.choice(prompt_templates)

    # Fill in the template with attributes
    prompt = template.format(
        start_phrase=start_phrase,
        age=row["age"],
        gender=row["sex"],
        orientation=row["orientation"],
        trait=trait,
        physical=physical,
        random_hobby=random_hobby
    )
    
    # The completion is just the essay (description of the profile)
    completion = row['essay0'] if pd.notna(row['essay0']) else "No description available."
    return {"prompt": prompt, "completion": completion}

# Drop rows where 'essay0' or critical columns are missing
df = df.dropna(subset=['age', 'sex', 'orientation', 'essay0'])

# Generate JSONL data for the first 100 entries
jsonl_data = [format_data(row) for _, row in df.iterrows()]

# Save to a JSONL file
output_file = "formatted_profiles_100_randomized.jsonl"
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"Data successfully saved to {output_file}")
