import pandas as pd
import json
import random

# Load the first 100 rows of the Excel file
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
    "Hoping to find",
    "Dreaming of meeting",
    "Searching for"
]

# Define traits, hobbies, physical features, and education terms for variety
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

education_terms = [
    "college-educated",
    "with a college degree",
    "holding a master's degree",
    "highly educated",
    "currently pursuing a degree",
    "with a strong educational background"
]

# Refined prompt templates with optional education
prompt_templates = [
    "{start_phrase} someone {physical}{education} and {trait}, aged {age_range}.",
    "{start_phrase} a {age_range} {gender} who is {ethnicity} and {trait}.",
    "{start_phrase} a {age_range} {gender}, {orientation} partner who is {trait}.",
    "{start_phrase} a {age_range} {gender} with {physical} looks{education} and {ethnicity}.",
    "{start_phrase} someone who loves {random_hobby}{education}, aged {age_range}."
]

# Function to infer hobbies or add randomness
def infer_random_hobby():
    hobbies = ["hiking", "reading", "dancing", "cooking", "traveling", "exploring nature", "fitness"]
    return random.choice(hobbies)

# Define a function to normalize gender
def normalize_gender(gender):
    return gender_map.get(gender.lower(), "unknown")  # Map gender or use "unknown" if not in the map

# Define a function to generate age range
def generate_age_range(age):
    lower_bound = max(age - random.randint(1, 5), 18)  # Ensure age is not below 18
    upper_bound = age + random.randint(1, 5)
    return f"{lower_bound}-{upper_bound}"

# Define a function to normalize ethnicity
def normalize_ethnicity(ethnicity):
    return ethnicity if isinstance(ethnicity, str) and ethnicity.strip() else "unknown"

# Function to decide if education is included
def include_education(row):
    return random.choice([True, False]) and pd.notna(row["education"])

# Function to clean and format prompts
def clean_prompt(prompt):
    # Remove multiple spaces, handle trailing commas or conjunctions
    prompt = prompt.replace(", ,", ",").replace(" ,", ",").strip()
    if ", and" in prompt:
        prompt = prompt.replace(", and", " and")
    if prompt.endswith(","):
        prompt = prompt[:-1]
    return prompt

# Define a function to format data into JSONL format
def format_data(row):
    # Randomly select attributes and phrases
    start_phrase = random.choice(phrases_start)
    trait = random.choice(traits)
    physical = random.choice(physical_features)
    random_hobby = infer_random_hobby()

    # Normalize gender and ethnicity
    normalized_gender = normalize_gender(row["sex"])
    normalized_ethnicity = normalize_ethnicity(row["ethnicity"])

    # Generate age range
    age_range = generate_age_range(row["age"])

    # Optionally include education
    education = f", {random.choice(education_terms)}" if include_education(row) else ""

    # Select a random template and format it
    template = random.choice(prompt_templates)
    prompt = template.format(
        start_phrase=start_phrase,
        age_range=age_range,
        gender=normalized_gender,
        orientation=row["orientation"],
        trait=trait,
        physical=physical,
        ethnicity=normalized_ethnicity,
        education=education,
        random_hobby=random_hobby
    )

    # Clean up the prompt for grammatical correctness
    prompt = clean_prompt(prompt)

    # Include the entire row (multiple fields) in the completion
    completion = {
        "age": row["age"],
        "sex": normalized_gender,
        "orientation": row["orientation"],
        "ethnicity": normalized_ethnicity,
        "body_type": row["body_type"],
        "location": row["location"],
        "education": row["education"] if pd.notna(row["education"]) else "",
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
output_file = "formatted_profiles_cleaned.jsonl"
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"Data successfully saved to {output_file}")
