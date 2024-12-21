import pandas as pd
import json

# Ensure you have the necessary library installed
# Run this in your terminal if it's not already installed:
# pip install openpyxl

# Load the first 100 rows of the Excel file
file_path = "venusdataset.xlsx"  # Adjust the file name if necessary
df = pd.read_excel(file_path, engine='openpyxl', nrows=100)

# Define possible phrases for variation
phrases_start = [
    "I'm looking for",
    "I want",
    "Seeking",
    "Hoping to meet",
    "Prefer someone who is",
    "Searching for"
]

phrases_end = [
    "Prefer someone from",
    "Ideally located in",
    "Hoping they are from",
    "Someone residing in",
    "Based near"
]

# Define a function to format data into JSONL format
def format_data(row):
    prompt = (
        f"Looking for a {row['age']} year-old, {row['sex']}, {row['orientation']}, "
        f"who is {row['body_type']} and enjoys '{str(row['essay0'])[:50]}...'. "
        f"Prefer someone from {row['location']}."
    )
    completion = row['essay0'] if pd.notna(row['essay0']) else "No description available."
    return {"prompt": prompt, "completion": completion}

# Drop rows where 'essay0' or critical columns are missing
df = df.dropna(subset=['age', 'sex', 'orientation', 'body_type', 'location', 'essay0'])

# Generate JSONL data for the first 100 entries
jsonl_data = [format_data(row) for _, row in df.iterrows()]

# Save to a JSONL file
output_file = "formatted_profiles_100.jsonl"
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"Data successfully saved to {output_file}")
