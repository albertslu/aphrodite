import pandas as pd
import json

# Load the Excel file
file_path = "okcupid_profiles.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Define a function to format data into JSONL format
def format_data(row):
    prompt = (
        f"Looking for a {row['age']} year-old, {row['sex']}, {row['orientation']}, "
        f"who is {row['body_type']} and enjoys '{row['essay0'][:50]}...'. "
        f"Prefer someone from {row['location']}."
    )
    completion = row['essay0']
    return {"prompt": prompt, "completion": completion}

# Filter rows with valid descriptions
df = df.dropna(subset=['essay0'])

# Generate JSONL data
jsonl_data = [format_data(row) for _, row in df.iterrows()]

# Save to a JSONL file
output_file = "formatted_profiles.jsonl"
with open(output_file, 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"Data successfully saved to {output_file}")
