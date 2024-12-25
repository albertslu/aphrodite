import openai
import json
import pandas as pd

openai.api_key = "YOUR_API_KEY"

# Load JSONL prompts
with open("formatted_profiles_100.jsonl", "r") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

# Load Excel spreadsheet
file_path = "venusdataset.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# Process profiles
def process_profiles(df):
    profiles = []
    for _, row in df.iterrows():
        profile = {
            "age": row.get("age", "Unknown"),
            "gender": row.get("sex", "Unknown"),
            "location": row.get("location", "Unknown"),
            "description": row.get("essay0", "No description available.")
        }
        profiles.append(profile)
    return profiles

profiles = process_profiles(df)

# Test prompts with GPT
def match_prompt_with_profiles(prompt, profiles):
    profiles_text = "\n".join([
        f"{i+1}. {profile['description']} (Age: {profile['age']}, Gender: {profile['gender']}, Location: {profile['location']})"
        for i, profile in enumerate(profiles[:5])  # Limit to 5 profiles
    ])
    query = f"""
    Match the following description to a profile:
    "{prompt}"

    Profiles:
    {profiles_text}

    Return the best match:
    """
    response = openai.Completion.create(
        model="gpt-4",
        prompt=query,
        max_tokens=150,
        temperature=0.7
    )
    return response["choices"][0]["text"].strip()

# Test with first 5 prompts
for prompt in prompts[:5]:
    result = match_prompt_with_profiles(prompt, profiles)
    print(f"Prompt: {prompt}\nBest Match: {result}\n")
