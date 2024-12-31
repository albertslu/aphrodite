import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import jsonlines
import re

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))

def extract_age_range(prompt):
    # Extract age range from prompt using regex
    age_match = re.search(r'aged? (\d+)-(\d+)|(\d+)-(\d+) \w+', prompt)
    if age_match:
        groups = age_match.groups()
        if groups[0] and groups[1]:  # "aged X-Y" format
            return int(groups[0]), int(groups[1])
        elif groups[2] and groups[3]:  # "X-Y year" format
            return int(groups[2]), int(groups[3])
    return None, None

def extract_gender(prompt):
    prompt = prompt.lower()
    # Check for female first to avoid matching 'male' in 'female'
    if 'female' in prompt:
        return 'F'
    elif 'male' in prompt:
        return 'M'
    return None

def extract_ethnicity(prompt):
    ethnicities = ['white', 'black', 'asian', 'hispanic', 'latin', 'pacific islander', 'indian', 'middle eastern', 'native american']
    found = []
    for ethnicity in ethnicities:
        if ethnicity in prompt.lower():
            found.append(ethnicity)
    return found if found else None

# Function to check if profile ethnicity matches any desired ethnicity
def matches_ethnicity(profile_ethnicity, desired_ethnicities):
    if not desired_ethnicities:
        return True
    if not profile_ethnicity:
        return False
    profile_ethnicity = profile_ethnicity.lower()
    # Return True if ANY of the desired ethnicities match
    return any(eth in profile_ethnicity for eth in desired_ethnicities)

def matches_gender(profile_gender, desired_gender):
    if not desired_gender:
        return True
    if not profile_gender:
        return False
    # Convert gender to standard format (M/F)
    profile_gender = profile_gender.upper().strip()
    if profile_gender in ['M', 'MALE']:
        profile_gender = 'M'
    elif profile_gender in ['F', 'FEMALE']:
        profile_gender = 'F'
    return profile_gender == desired_gender

# Function to generate embeddings
def generate_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# Load profiles for matching
with open("extracted_250_random_profiles.json", "r") as file:
    profiles = json.load(file)

# Load prompts from JSONL file
prompts = []
with jsonlines.open("formatted_profiles_cleaned.jsonl", "r") as reader:
    for obj in reader:
        prompts.append(obj["prompt"])

# Generate embeddings for profiles
profile_embeddings = []
profile_indices = []  # Keep track of which profiles have embeddings
for i, profile in enumerate(profiles):
    # Combine all essay fields
    essay_fields = [f"essay{i}" for i in range(10)]
    profile_texts = []
    for field in essay_fields:
        if field in profile and profile[field]:
            profile_texts.append(profile[field])

    profile_text = " ".join(profile_texts)
    if profile_text.strip():  # Only process if there's text
        embedding = generate_embedding(profile_text)
        profile_embeddings.append(embedding)
        profile_indices.append(i)

# Generate embeddings for prompts
prompt_embeddings = []
for prompt in prompts:
    embedding = generate_embedding(prompt)
    prompt_embeddings.append(embedding)

# Calculate matches based on criteria and similarity
results = []
for i, prompt_embedding in enumerate(prompt_embeddings):
    prompt = prompts[i]
    min_age, max_age = extract_age_range(prompt)
    desired_gender = extract_gender(prompt)
    desired_ethnicities = extract_ethnicity(prompt)

    # First filter by biometric/physical criteria
    valid_profiles = []
    valid_embeddings = []
    valid_indices = []
    
    print(f"\nProcessing prompt: {prompt}")
    print(f"Age range: {min_age}-{max_age}")
    print(f"Desired gender: {desired_gender}")
    print(f"Desired ethnicities: {desired_ethnicities}")
    
    for idx, profile_idx in enumerate(profile_indices):
        profile = profiles[profile_idx]
        
        # Age check with ±2 years flexibility
        if min_age and max_age:
            profile_age = profile.get('age')
            if not profile_age:
                continue
            profile_age = int(profile_age)
            if not (min_age - 2 <= profile_age <= max_age + 2):
                continue

        # Gender check using new matching function
        if desired_gender:
            profile_gender = profile.get('sex')
            if not matches_gender(profile_gender, desired_gender):
                continue

        # Ethnicity check - now using the helper function
        if desired_ethnicities:
            profile_ethnicity = profile.get('ethnicity', '')
            if not matches_ethnicity(profile_ethnicity, desired_ethnicities):
                continue

        valid_profiles.append(profile_idx)
        valid_embeddings.append(profile_embeddings[idx])
        valid_indices.append(idx)

    print(f"Found {len(valid_profiles)} profiles matching basic criteria")

    # If we have valid profiles, calculate similarity
    matches_info = []
    if valid_embeddings:
        similarities = cosine_similarity([prompt_embedding], valid_embeddings)[0]
        # Lower threshold for initial matches
        threshold = 0.3  # Lowered from 0.5
        valid_matches = [(idx, score) for idx, score in enumerate(similarities) if score >= threshold]
        valid_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = valid_matches[:3]

        for match_idx, similarity_score in top_matches:
            profile_idx = valid_profiles[match_idx]
            profile = profiles[profile_idx]
            match_info = {
                "profile_index": int(profile_idx),
                "similarity_score": float(similarity_score),
                "age": int(profile.get("age", -1)),
                "location": profile.get("location", "N/A"),
                "education": profile.get("education", "N/A"),
                "ethnicity": profile.get("ethnicity", "N/A"),
                "body_type": profile.get("body_type", "N/A"),
            }
            matches_info.append(match_info)

    # If we don't have enough matches, fill with most similar profiles that at least match gender
    if len(matches_info) < 3:
        print(f"Only found {len(matches_info)} matches, looking for backup matches...")
        
    while len(matches_info) < 3:
        # Add profiles that at least match gender if specified
        backup_profiles = []
        backup_embeddings = []
        for idx, profile_idx in enumerate(profile_indices):
            if profile_idx in valid_profiles:
                continue
            profile = profiles[profile_idx]
            if not desired_gender or (profile.get('sex', '').lower() == desired_gender.lower()):
                backup_profiles.append(profile_idx)
                backup_embeddings.append(profile_embeddings[idx])

        if backup_embeddings:
            backup_similarities = cosine_similarity([prompt_embedding], backup_embeddings)[0]
            top_backup = np.argsort(backup_similarities)[-1]
            profile_idx = backup_profiles[top_backup]
            profile = profiles[profile_idx]
            match_info = {
                "profile_index": int(profile_idx),
                "similarity_score": float(backup_similarities[top_backup]),
                "age": int(profile.get("age", -1)),
                "location": profile.get("location", "N/A"),
                "education": profile.get("education", "N/A"),
                "ethnicity": profile.get("ethnicity", "N/A"),
                "body_type": profile.get("body_type", "N/A"),
            }
            matches_info.append(match_info)
            valid_profiles.append(profile_idx)
        else:
            print("No backup matches found!")
            break

    print(f"Final number of matches: {len(matches_info)}\n")

    results.append({"prompt": prompts[i], "matches": matches_info})

# Save results with pretty printing
with open("similarity_results.json", "w") as file:
    json.dump(results, file, indent=2)

print("Similarity calculations saved to similarity_results.json")
