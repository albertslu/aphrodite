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

def check_height_requirement(prompt, profile):
    """
    Check if profile meets height requirements from prompt
    Men: tall >= 72 inches (6'0"), short <= 67 inches (5'7")
    Women: tall >= 68 inches (5'8"), short <= 63 inches (5'3")
    Allow ±2 inches flexibility only for the minimum height requirement for tall
    """
    height = profile.get('height', None)
    if height is None:
        return True  # Skip height check if not specified
        
    gender = profile.get('sex', '').lower()
    if not gender:  # Skip if gender not specified
        return True
        
    prompt_lower = prompt.lower()
    
    # Define height thresholds
    if gender == 'm':
        tall_threshold = 72  # 6'0"
        short_threshold = 67  # 5'7"
    else:  # 'f'
        tall_threshold = 68  # 5'8"
        short_threshold = 63  # 5'3"
    
    # Check if prompt mentions height
    if "tall" in prompt_lower:
        # Only allow 2 inches flexibility for minimum height
        return height >= (tall_threshold - 2)
    elif "short" in prompt_lower:
        # No flexibility for maximum height
        return height <= short_threshold
    
    return True  # No height requirement in prompt

def format_height(inches):
    feet = inches // 12
    remaining_inches = inches % 12
    return f"{feet}'{remaining_inches}\""

# Function to generate embeddings
def generate_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# Load profiles for matching
print("Starting similarity calculation...", flush=True)
with open("extracted_500_random_profiles.json", "r") as file:
    profiles = json.load(file)
print(f"Loaded {len(profiles)} profiles", flush=True)

# Load prompts from JSONL file
prompts = []
with jsonlines.open("formatted_profiles_cleaned.jsonl", "r") as reader:
    for obj in reader:
        prompts.append(obj["prompt"])
print(f"Loaded {len(prompts)} prompts", flush=True)

# Generate embeddings for profiles
print(f"Generating embeddings for {len(profiles)} profiles...", flush=True)
profile_embeddings = []
profile_indices = []  # Keep track of which profiles have embeddings
for i, profile in enumerate(profiles):
    if i % 50 == 0:  # Print progress every 50 profiles
        print(f"Processing profile {i}/{len(profiles)}", flush=True)
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
print(f"\nGenerating embeddings for {len(prompts)} prompts...", flush=True)
prompt_embeddings = []
for i, prompt in enumerate(prompts):
    if i % 10 == 0:  # Print progress every 10 prompts
        print(f"Processing prompt {i}/{len(prompts)}", flush=True)
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
    
    print(f"\nProcessing prompt: {prompt}", flush=True)
    print(f"Age range: {min_age}-{max_age}", flush=True)
    print(f"Desired gender: {desired_gender}", flush=True)
    print(f"Desired ethnicities: {desired_ethnicities}", flush=True)
    
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

        # Height check
        if not check_height_requirement(prompt, profile):
            continue

        valid_profiles.append(profile_idx)
        valid_embeddings.append(profile_embeddings[idx])
        valid_indices.append(idx)

    print(f"Found {len(valid_profiles)} profiles matching basic criteria", flush=True)

    # If we have valid profiles, calculate similarity
    matches_info = []
    if valid_embeddings:
        similarities = cosine_similarity([prompt_embedding], valid_embeddings)[0]
        # Get target age (middle of range) for sorting
        target_age = (min_age + max_age) / 2 if min_age and max_age else None
        
        # Create matches with all valid profiles
        valid_matches = [(idx, score) for idx, score in enumerate(similarities)]
        
        # Sort by age difference first (if age specified), then by similarity score
        if target_age:
            valid_matches.sort(key=lambda x: (
                abs(int(profiles[valid_profiles[x[0]]].get('age', 0)) - target_age),  # Smaller age diff = better
                -x[1]  # Higher similarity = better
            ))
        else:
            # If no age specified, sort by similarity only
            valid_matches.sort(key=lambda x: -x[1])
        
        # Take top 3 matches
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
                "sex": profile.get("sex", "N/A"),  # Added gender to output
            }
            
            # Add height information if prompt mentions height
            if "tall" in prompt.lower() or "short" in prompt.lower():
                height = profile.get("height")
                if height is not None:
                    match_info["height"] = format_height(height)
                else:
                    match_info["height"] = "Not specified"
            
            matches_info.append(match_info)

    # Only use backup matches if we have less than 3 valid matches
    if len(matches_info) < 3:
        print(f"Only found {len(matches_info)} matches, looking for backup matches...", flush=True)
        
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
                "sex": profile.get("sex", "N/A"),  # Added gender to output
            }
            
            # Add height information if prompt mentions height
            if "tall" in prompt.lower() or "short" in prompt.lower():
                height = profile.get("height")
                if height is not None:
                    match_info["height"] = format_height(height)
                else:
                    match_info["height"] = "Not specified"
            
            matches_info.append(match_info)
            valid_profiles.append(profile_idx)
        else:
            print("No backup matches found!", flush=True)
            break

    print(f"Final number of matches: {len(matches_info)}\n", flush=True)

    results.append({"prompt": prompts[i], "matches": matches_info})

# Save results with pretty printing
with open("similarity_results_500.json", "w") as file:
    json.dump(results, file, indent=2)

print("Similarity calculations saved to similarity_results_500.json", flush=True)
