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

def matches_ethnicity(profile_ethnicity, desired_ethnicities):
    if not desired_ethnicities:
        return True
    if not profile_ethnicity:
        return False
    profile_ethnicity = profile_ethnicity.lower()
    return any(eth in profile_ethnicity for eth in desired_ethnicities)

def matches_gender(profile_gender, desired_gender):
    if not desired_gender:
        return True
    if not profile_gender:
        return False
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
        return height >= (tall_threshold - 2)
    elif "short" in prompt_lower:
        return height <= short_threshold
    
    return True

def check_education_requirement(prompt, profile):
    """Check if profile meets education requirements from prompt"""
    if not profile.get('education'):
        return True  # Skip check if education not specified
    
    prompt_lower = prompt.lower()
    education_lower = profile['education'].lower()
    
    # College requirement
    if any(term in prompt_lower for term in ['college', 'university', 'degree']):
        return any(term in education_lower for term in ['college', 'university', 'degree', 'masters', 'phd'])
    
    # Graduate degree requirement
    if any(term in prompt_lower for term in ['graduate degree', 'masters']):
        return any(term in education_lower for term in ['masters', 'graduate'])
    
    # PhD requirement
    if 'phd' in prompt_lower:
        return 'phd' in education_lower
    
    return True

def format_height(inches):
    feet = inches // 12
    remaining_inches = inches % 12
    return f"{feet}'{remaining_inches}\""

def generate_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# Load profiles for matching
print("Starting similarity calculation...", flush=True)
with open("extracted_1000_random_profiles.json", "r") as file:
    profiles = json.load(file)
print(f"Loaded {len(profiles)} profiles", flush=True)

# Load prompts from the new file
prompts = []
with open("generated_200_prompts.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        prompts.append(data["prompt"])
print(f"Loaded {len(prompts)} prompts", flush=True)

# Generate embeddings for all profiles
print("Generating profile embeddings...", flush=True)
profile_embeddings = []
for profile in profiles:
    # Combine relevant profile information for embedding
    profile_text = f"{profile.get('essay0', '')} {profile.get('essay1', '')}"
    if profile_text.strip():
        embedding = generate_embedding(profile_text)
        profile_embeddings.append(embedding)
    else:
        profile_embeddings.append(None)

# Generate embeddings for all prompts
print("Generating prompt embeddings...", flush=True)
prompt_embeddings = [generate_embedding(prompt) for prompt in prompts]

# Store results
results = []

# Process each prompt
for i, prompt_embedding in enumerate(prompt_embeddings):
    prompt = prompts[i]
    min_age, max_age = extract_age_range(prompt)
    desired_gender = extract_gender(prompt)
    desired_ethnicities = extract_ethnicity(prompt)
    
    print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:100]}...", flush=True)
    
    # Calculate similarities and filter profiles
    similarities = []
    profile_indices = []
    
    for j, (profile, profile_embedding) in enumerate(zip(profiles, profile_embeddings)):
        if profile_embedding is None:
            continue
            
        # Check basic criteria
        age = profile.get('age')
        if age is None or (min_age and (age < min_age or age > max_age)):
            continue
            
        gender = profile.get('sex')
        if not matches_gender(gender, desired_gender):
            continue
            
        ethnicity = profile.get('ethnicity')
        if not matches_ethnicity(ethnicity, desired_ethnicities):
            continue
            
        if not check_height_requirement(prompt, profile):
            continue
            
        if not check_education_requirement(prompt, profile):
            continue
        
        # Calculate similarity
        similarity = cosine_similarity([prompt_embedding], [profile_embedding])[0][0]
        similarities.append(similarity)
        profile_indices.append(j)
    
    # Get top 3 matches
    matches_info = []
    if similarities:
        top_indices = np.argsort(similarities)[-3:][::-1]
        for idx in top_indices:
            profile_idx = profile_indices[idx]
            profile = profiles[profile_idx]
            
            match_info = {
                "profile_index": profile_idx,
                "similarity_score": float(similarities[idx]),
                "age": profile.get('age'),
                "gender": profile.get('sex'),
                "ethnicity": profile.get('ethnicity', ''),
                "education": profile.get('education', ''),
                "location": profile.get('location', ''),
                "height": format_height(profile['height']) if 'height' in profile else None,
                "body_type": profile.get('body_type', ''),
                "essay0": profile.get('essay0', '')[:500] + "..." if len(profile.get('essay0', '')) > 500 else profile.get('essay0', ''),
                "essay1": profile.get('essay1', '')[:500] + "..." if len(profile.get('essay1', '')) > 500 else profile.get('essay1', '')
            }
            matches_info.append(match_info)
    
    # Add backup matches if needed
    while len(matches_info) < 3:
        # Add profiles that at least match gender if specified
        backup_profiles = []
        backup_embeddings = []
        for idx, profile_idx in enumerate(profile_indices):
            profile = profiles[profile_idx]
            if matches_gender(profile.get('sex'), desired_gender):
                backup_profiles.append(profile)
                backup_embeddings.append(profile_embeddings[profile_idx])
        
        if not backup_profiles:
            break
        
        # Get most similar backup profile
        backup_similarities = cosine_similarity([prompt_embedding], backup_embeddings)[0]
        best_backup_idx = np.argmax(backup_similarities)
        backup_profile = backup_profiles[best_backup_idx]
        
        match_info = {
            "profile_index": profiles.index(backup_profile),
            "similarity_score": float(backup_similarities[best_backup_idx]),
            "age": backup_profile.get('age'),
            "gender": backup_profile.get('sex'),
            "ethnicity": backup_profile.get('ethnicity', ''),
            "education": backup_profile.get('education', ''),
            "location": backup_profile.get('location', ''),
            "height": format_height(backup_profile['height']) if 'height' in backup_profile else None,
            "body_type": backup_profile.get('body_type', ''),
            "essay0": backup_profile.get('essay0', '')[:500] + "..." if len(backup_profile.get('essay0', '')) > 500 else backup_profile.get('essay0', ''),
            "essay1": backup_profile.get('essay1', '')[:500] + "..." if len(backup_profile.get('essay1', '')) > 500 else backup_profile.get('essay1', '')
        }
        matches_info.append(match_info)
        
        # Remove used backup profile
        profile_indices = [idx for idx in profile_indices if idx != match_info["profile_index"]]
    
    # Store results for this prompt
    results.append({
        "prompt": prompt,
        "matches": matches_info
    })

# Save results with pretty printing
with open("similarity_results_1000.json", "w") as file:
    json.dump(results, file, indent=2)

print("Similarity calculations saved to similarity_results_1000.json", flush=True)
