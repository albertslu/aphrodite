import json
import os
import time
from openai import OpenAI
import sys
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))

def extract_age_range(prompt):
    """Extract age range from prompt"""
    age_pattern = r'(\d+)[-\s]*(\d+)'
    matches = re.findall(age_pattern, prompt)
    if matches:
        min_age, max_age = map(int, matches[0])
        # Add a small buffer to age range
        return max(18, min_age - 2), min(100, max_age + 2)
    return None, None

def extract_gender_preference(prompt):
    """Extract gender preference from prompt"""
    prompt_lower = prompt.lower()
    if 'male' in prompt_lower or ' m ' in prompt_lower or 'man' in prompt_lower or 'men' in prompt_lower:
        return 'm'
    elif 'female' in prompt_lower or ' f ' in prompt_lower or 'woman' in prompt_lower or 'women' in prompt_lower:
        return 'f'
    return None

def extract_orientation(prompt):
    """Extract orientation preference from prompt"""
    prompt_lower = prompt.lower()
    if 'straight' in prompt_lower or 'heterosexual' in prompt_lower:
        return 'straight'
    elif 'gay' in prompt_lower or 'homosexual' in prompt_lower:
        return 'gay'
    elif 'bisexual' in prompt_lower or 'bi' in prompt_lower:
        return 'bisexual'
    return None

def check_basic_criteria(prompt, profile):
    """Check basic non-negotiable criteria (gender, orientation, age)"""
    # Age check
    min_age, max_age = extract_age_range(prompt)
    if min_age is not None and max_age is not None:
        profile_age = profile.get('age')
        if profile_age is None:
            return False, "Age information missing"
        # Add flexibility to age range (+/- 2 years)
        if not (min_age - 2 <= profile_age <= max_age + 2):
            return False, "Age outside preferred range"
    
    # Gender check
    preferred_gender = extract_gender_preference(prompt)
    if preferred_gender:
        profile_gender = profile.get('sex')
        if profile_gender is None:
            return False, "Gender information missing"
        if profile_gender.lower() != preferred_gender.lower():
            return False, "Gender does not match preference"
    
    # Orientation check - only if orientation is specified
    preferred_orientation = extract_orientation(prompt)
    if preferred_orientation:
        profile_orientation = profile.get('orientation')
        if profile_orientation is None:
            return False, "Orientation information missing"
        if profile_orientation != preferred_orientation:
            # Special case: if profile is bisexual, they can match with straight/gay preferences
            if profile_orientation == 'bisexual':
                return True, "Passed basic criteria (bisexual matches all preferences)"
            return False, "Orientation does not match preference"
    
    return True, "Passed basic criteria"

def evaluate_match(prompt, profile_info, retry_count=0):
    """
    Use hierarchical filtering and GPT to evaluate if a profile matches the prompt requirements
    """
    try:
        # Step 1: Check basic criteria first
        passes_basic, reason = check_basic_criteria(prompt, profile_info)
        if not passes_basic:
            return False, f"No. {reason}."

        # Step 2: Prepare detailed evaluation for remaining criteria
        evaluation_prompt = f"""
        Given this dating app prompt: "{prompt}"
        And this profile information:
        - Age: {profile_info.get('age', 'N/A')}
        - Gender: {profile_info.get('sex', 'N/A')}
        - Orientation: {profile_info.get('orientation', 'N/A')}
        - Location: {profile_info.get('location', 'N/A')}
        - Education: {profile_info.get('education', 'N/A')}
        - Ethnicity: {profile_info.get('ethnicity', 'N/A')}
        - Body Type: {profile_info.get('body_type', 'N/A')}
        - Height: {profile_info.get('height', 'N/A')}
        - Religion: {profile_info.get('religion', 'N/A')}
        
        The profile has already passed basic criteria checks (age/gender/orientation).
        Now evaluate if this profile matches the remaining requirements in the prompt.
        Consider physical attributes (height, body type, ethnicity), location, education, religion, and other specific requirements.
        
        Be somewhat lenient in matching - if most key requirements are met, consider it a match even if some minor preferences don't align perfectly.
        If a field is 'N/A' and that field is not specifically mentioned in the prompt, ignore it in the evaluation.
        
        Follow this priority order:
        1. Physical/biometric criteria (height, ethnicity, body type)
        2. Location and education requirements
        3. Other preferences (religion, personality traits, interests)
        
        Respond with either 'Yes' or 'No' and a brief explanation.
        Keep the explanation concise, maximum 2 sentences.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a dating app matching evaluator. Your task is to determine if a profile matches the given prompt requirements. Be somewhat lenient with matches - if most key requirements are met, consider it a match even if some minor preferences don't align perfectly."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.4,  # Slightly increased for more varied responses
            timeout=10.0  # 10 second timeout
        )

        evaluation = response.choices[0].message.content
        is_match = evaluation.lower().startswith('yes')
        return is_match, evaluation

    except Exception as e:
        if retry_count < 2:  # Retry up to 2 times
            print(f"\nRetrying evaluation due to error: {str(e)}")
            time.sleep(2)  # Wait 2 seconds before retrying
            return evaluate_match(prompt, profile_info, retry_count + 1)
        else:
            print(f"\nFailed to evaluate after retries: {str(e)}")
            return False, f"Error: {str(e)}"

def evaluate_batch(prompts_and_profiles, batch_size=5):
    """
    Evaluate multiple profiles in parallel using batched API calls
    """
    print(f"Starting batch evaluation of {len(prompts_and_profiles)} profiles in batches of {batch_size}...", flush=True)
    batches = [prompts_and_profiles[i:i + batch_size] for i in range(0, len(prompts_and_profiles), batch_size)]
    all_results = []
    
    for batch_num, batch in enumerate(batches):
        print(f"Processing batch {batch_num + 1}/{len(batches)}...", flush=True)
        # Prepare all prompts for the batch
        messages = []
        for prompt, profile in batch:
            evaluation_prompt = f"""
            Given this dating app prompt: "{prompt}"
            And this profile information:
            - Age: {profile.get('age', 'N/A')}
            - Gender: {profile.get('sex', 'N/A')}
            - Orientation: {profile.get('orientation', 'N/A')}
            - Location: {profile.get('location', 'N/A')}
            - Education: {profile.get('education', 'N/A')}
            - Ethnicity: {profile.get('ethnicity', 'N/A')}
            - Body Type: {profile.get('body_type', 'N/A')}
            - Height: {profile.get('height', 'N/A')}
            - Religion: {profile.get('religion', 'N/A')}
            
            Consider physical attributes (height, body type, ethnicity), location, education, religion, and other specific requirements.
            
            Important ethnicity matching rules:
            - If a profile has multiple ethnicities (e.g. "asian, white"), treat this as matching ANY of those ethnicities
            - Only require ALL ethnicities to match if the prompt specifically asks for a combination (e.g. "asian AND white")
            - Default to OR logic for ethnicity matching
            
            Respond with either 'Yes' or 'No' and a brief explanation.
            Keep the explanation concise, maximum 2 sentences.
            """
            messages.append({"role": "user", "content": evaluation_prompt})
        
        # Make a single API call for the batch
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a dating app matching evaluator. Evaluate multiple profiles in sequence. For each profile, respond with Yes/No and a brief explanation."},
                    *messages
                ]
            )
            
            # Process responses
            for i, choice in enumerate(response.choices):
                result = choice.message.content.strip()
                is_match = result.lower().startswith('yes')
                explanation = result.split('.')[0]  # Get first sentence
                all_results.append((is_match, explanation))
                
        except Exception as e:
            # Handle failed batch by processing individually
            print(f"Batch failed, processing individually: {str(e)}")
            for prompt, profile in batch:
                result = evaluate_match(prompt, profile)
                all_results.append(result)
    
    return all_results

def load_progress():
    """Load progress from file or create default progress"""
    try:
        with open("evaluation_progress.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create default progress
        default_progress = {"last_prompt_index": -1, "last_match_index": -1}
        save_progress(default_progress)
        return default_progress

def save_progress(progress_data):
    with open("evaluation_progress.json", "w") as f:
        json.dump(progress_data, f)

def save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count):
    """Save final evaluation results to JSON file"""
    results = {
        "overall_accuracy": (accurate_matches/total_matches*100) if total_matches > 0 else 0,
        "total_matches_evaluated": total_matches,
        "accurate_matches": accurate_matches,
        "yes_matches": yes_count,
        "no_matches": no_count,
        "evaluation_timestamp": datetime.now().isoformat(),
        "detailed_results": evaluation_results
    }
    
    with open("evaluation_results_500.json", "w") as f:
        json.dump(results, f, indent=2)

# Load similarity results
print("Loading similarity results...", flush=True)
with open("similarity_results_500.json", "r") as file:
    results = json.load(file)
print(f"Loaded results for {len(results)} prompts", flush=True)

# Main evaluation loop
print("Starting evaluation of matches...")
evaluation_results = []
total_matches = 0
accurate_matches = 0
yes_count = 0
no_count = 0

# Load progress
progress_data = load_progress()
start_index = progress_data.get("current_index", 0)
print(f"Starting from index {start_index}", flush=True)

# Prepare batches of prompts and profiles
print("Preparing profiles for evaluation...", flush=True)
prompts_and_profiles = []
batch_map = {}  # Map to track which profiles belong to which prompts
current_idx = 0

with open("extracted_500_random_profiles.json", "r") as f:
    profiles = json.load(f)
print(f"Loaded {len(profiles)} profiles", flush=True)

for result in results[start_index:]:
    prompt = result["prompt"]
    for match in result["matches"]:
        profile = profiles[match["profile_index"]]
        # Only add to batch if passes basic criteria
        passes_basic, _ = check_basic_criteria(prompt, profile)
        if passes_basic:
            prompts_and_profiles.append((prompt, profile))
            batch_map[current_idx] = (result, match)
            current_idx += 1

print(f"Found {len(prompts_and_profiles)} profiles to evaluate", flush=True)

# Process in batches
batch_results = evaluate_batch(prompts_and_profiles)

# Update results with batch evaluations
for i, (is_match, explanation) in enumerate(batch_results):
    result, match = batch_map[i]
    match["is_accurate_match"] = is_match
    match["explanation"] = explanation
    
    if is_match:
        accurate_matches += 1
        yes_count += 1
    else:
        no_count += 1
        
    total_matches += 1
    
    # Save progress every 10 matches
    if total_matches % 10 == 0:
        progress_data = {
            "current_index": start_index + (i // 3),  # Approximate prompt index
            "total_matches": total_matches,
            "accurate_matches": accurate_matches,
            "yes_count": yes_count,
            "no_count": no_count
        }
        save_progress(progress_data)

    if i % 3 == 2 or i == len(batch_results) - 1:  # Every 3rd result or last result
        evaluation_results.append(result)

# Save final results
save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count)
print("\nEvaluation complete!")
print(f"Total matches evaluated: {total_matches}")
print(f"Accurate matches: {accurate_matches}")
if total_matches > 0:
    print(f"Accuracy: {(accurate_matches/total_matches*100):.2f}%")
else:
    print("Accuracy: N/A (no matches evaluated)")
print(f"Yes matches: {yes_count}")
print(f"No matches: {no_count}")

# Reset progress file to start fresh next time
save_progress({"current_index": -1, "last_prompt_index": -1, "last_match_index": -1})
