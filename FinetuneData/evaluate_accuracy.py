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
    min_age, max_age = extract_age_range(prompt)
    preferred_gender = extract_gender_preference(prompt)
    preferred_orientation = extract_orientation(prompt)
    
    # Age check - only if age range is specified
    if min_age and max_age:
        if not (min_age - 3 <= profile['age'] <= max_age + 3):  # Add more flexibility
            return False, "Age outside specified range"
    
    # Gender check - only if gender is specified
    if preferred_gender and profile['sex'] != preferred_gender:
        return False, "Gender does not match preference"
    
    # Orientation check - only if orientation is specified
    if preferred_orientation and profile['orientation'] != preferred_orientation:
        # Special case: if profile is bisexual, they can match with straight/gay preferences
        if profile['orientation'] == 'bisexual':
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
        - Age: {profile_info['age']}
        - Gender: {profile_info['sex']}
        - Orientation: {profile_info['orientation']}
        - Location: {profile_info['location']}
        - Education: {profile_info['education']}
        - Ethnicity: {profile_info['ethnicity']}
        - Body Type: {profile_info['body_type']}
        - Height: {profile_info['height']}
        - Religion: {profile_info['religion']}
        
        The profile has already passed basic criteria checks (age/gender/orientation).
        Now evaluate if this profile matches the remaining requirements in the prompt.
        Consider physical attributes (height, body type, ethnicity), location, education, religion, and other specific requirements.
        
        Be somewhat lenient in matching - if most key requirements are met, consider it a match even if some minor preferences don't align perfectly.
        
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

def load_progress():
    try:
        with open("evaluation_progress.json", "r") as f:
            data = json.load(f)
            # Ensure all required fields exist
            if "last_prompt_index" not in data or "last_match_index" not in data:
                return {"last_prompt_index": -1, "last_match_index": 0}
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_prompt_index": -1, "last_match_index": 0}

def save_progress(progress_data):
    with open("evaluation_progress.json", "w") as f:
        json.dump(progress_data, f)

def save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count):
    results = {
        "overall_accuracy": (accurate_matches / total_matches * 100) if total_matches > 0 else 0,
        "total_matches_evaluated": total_matches,
        "accurate_matches": accurate_matches,
        "yes_matches": yes_count,
        "no_matches": no_count,
        "evaluation_timestamp": datetime.now().isoformat(),
        "detailed_results": evaluation_results
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

# Load similarity results
print("Loading similarity results...")
with open("similarity_results.json", "r") as file:
    results = json.load(file)

# Load progress
progress = load_progress()
last_prompt_index = progress["last_prompt_index"]
last_match_index = progress["last_match_index"]

evaluation_results = []
total_matches = 0
accurate_matches = 0
yes_count = 0
no_count = 0

try:
    # Process each prompt and its matches
    for prompt_index, result in enumerate(results):
        if prompt_index <= last_prompt_index and prompt_index != len(results) - 1:
            continue
            
        prompt = result["prompt"]
        matches = result["matches"]
        prompt_results = {"prompt": prompt, "matches": []}
        
        print(f"\nEvaluating prompt {prompt_index + 1}/{len(results)}: {prompt}")
        
        # Process each match for the current prompt
        for match_index, match in enumerate(matches):
            if prompt_index == last_prompt_index and match_index < last_match_index:
                continue
                
            profile_index = match["profile_index"]
            similarity_score = match["similarity_score"]
            
            # Load the profile from extracted profiles
            with open("extracted_250_random_profiles.json", "r") as f:
                profiles = json.load(f)
                profile_info = profiles[profile_index]
            
            print(f"Evaluating match {match_index + 1}/{len(matches)} (Profile {profile_index})... ", end="")
            is_match, explanation = evaluate_match(prompt, profile_info)
            
            if is_match:
                yes_count += 1
                print("[+]")
            else:
                no_count += 1
                print("[-]")
            
            match_result = {
                "profile_index": profile_index,
                "similarity_score": similarity_score,
                "is_accurate_match": is_match,
                "explanation": explanation
            }
            prompt_results["matches"].append(match_result)
            
            total_matches += 1
            if is_match:
                accurate_matches += 1
            
            # Save progress after each match
            progress["last_prompt_index"] = prompt_index
            progress["last_match_index"] = match_index + 1
            save_progress(progress)
            
            # Optional: Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        evaluation_results.append(prompt_results)
        
    # Save final results
    save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count)
    print("\nEvaluation complete!")
    print(f"Total matches evaluated: {total_matches}")
    print(f"Accurate matches: {accurate_matches}")
    print(f"Accuracy: {(accurate_matches/total_matches*100):.2f}%")
    print(f"Yes matches: {yes_count}")
    print(f"No matches: {no_count}")

except KeyboardInterrupt:
    print("\nEvaluation interrupted. Progress has been saved.")
    save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count)
    sys.exit(0)
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    save_final_results(evaluation_results, total_matches, accurate_matches, yes_count, no_count)
    raise
