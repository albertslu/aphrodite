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
        return min_age, max_age
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
        if profile_age:
            profile_age = int(profile_age)
            # Apply ±2 years flexibility here
            if not (min_age - 2 <= profile_age <= max_age + 2):
                return False
        else:
            return False

    # Gender check
    desired_gender = extract_gender_preference(prompt)
    if desired_gender:
        profile_gender = profile.get('sex', '').lower()
        if not profile_gender or profile_gender != desired_gender:
            return False

    # Orientation check
    desired_orientation = extract_orientation(prompt)
    if desired_orientation:
        profile_orientation = profile.get('orientation', '').lower()
        if not profile_orientation or profile_orientation != desired_orientation:
            return False

    return True

def evaluate_match(prompt, profile_info, retry_count=0):
    """
    Use hierarchical filtering and GPT to evaluate if a profile matches the prompt requirements
    """
    try:
        # Step 1: Check basic criteria first
        passes_basic = check_basic_criteria(prompt, profile_info)
        if not passes_basic:
            return False, "No. Does not pass basic criteria."

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

def evaluate_batch(prompts_and_profiles, batch_size=3):
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
            # First check basic criteria
            if not check_basic_criteria(prompt, profile):
                all_results.append((False, "No. Does not pass basic criteria."))
                continue
            
            # Extract essay content for personality traits
            essay_fields = [f"essay{i}" for i in range(10)]
            essays = []
            for field in essay_fields:
                if field in profile and profile[field]:
                    essays.append(profile[field])
            essay_content = " ".join(essays)
            
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
            
            Profile Essays (for personality traits):
            {essay_content[:1000] if essay_content else "No essays available"}
            
            This profile has already passed basic criteria checks (age/gender/orientation/ethnicity).
            Be lenient in your evaluation - this is a dating app and we want to encourage potential matches!
            
            Evaluation Guidelines:
            1. Basic Criteria (already checked):
               - Age is within range (±2 years)
               - Gender matches
               - For ethnicity, matching ANY ONE requested ethnicity is good enough
            
            2. Physical/Location Requirements:
               - If prompt specifies location/height/body type, check these
               - If not mentioned in prompt, ignore these attributes
               - If profile is missing an attribute that prompt asks for, be lenient
            
            3. Personality Traits (be very lenient):
               - For subjective traits (creative, fun-loving, etc.), assume it's fine
               - Only reject if profile EXPLICITLY contradicts the trait
               - Example: If prompt wants "creative" and profile says "I hate creative things", reject
               - But if profile doesn't mention creativity, that's fine!
            
            4. Overall Approach:
               - This is a dating app - we want to encourage potential matches!
               - If nothing in the profile contradicts the prompt, lean towards Yes
               - A match doesn't need to be perfect, just promising enough for a first date
               - Remember: Users can judge personality better in person
            
            Respond with EXACTLY "Yes" or "No" followed by a brief explanation.
            Keep the explanation very short, under 10 words.
            """
            messages.append({
                "role": "system",
                "content": "You are a dating app matching evaluator. Be lenient and encouraging - if the profile doesn't explicitly contradict the prompt's requirements, consider it a potential match. We want to encourage connections that could work, even if they're not perfect matches. Only reject if there's a clear mismatch on objective criteria or explicit contradiction of requirements."
            })
            messages.append({"role": "user", "content": evaluation_prompt})
        
        # Make batch API call
        try:
            client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))
            responses = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": msg["content"]} if msg["role"] == "system" else {"role": "user", "content": msg["content"]} for msg in messages],
                n=1,
                temperature=0.4,
            )
            
            # Process responses
            for response in responses.choices:
                answer = response.message.content.strip().lower()
                # Print response for debugging
                print(f"GPT Response: {answer}", flush=True)
                # More strict yes/no check
                first_word = answer.split()[0]
                is_match = first_word == 'yes'
                explanation = ' '.join(answer.split()[1:]) if len(answer.split()) > 1 else ''
                all_results.append((is_match, explanation))
                
        except Exception as e:
            print(f"Error in batch {batch_num + 1}: {str(e)}", flush=True)
            # Add failed results
            for _ in batch:
                all_results.append((False, f"Error: {str(e)}"))
    
    return all_results

def load_progress():
    """Load progress from file or create default progress"""
    # Always start fresh
    return {"current_index": 0, "total_matches": 0, "accurate_matches": 0, "yes_count": 0, "no_count": 0}

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
        passes_basic = check_basic_criteria(prompt, profile)
        if passes_basic:
            prompts_and_profiles.append((prompt, profile))
            batch_map[current_idx] = (result, match)
            current_idx += 1

print(f"Found {len(prompts_and_profiles)} profiles to evaluate", flush=True)

# Process in batches
batch_results = evaluate_batch(prompts_and_profiles)

# Update results with batch evaluations
results_map = {}  # Map to track results by prompt
total_evaluated = len(prompts_and_profiles)  # This is the actual number of profiles evaluated

for i, (is_match, explanation) in enumerate(batch_results):
    result, match = batch_map[i]
    match["is_accurate_match"] = is_match
    match["explanation"] = explanation
    
    if is_match:
        accurate_matches += 1
        yes_count += 1
    else:
        no_count += 1
    
    # Group matches by prompt
    prompt = result["prompt"]
    if prompt not in results_map:
        results_map[prompt] = result.copy()
        results_map[prompt]["matches"] = []
    results_map[prompt]["matches"].append(match)
    
    # Save progress every 10 matches
    if i > 0 and i % 10 == 0:
        progress_data = {
            "current_index": start_index + (i // 3),  # Approximate prompt index
            "total_evaluated": total_evaluated,
            "accurate_matches": accurate_matches,
            "yes_count": yes_count,
            "no_count": no_count
        }
        save_progress(progress_data)

# Convert map to list for final results
evaluation_results = list(results_map.values())

# Save final results with actual total evaluated
save_final_results(evaluation_results, total_evaluated, accurate_matches, yes_count, no_count)
print("\nEvaluation complete!")
print(f"Total matches evaluated: {total_evaluated}")
print(f"Accurate matches: {accurate_matches}")
if total_evaluated > 0:
    print(f"Accuracy: {(accurate_matches/total_evaluated*100):.2f}%")
else:
    print("Accuracy: N/A (no matches evaluated)")
print(f"Yes matches: {yes_count}")
print(f"No matches: {no_count}")

# Reset progress file to start fresh next time
save_progress({"current_index": -1, "last_prompt_index": -1, "last_match_index": -1})
