import json
import re
from datetime import datetime

def load_data():
    with open("similarity_results_1000.json", "r") as f:
        return json.load(f)

def extract_requirements(prompt):
    """Extract all requirements from a prompt"""
    requirements = {
        "age_range": None,
        "gender": None,
        "ethnicity": [],
        "education": None,
        "height": None,
        "location": None,
        "interests": [],
        "personality": []
    }
    
    # Parse age range (format: XX-XX)
    age_match = re.search(r'aged? (\d+)-(\d+)', prompt.lower())
    if age_match:
        requirements["age_range"] = (int(age_match.group(1)), int(age_match.group(2)))
    
    # Parse gender
    if "female" in prompt.lower():
        requirements["gender"] = "F"
    elif "male" in prompt.lower():
        requirements["gender"] = "M"
    
    # Parse ethnicity
    ethnicities = ["asian", "white", "black", "hispanic", "latin", "pacific islander", 
                  "indian", "middle eastern", "native american"]
    for ethnicity in ethnicities:
        if ethnicity in prompt.lower():
            requirements["ethnicity"].append(ethnicity)
    
    # Parse education requirements
    edu_keywords = {
        "college": ["college", "university", "undergraduate"],
        "masters": ["masters", "graduate school", "grad school"],
        "phd": ["phd", "doctorate"],
        "any_degree": ["degree", "educated"]
    }
    
    for edu_level, keywords in edu_keywords.items():
        if any(keyword in prompt.lower() for keyword in keywords):
            requirements["education"] = edu_level
            break
    
    # Parse height preferences
    if "tall" in prompt.lower():
        requirements["height"] = "tall"
    elif "short" in prompt.lower():
        requirements["height"] = "short"
    
    # Parse location
    locations = ["bay area", "san francisco", "silicon valley", "east bay", 
                "oakland", "berkeley", "south bay"]
    for loc in locations:
        if loc in prompt.lower():
            requirements["location"] = loc
            break
    
    # Parse interests
    interests = ["traveling", "hiking", "reading", "cooking", "photography",
                "art", "music", "dancing", "fitness", "yoga", "meditation",
                "gaming", "sports", "movies", "theatre", "concerts", "writing", 
                "technology"]
    for interest in interests:
        if interest in prompt.lower():
            requirements["interests"].append(interest)
    
    # Parse personality traits
    traits = ["adventurous", "intellectual", "creative", "ambitious", "easy-going",
             "passionate", "down-to-earth", "fun-loving", "caring", "honest",
             "spontaneous", "thoughtful", "outgoing", "introverted", "confident",
             "humble", "goal-oriented", "free-spirited", "romantic", "practical"]
    for trait in traits:
        if trait in prompt.lower():
            requirements["personality"].append(trait)
    
    return requirements

def check_height_requirement(prompt, profile):
    """
    Check if profile meets height requirements from prompt
    Men: tall >= 72 inches (6'0"), short <= 67 inches (5'7")
    Women: tall >= 68 inches (5'8"), short <= 63 inches (5'3")
    Allow ±2 inches flexibility only for the minimum height requirement for tall
    """
    height = profile.get('height')
    if not height:
        return True, None  # Skip height check if not specified
        
    gender = profile.get('gender', '').lower()
    if not gender:  # Skip if gender not specified
        return True, None
        
    # Convert height string to inches if it's in format "X'Y""
    if isinstance(height, str) and "'" in height:
        feet, inches = height.replace('"', '').split("'")
        height = int(feet) * 12 + int(inches)
    
    # Define height thresholds
    if gender == 'm':
        tall_threshold = 72  # 6'0"
        short_threshold = 67  # 5'7"
    else:  # 'f'
        tall_threshold = 68  # 5'8"
        short_threshold = 63  # 5'3"
    
    height_desc = f"{height//12}'{height%12}\""
    
    # Check if prompt mentions height
    if "tall" in prompt.lower():
        if height >= (tall_threshold - 2):
            return True, f"Height {height_desc} meets tall requirement for {gender}"
        else:
            return False, f"Height {height_desc} does not meet tall requirement for {gender} (min: {(tall_threshold-2)//12}'{(tall_threshold-2)%12}\")"
    elif "short" in prompt.lower():
        if height <= short_threshold:
            return True, f"Height {height_desc} meets short requirement for {gender}"
        else:
            return False, f"Height {height_desc} does not meet short requirement for {gender} (max: {short_threshold//12}'{short_threshold%12}\")"
    
    return True, None

def evaluate_match(prompt, match):
    """Evaluate a single match against prompt requirements"""
    requirements = extract_requirements(prompt)
    result = {
        "is_valid": True,
        "matches": [],
        "mismatches": []
    }
    
    # Check height requirements
    height_valid, height_message = check_height_requirement(prompt, match)
    if not height_valid:
        result["is_valid"] = False
        result["mismatches"].append(height_message)
    elif height_message:
        result["matches"].append(height_message)
    
    # Check age
    if requirements["age_range"]:
        min_age, max_age = requirements["age_range"]
        if min_age <= match['age'] <= max_age:
            result["matches"].append(f"Age {match['age']} within range {min_age}-{max_age}")
        else:
            result["is_valid"] = False
            result["mismatches"].append(f"Age {match['age']} outside range {min_age}-{max_age}")
    
    # Check gender
    if requirements["gender"]:
        profile_gender = match['gender'].upper() if match['gender'] else None
        if profile_gender == requirements["gender"]:
            result["matches"].append(f"Gender matches: {profile_gender}")
        else:
            result["is_valid"] = False
            result["mismatches"].append(f"Gender mismatch: wanted {requirements['gender']}, got {profile_gender}")
    
    # Check ethnicity
    if requirements["ethnicity"]:
        if match['ethnicity']:
            profile_ethnicities = [e.strip().lower() for e in match['ethnicity'].split(',')]
            matching_ethnicities = [e for e in requirements["ethnicity"] if e in profile_ethnicities]
            if matching_ethnicities:
                result["matches"].append(f"Matches ethnicity: {', '.join(matching_ethnicities)}")
            else:
                result["is_valid"] = False
                result["mismatches"].append(f"Ethnicity mismatch: wanted {requirements['ethnicity']}, got {profile_ethnicities}")
    
    # Check education
    if requirements["education"] and match['education']:
        edu = match['education'].lower()
        if requirements["education"] == "college":
            # Consider any higher education as meeting "college-educated" requirement
            if any(k in edu for k in ["college", "university", "undergraduate", "master", "graduate", "phd", "doctorate"]):
                result["matches"].append("Matches college education requirement")
            else:
                result["is_valid"] = False
                result["mismatches"].append(f"Education mismatch: wanted college education, got {edu}")
        elif requirements["education"] == "masters" and any(k in edu for k in ["master", "graduate"]):
            result["matches"].append("Matches masters education requirement")
        elif requirements["education"] == "phd" and any(k in edu for k in ["phd", "doctorate"]):
            result["matches"].append("Matches PhD education requirement")
        elif requirements["education"] == "any_degree" and any(k in edu for k in ["degree", "college", "university", "master", "phd"]):
            result["matches"].append("Has a degree as required")
        else:
            result["is_valid"] = False
            result["mismatches"].append(f"Education mismatch: wanted {requirements['education']}, got {edu}")
    
    # Check location if specified
    if requirements["location"] and match['location']:
        if requirements["location"].lower() in match['location'].lower():
            result["matches"].append(f"Location matches: {requirements['location']}")
        else:
            # Make location a soft requirement - don't invalidate match
            result["mismatches"].append(f"Location preference not met: wanted {requirements['location']}, got {match['location']}")
    
    return result

def main():
    results = load_data()
    
    total_prompts = len(results)
    total_matches = 0
    valid_matches = 0
    match_details = []
    
    print(f"Evaluating {total_prompts} prompts...")
    
    for prompt_result in results:
        prompt = prompt_result["prompt"]
        matches = prompt_result["matches"]
        
        prompt_matches = []
        for match in matches:
            total_matches += 1
            evaluation = evaluate_match(prompt, match)
            if evaluation["is_valid"]:
                valid_matches += 1
            
            prompt_matches.append({
                "profile_index": match["profile_index"],
                "similarity_score": match["similarity_score"],
                "evaluation": evaluation
            })
        
        match_details.append({
            "prompt": prompt,
            "matches": prompt_matches
        })
    
    # Calculate statistics
    accuracy = (valid_matches / total_matches) * 100 if total_matches > 0 else 0
    avg_matches_per_prompt = total_matches / total_prompts if total_prompts > 0 else 0
    
    # Prepare evaluation results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_prompts": total_prompts,
            "total_matches": total_matches,
            "valid_matches": valid_matches,
            "accuracy_percentage": accuracy,
            "avg_matches_per_prompt": avg_matches_per_prompt
        },
        "match_details": match_details
    }
    
    # Save detailed results
    with open("evaluation_results_1000.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Prompts: {total_prompts}")
    print(f"Total Matches: {total_matches}")
    print(f"Valid Matches: {valid_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Matches per Prompt: {avg_matches_per_prompt:.2f}")
    print("\nDetailed results saved to evaluation_results_1000.json")

if __name__ == "__main__":
    main()
