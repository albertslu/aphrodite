import json

def load_data():
    with open("similarity_results_500.json", "r") as f:
        return json.load(f)

def evaluate_match(prompt_data):
    prompt = prompt_data["prompt"]
    matches = prompt_data["matches"]
    
    print(f"\nEvaluating prompt: {prompt}")
    
    # Extract requirements from prompt
    requirements = {
        "age_range": None,
        "gender": None,
        "ethnicity": [],
        "other_traits": []
    }
    
    # Parse age range (format: XX-XX)
    import re
    age_match = re.search(r'(\d+)-(\d+)', prompt)
    if age_match:
        requirements["age_range"] = (int(age_match.group(1)), int(age_match.group(2)))
    
    # Parse gender
    if "male" in prompt.lower():
        requirements["gender"] = "M"
    elif "female" in prompt.lower():
        requirements["gender"] = "F"
    
    # Parse ethnicity (common formats)
    ethnicities = ["asian", "white", "black", "hispanic", "latin", "pacific islander", "indian", "middle eastern", "native american"]
    for ethnicity in ethnicities:
        if ethnicity in prompt.lower():
            requirements["ethnicity"].append(ethnicity)
    
    # Evaluate each match
    results = []
    for idx, match in enumerate(matches, 1):
        print(f"\nMatch {idx}:")
        print(f"Profile {match['profile_index']}")
        print(f"Age: {match['age']}")
        print(f"Ethnicity: {match['ethnicity']}")
        
        is_valid = True
        reasons = []
        
        # Check age (with ±2 years flexibility)
        if requirements["age_range"]:
            min_age, max_age = requirements["age_range"]
            min_age -= 2  # Add flexibility
            max_age += 2
            if not (min_age <= match['age'] <= max_age):
                is_valid = False
                reasons.append(f"Age {match['age']} outside range {min_age}-{max_age}")
        
        # Check ethnicity
        if requirements["ethnicity"]:
            profile_ethnicities = [e.strip().lower() for e in match['ethnicity'].split(',')]
            # Match is valid if profile has ANY ONE of the required ethnicities
            if not any(req_eth in profile_ethnicities for req_eth in requirements["ethnicity"]):
                is_valid = False
                reasons.append(f"Ethnicity {match['ethnicity']} doesn't match any required: {requirements['ethnicity']}")
        
        # Record result
        result = "VALID" if is_valid else "INVALID"
        if reasons:
            result += f" - {'; '.join(reasons)}"
        results.append(result)
        print(result)
    
    return results

def main():
    data = load_data()
    first_10_prompts = data[:10]
    
    total_valid = 0
    total_matches = 0
    
    print("Starting evaluation of first 10 prompts...")
    for prompt_data in first_10_prompts:
        results = evaluate_match(prompt_data)
        valid_count = sum(1 for r in results if "VALID" in r)
        total_valid += valid_count
        total_matches += len(results)
    
    accuracy = (total_valid / total_matches) * 100
    print(f"\nOverall accuracy for first 10 prompts: {accuracy:.2f}%")
    print(f"Valid matches: {total_valid}/{total_matches}")

if __name__ == "__main__":
    main()
