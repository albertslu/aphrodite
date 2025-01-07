import json
from datetime import datetime

def load_data():
    with open("similarity_results_500.json", "r") as f:
        return json.load(f)

def evaluate_match(prompt_data):
    prompt = prompt_data["prompt"]
    matches = prompt_data["matches"]
    
    # Extract requirements from prompt
    requirements = {
        "age_range": None,
        "gender": None,
        "ethnicity": [],
        "education": None,
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
    
    # Evaluate each match
    match_results = []
    for match in matches:
        result = {
            "profile_index": match["profile_index"],
            "age": match["age"],
            "ethnicity": match["ethnicity"],
            "education": match["education"],
            "similarity_score": match["similarity_score"],
            "is_valid": True,
            "reasons": [],
            "criteria_matched": [],
        }
        
        # Check age
        if requirements["age_range"]:
            min_age, max_age = requirements["age_range"]
            if min_age <= match['age'] <= max_age:
                result["criteria_matched"].append(f"Age {match['age']} within range {min_age}-{max_age}")
            else:
                result["is_valid"] = False
                result["reasons"].append(f"Age {match['age']} outside range {min_age}-{max_age}")
        
        # Check ethnicity
        if requirements["ethnicity"]:
            profile_ethnicities = [e.strip().lower() for e in match['ethnicity'].split(',')]
            # Match is valid if profile has ANY ONE of the required ethnicities
            matching_ethnicities = [e for e in requirements["ethnicity"] if e in profile_ethnicities]
            if matching_ethnicities:
                result["criteria_matched"].append(f"Matches ethnicity: {', '.join(matching_ethnicities)}")
            else:
                result["is_valid"] = False
                result["reasons"].append(f"Ethnicity {match['ethnicity']} doesn't match any required: {requirements['ethnicity']}")
        
        # Check education if specified
        if requirements["education"] and "education" in match:
            edu = match["education"].lower()
            if requirements["education"] == "college" and any(k in edu for k in ["college", "university", "undergraduate"]):
                result["criteria_matched"].append("Matches college education requirement")
            elif requirements["education"] == "masters" and any(k in edu for k in ["master", "graduate"]):
                result["criteria_matched"].append("Matches masters education requirement")
            elif requirements["education"] == "phd" and any(k in edu for k in ["phd", "doctorate"]):
                result["criteria_matched"].append("Matches PhD education requirement")
            elif requirements["education"] == "any_degree" and any(k in edu for k in ["degree", "college", "university", "master", "phd"]):
                result["criteria_matched"].append("Has a degree as required")
            else:
                result["is_valid"] = False
                result["reasons"].append(f"Education '{edu}' doesn't match requirement: {requirements['education']}")
        
        match_results.append(result)
    
    evaluation_result = {
        "prompt": prompt,
        "requirements": requirements,
        "matches": match_results,
        "match_count": len(match_results),
        "valid_match_count": sum(1 for m in match_results if m["is_valid"])
    }
    
    return evaluation_result

def main():
    data = load_data()
    all_results = []
    
    print(f"Starting evaluation of {len(data)} prompts...")
    for i, prompt_data in enumerate(data, 1):
        print(f"Evaluating prompt {i}/{len(data)}: {prompt_data['prompt'][:50]}...")
        result = evaluate_match(prompt_data)
        all_results.append(result)
    
    # Calculate overall statistics
    total_matches = sum(r["match_count"] for r in all_results)
    total_valid = sum(r["valid_match_count"] for r in all_results)
    accuracy = (total_valid / total_matches) * 100 if total_matches > 0 else 0
    
    # Create final output
    output = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_prompts": len(all_results),
        "total_matches": total_matches,
        "valid_matches": total_valid,
        "accuracy_percentage": accuracy,
        "results": all_results
    }
    
    # Save results
    output_file = "evaluation_results_all.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Total prompts evaluated: {len(all_results)}")
    print(f"Total matches evaluated: {total_matches}")
    print(f"Valid matches: {total_valid}/{total_matches}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
