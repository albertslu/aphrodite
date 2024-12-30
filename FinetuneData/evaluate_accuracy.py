import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))

def evaluate_match(prompt, profile_info):
    """
    Use GPT to evaluate if a profile matches the prompt requirements
    """
    evaluation_prompt = f"""
    Given this dating app prompt: "{prompt}"
    And this profile information:
    - Age: {profile_info['age']}
    - Location: {profile_info['location']}
    - Education: {profile_info['education']}
    - Ethnicity: {profile_info['ethnicity']}
    - Body Type: {profile_info['body_type']}

    Evaluate if this profile matches the requirements in the prompt.
    Consider age ranges, physical attributes, education level, and other specific requirements mentioned in the prompt.
    Respond with either 'Yes' or 'No' and a brief explanation.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a dating app matching evaluator. Your task is to determine if a profile matches the given prompt requirements. Be strict about age ranges and specific requirements mentioned in the prompt."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3
    )

    evaluation = response.choices[0].message.content
    is_match = evaluation.lower().startswith('yes')
    return is_match, evaluation

# Load similarity results
with open("similarity_results.json", "r") as file:
    results = json.load(file)

# Evaluate each match
evaluation_results = []
total_matches = 0
accurate_matches = 0

for result in results:
    prompt = result["prompt"]
    prompt_evaluation = {
        "prompt": prompt,
        "matches": []
    }
    
    for match in result["matches"]:
        is_match, explanation = evaluate_match(prompt, match)
        match_evaluation = {
            "profile_index": match["profile_index"],
            "similarity_score": match["similarity_score"],
            "is_accurate_match": is_match,
            "explanation": explanation
        }
        prompt_evaluation["matches"].append(match_evaluation)
        
        total_matches += 1
        if is_match:
            accurate_matches += 1
    
    evaluation_results.append(prompt_evaluation)

# Calculate overall accuracy
overall_accuracy = (accurate_matches / total_matches) * 100 if total_matches > 0 else 0

# Save evaluation results
output = {
    "overall_accuracy": overall_accuracy,
    "total_matches_evaluated": total_matches,
    "accurate_matches": accurate_matches,
    "detailed_results": evaluation_results
}

with open("evaluation_results.json", "w") as file:
    json.dump(output, file, indent=2)

print(f"\nEvaluation completed!")
print(f"Overall accuracy: {overall_accuracy:.2f}%")
print(f"Total matches evaluated: {total_matches}")
print(f"Accurate matches: {accurate_matches}")
print("Detailed results saved to evaluation_results.json")
