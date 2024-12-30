import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import sys
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))

def evaluate_match(prompt, profile_info, retry_count=0):
    """
    Use GPT to evaluate if a profile matches the prompt requirements
    """
    try:
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
        Keep the explanation concise, maximum 2 sentences.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a dating app matching evaluator. Your task is to determine if a profile matches the given prompt requirements. Be strict about age ranges and specific requirements mentioned in the prompt."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
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
            return False, f"Error in evaluation: {str(e)}"

def load_progress():
    try:
        with open("evaluation_progress.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"completed_prompts": [], "results": []}

def save_progress(progress_data):
    with open("evaluation_progress.json", "w") as f:
        json.dump(progress_data, f, indent=2)

def save_final_results(evaluation_results, total_matches, accurate_matches):
    overall_accuracy = (accurate_matches / total_matches) * 100 if total_matches > 0 else 0
    
    output = {
        "overall_accuracy": overall_accuracy,
        "total_matches_evaluated": total_matches,
        "accurate_matches": accurate_matches,
        "evaluation_timestamp": datetime.now().isoformat(),
        "detailed_results": evaluation_results
    }

    with open("evaluation_results.json", "w") as file:
        json.dump(output, file, indent=2)

    return overall_accuracy

# Load similarity results
print("Loading similarity results...")
with open("similarity_results.json", "r") as file:
    results = json.load(file)

# Load previous progress
progress_data = load_progress()
evaluation_results = progress_data["results"]
completed_prompts = set(progress_data["completed_prompts"])

# Initialize counters
total_matches = sum(len(result["matches"]) for result in evaluation_results)
accurate_matches = sum(
    sum(1 for match in result["matches"] if match["is_accurate_match"])
    for result in evaluation_results
)

# Process remaining prompts
total_prompts = len(results)
for i, result in enumerate(results):
    prompt = result["prompt"]
    
    # Skip if already processed
    if prompt in completed_prompts:
        continue
        
    print(f"\nProcessing prompt {i+1}/{total_prompts}: {prompt[:50]}...")
    
    prompt_evaluation = {
        "prompt": prompt,
        "matches": []
    }
    
    for j, match in enumerate(result["matches"]):
        print(f"  Evaluating match {j+1}/{len(result['matches'])}...", end='\r')
        
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
            
        # Save progress after each match
        evaluation_results.append(prompt_evaluation)
        completed_prompts.add(prompt)
        save_progress({
            "completed_prompts": list(completed_prompts),
            "results": evaluation_results
        })
        
        # Add a small delay between API calls
        time.sleep(0.5)

# Save final results
overall_accuracy = save_final_results(evaluation_results, total_matches, accurate_matches)

print(f"\nEvaluation completed!")
print(f"Overall accuracy: {overall_accuracy:.2f}%")
print(f"Total matches evaluated: {total_matches}")
print(f"Accurate matches: {accurate_matches}")
print("Detailed results saved to evaluation_results.json")
