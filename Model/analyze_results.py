import json

# Load evaluation results
with open("evaluation_results.json", "r", encoding='utf-8') as file:
    results = json.load(file)

# Initialize counters
total_matches = 0
yes_matches = 0
no_matches = 0
total_similarity = 0
yes_similarity = 0
no_similarity = 0

# Analyze each prompt and its matches
for prompt_result in results["detailed_results"]:
    for match in prompt_result["matches"]:
        total_matches += 1
        total_similarity += match["similarity_score"]
        
        if match["is_accurate_match"]:
            yes_matches += 1
            yes_similarity += match["similarity_score"]
        else:
            no_matches += 1
            no_similarity += match["similarity_score"]

# Calculate averages
avg_similarity = total_similarity / total_matches if total_matches > 0 else 0
avg_yes_similarity = yes_similarity / yes_matches if yes_matches > 0 else 0
avg_no_similarity = no_similarity / no_matches if no_matches > 0 else 0

print("\nMatch Analysis Results:")
print("-----------------------")
print(f"Total matches evaluated: {total_matches}")
print(f"Yes matches: {yes_matches} ({(yes_matches/total_matches*100):.2f}%)")
print(f"No matches: {no_matches} ({(no_matches/total_matches*100):.2f}%)")
print(f"\nSimilarity Scores:")
print(f"Average overall similarity: {avg_similarity:.3f}")
print(f"Average similarity for Yes matches: {avg_yes_similarity:.3f}")
print(f"Average similarity for No matches: {avg_no_similarity:.3f}")

# Analyze some examples of matches
print("\nExample matches:")
for prompt_result in results["detailed_results"][:3]:  # Show first 3 prompts
    print(f"\nPrompt: {prompt_result['prompt']}")
    for match in prompt_result["matches"]:
        print(f"Match (Score: {match['similarity_score']:.3f}): {'YES' if match['is_accurate_match'] else 'NO'}")
        print(f"Explanation: {match.get('explanation', 'No explanation provided')}")
