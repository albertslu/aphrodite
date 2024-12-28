import json

# Define the paths
input_file_path = "formatted_profiles_100_normalized.jsonl"  # Replace with your JSONL file path
output_file_path = "prompts_only.jsonl"

# Read the JSONL file and extract prompts
prompts = []
with open(input_file_path, "r") as file:
    for line in file:
        entry = json.loads(line)
        if "prompt" in entry:
            prompts.append({"prompt": entry["prompt"]})

# Save only the prompts to a new JSONL file
with open(output_file_path, "w") as output_file:
    for prompt in prompts:
        output_file.write(json.dumps(prompt) + "\n")

print(f"Prompts successfully saved to {output_file_path}")
