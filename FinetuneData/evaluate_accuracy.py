import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load validation data
validation_data = [
    {"prompt": "Looking for someone adventurous and kind-hearted, aged 25-30.", "expected_matches": [0, 2]},
    {"prompt": "Seeking a tall, athletic partner who loves hiking.", "expected_matches": [1, 3]},
    # Add more validation cases here
]

# Load similarity results
with open("similarity_results.json", "r") as file:
    results = json.load(file)

# Evaluate accuracy
for val in validation_data:
    prompt = val["prompt"]
    expected_matches = val["expected_matches"]

    # Find the corresponding result for this prompt
    result = next((res for res in results if res["prompt"] == prompt), None)
    if result is None:
        print(f"No result found for prompt: {prompt}")
        continue

    top_matches = result["top_matches"]
    y_true = [1 if i in expected_matches else 0 for i in range(len(expected_matches))]
    y_pred = [1 if i in top_matches else 0 for i in range(len(expected_matches))]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Prompt: {prompt}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
