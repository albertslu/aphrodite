import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv('SECRET_API_KEY')

# Function to generate embeddings
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Load profiles
with open("profiles.json", "r") as file:
    profiles = json.load(file)

# Generate embeddings for profiles
profile_embeddings = []
for profile in profiles:
    profile_text = profile['essay0']  # Use relevant fields (e.g., essay0)
    embedding = generate_embedding(profile_text)
    profile_embeddings.append(embedding)

# Save profile embeddings
np.save("profile_embeddings.npy", profile_embeddings)

# Define prompts
prompts = [
    "Looking for someone adventurous and kind-hearted, aged 25-30.",
    "Seeking a tall, athletic partner who loves hiking.",
    # Add more prompts here
]

# Generate embeddings for prompts
prompt_embeddings = []
for prompt in prompts:
    embedding = generate_embedding(prompt)
    prompt_embeddings.append(embedding)

# Calculate cosine similarity
results = []
for i, prompt_embedding in enumerate(prompt_embeddings):
    similarities = cosine_similarity([prompt_embedding], profile_embeddings)[0]
    top_matches = np.argsort(similarities)[-3:][::-1]  # Top 3 matches
    results.append({
        "prompt": prompts[i],
        "top_matches": top_matches.tolist(),
        "similarity_scores": similarities[top_matches].tolist()
    })

# Save results
with open("similarity_results.json", "w") as file:
    json.dump(results, file)

print("Similarity calculations saved to similarity_results.json")
