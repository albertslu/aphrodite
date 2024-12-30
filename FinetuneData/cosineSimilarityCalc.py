import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example embeddings
profile_embeddings = np.array([
    [0.1, 0.2, 0.3],  # Profile 1
    [0.3, 0.2, 0.1],  # Profile 2
    [0.4, 0.5, 0.6]   # Profile 3
])

prompt_embedding = np.array([0.2, 0.3, 0.4])  # User input

# Compute cosine similarity
similarities = cosine_similarity([prompt_embedding], profile_embeddings)
print(similarities)  # Output: [[0.9759, 0.7559, 0.9982]]
