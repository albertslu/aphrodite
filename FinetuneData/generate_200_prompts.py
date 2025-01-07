import json
import random
from datetime import datetime

# Enhanced lists for more variety
start_phrases = [
    "Looking for", "Hoping to meet", "Searching for", "Interested in finding",
    "Would love to connect with", "Seeking", "Want to meet", "Hoping to find",
    "Dreaming of meeting", "In search of"
]

personality_traits = [
    "adventurous", "intellectual", "creative", "ambitious", "easy-going",
    "passionate", "down-to-earth", "fun-loving", "caring", "honest",
    "spontaneous", "thoughtful", "outgoing", "introverted", "confident",
    "humble", "goal-oriented", "free-spirited", "romantic", "practical"
]

physical_features = [
    "tall", "short", "athletic", "fit", "slim", "curvy",
    "with dark hair", "with light hair", "with a great smile",
    "with an athletic build", "with a slim build", "with a muscular build"
]

interests = [
    "traveling", "hiking", "reading", "cooking", "photography",
    "art", "music", "dancing", "fitness", "yoga", "meditation",
    "outdoor adventures", "trying new restaurants", "gaming",
    "sports", "movies", "theatre", "concerts", "writing", "technology"
]

education_levels = [
    "college-educated", "with a graduate degree", "pursuing higher education",
    "academically driven", "with a PhD", "who values education",
    "with a strong academic background", "intellectually curious"
]

relationship_goals = [
    "for a long-term relationship", "to build something meaningful",
    "for friendship first", "who's ready for commitment",
    "for dating and see where it goes", "with serious intentions",
    "for a genuine connection", "to share life's adventures"
]

locations = [
    "in the Bay Area", "in San Francisco", "in Silicon Valley",
    "in the East Bay", "in Oakland", "in Berkeley",
    "in South Bay", "living in the city"
]

ethnicities = [
    "asian", "white", "black", "hispanic", "latin", "middle eastern",
    "pacific islander", "indian", "native american"
]

def generate_age_range():
    min_age = random.randint(18, 45)
    max_age = min_age + random.randint(2, 8)
    return f"{min_age}-{max_age}"

def generate_prompt():
    templates = [
        # Simple template
        "{start} someone aged {age} {goal}.",
        
        # Physical + personality template
        "{start} a {physical} {gender} aged {age} who is {trait} {goal}.",
        
        # Location-based template
        "{start} someone {location} aged {age}, {trait} and {interest}.",
        
        # Education-focused template
        "{start} a {education} {gender} aged {age} who loves {interest}.",
        
        # Ethnicity + personality template
        "{start} a {ethnicity} {gender} aged {age} who is {trait}.",
        
        # Detailed template
        "{start} someone {physical}, {trait}, and passionate about {interest} aged {age} {location}.",
        
        # Interest-focused template
        "{start} someone who loves {interest} and {interest2}, aged {age} {goal}.",
        
        # Personality-focused template
        "{start} a {trait} and {trait2} person aged {age} {location}.",
    ]
    
    template = random.choice(templates)
    gender = random.choice(["male", "female"])
    
    return template.format(
        start=random.choice(start_phrases),
        age=generate_age_range(),
        physical=random.choice(physical_features),
        trait=random.choice(personality_traits),
        trait2=random.choice(personality_traits),
        interest=random.choice(interests),
        interest2=random.choice(interests),
        education=random.choice(education_levels),
        goal=random.choice(relationship_goals),
        location=random.choice(locations),
        gender=gender,
        ethnicity=random.choice(ethnicities)
    )

# Generate 200 unique prompts
prompts = set()
while len(prompts) < 200:
    prompt = generate_prompt()
    prompts.add(prompt)

# Convert to list and sort for consistency
prompts_list = sorted(list(prompts))

# Save prompts to a file
output_file = "generated_200_prompts.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for prompt in prompts_list:
        json_line = {"prompt": prompt}
        f.write(json.dumps(json_line) + "\n")

print(f"Successfully generated {len(prompts_list)} unique prompts and saved to {output_file}")
