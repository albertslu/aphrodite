import os
import json
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

class PromptFaceMatcher:
    def __init__(self, images_dir, prompts_file):
        self.images_dir = images_dir
        self.prompts_file = prompts_file
        self.demographic_data = {}
        self.prompts_data = []
        self.age_weight = 0.3
        self.gender_weight = 0.3
        self.ethnicity_weight = 0.4
        
        # Load prompts
        self.load_prompts()

    def load_prompts(self):
        self.prompts_data = []
        with open(self.prompts_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.prompts_data.append(json.loads(line))

    def extract_demographics_from_prompt(self, prompt):
        # Extract age range
        age_match = re.search(r'aged\s*(\d+)(?:\s*-\s*(\d+))?', prompt.lower())
        age = None
        if age_match:
            if age_match.group(2):  # If there's a range
                age = (int(age_match.group(1)) + int(age_match.group(2))) // 2  # Use average
            else:
                age = int(age_match.group(1))

        # Extract gender
        gender = None
        if 'female' in prompt.lower() or 'woman' in prompt.lower():
            gender = 'Woman'
        elif 'male' in prompt.lower() or 'man' in prompt.lower():
            gender = 'Man'

        # Extract ethnicity (simplified mapping)
        ethnicity_mapping = {
            'asian': 'asian',
            'caucasian': 'white',
            'white': 'white',
            'black': 'black',
            'african': 'black',
            'hispanic': 'latino hispanic',
            'latino': 'latino hispanic',
            'middle eastern': 'middle eastern',
            'indian': 'indian'
        }
        
        ethnicity = None
        for key, value in ethnicity_mapping.items():
            if key in prompt.lower():
                ethnicity = value
                break

        return {
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity
        }

    def analyze_image(self, image_path):
        try:
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['age', 'gender', 'race'],
                enforce_detection=False
            )
            
            # If multiple faces are detected, use the first one
            if isinstance(result, list):
                result = result[0]
            
            return {
                'age': result['age'],
                'gender': result['gender'],
                'ethnicity': result['dominant_race'],
                'ethnicity_scores': result['race']
            }
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_all_images(self):
        self.image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(self.image_files)} images...")
        self.demographic_data = {}
        
        for img_file in tqdm(self.image_files):
            try:
                img_path = os.path.join(self.images_dir, img_file)
                result = DeepFace.analyze(img_path=img_path, 
                                       actions=['age', 'gender', 'race'],
                                       enforce_detection=False,
                                       silent=True)
                
                # Convert numpy float32 to regular Python float
                race_scores = {k: float(v) for k, v in result[0]['race'].items()}
                
                self.demographic_data[img_file] = {
                    'age': float(result[0]['age']),
                    'gender': result[0]['dominant_gender'],
                    'race': race_scores,
                    'dominant_race': result[0]['dominant_race']
                }
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        # Save demographic data
        with open('face_demographic_data.json', 'w') as f:
            json.dump(self.demographic_data, f, indent=2)
        
        print(f"Successfully processed {len(self.demographic_data)} images")

    def calculate_similarity(self, prompt_demo, face_demo):
        if not face_demo:
            return 0.0

        similarity_score = 0
        weights_sum = 0
        
        # Age similarity
        if prompt_demo['age'] is not None:
            age_diff = abs(prompt_demo['age'] - face_demo['age'])
            age_similarity = max(0, 1 - (age_diff / 50))  # Assuming max age difference of 50 years
            similarity_score += self.age_weight * age_similarity
            weights_sum += self.age_weight

        # Gender similarity
        if prompt_demo['gender'] is not None:
            gender_similarity = 1.0 if prompt_demo['gender'] == face_demo['gender'] else 0.0
            similarity_score += self.gender_weight * gender_similarity
            weights_sum += self.gender_weight

        # Ethnicity similarity
        if prompt_demo['ethnicity'] is not None:
            ethnicity_similarity = face_demo['race'].get(prompt_demo['ethnicity'], 0.0)
            similarity_score += self.ethnicity_weight * ethnicity_similarity
            weights_sum += self.ethnicity_weight

        return similarity_score / weights_sum if weights_sum > 0 else 0.0

    def find_matches(self, top_k=5):
        matches = []
        
        for prompt in self.prompts_data:
            prompt_text = prompt['prompt']
            prompt_demo = self.extract_demographics_from_prompt(prompt_text)
            
            # Skip prompts where we couldn't extract any demographic info
            if not any(prompt_demo.values()):
                continue
                
            prompt_matches = []
            for img, face_demo in self.demographic_data.items():
                similarity = self.calculate_similarity(prompt_demo, face_demo)
                prompt_matches.append((img, similarity))
            
            # Sort by similarity score in descending order
            prompt_matches.sort(key=lambda x: x[1], reverse=True)
            
            matches.append({
                'prompt': prompt_text,
                'extracted_demographics': prompt_demo,
                'top_matches': [
                    {
                        'image': img,
                        'similarity_score': score,
                        'demographics': self.demographic_data[img]
                    }
                    for img, score in prompt_matches[:top_k]
                ]
            })
        
        # Save matches to file
        with open('prompt_face_matches.json', 'w') as f:
            json.dump(matches, f, indent=2)
        
        return matches

def main():
    matcher = PromptFaceMatcher('AllFaces', 'generated_200_prompts.jsonl')
    
    # Process all images and save demographic data
    matcher.process_all_images()
    
    # Find matches for all prompts
    matches = matcher.find_matches(top_k=5)
    
    # Print some example matches
    print("\nExample matches:")
    for match in matches[:3]:  # Show first 3 prompts
        print(f"\nPrompt: {match['prompt']}")
        print(f"Extracted demographics: {match['extracted_demographics']}")
        print("Top matches:")
        for m in match['top_matches'][:3]:  # Show top 3 matches
            print(f"  {m['image']}: Similarity score = {m['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
