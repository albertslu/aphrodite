import os
import json
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class DemographicMatcher:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.demographic_data = {}
        self.age_weight = 0.3
        self.gender_weight = 0.3
        self.ethnicity_weight = 0.4

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
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.analyze_image, os.path.join(self.images_dir, img)): img 
                for img in image_files
            }
            
            for future in tqdm(as_completed(futures), total=len(image_files)):
                img = futures[future]
                result = future.result()
                if result:
                    self.demographic_data[img] = result

        # Save the demographic data
        with open('demographic_data.json', 'w') as f:
            json.dump(self.demographic_data, f, indent=2)
        
        print(f"Successfully processed {len(self.demographic_data)} images")

    def calculate_similarity(self, img1_data, img2_data):
        if not img1_data or not img2_data:
            return 0.0
        
        # Age similarity (inverse of normalized difference)
        age_diff = abs(img1_data['age'] - img2_data['age'])
        age_similarity = max(0, 1 - (age_diff / 50))  # Assuming max age difference of 50 years
        
        # Gender similarity (binary)
        gender_similarity = 1.0 if img1_data['gender'] == img2_data['gender'] else 0.0
        
        # Ethnicity similarity (cosine similarity of race probabilities)
        ethnicity1 = np.array([float(v) for v in img1_data['ethnicity_scores'].values()])
        ethnicity2 = np.array([float(v) for v in img2_data['ethnicity_scores'].values()])
        ethnicity_similarity = np.dot(ethnicity1, ethnicity2) / (np.linalg.norm(ethnicity1) * np.linalg.norm(ethnicity2))
        
        # Weighted sum
        total_similarity = (
            self.age_weight * age_similarity +
            self.gender_weight * gender_similarity +
            self.ethnicity_weight * ethnicity_similarity
        )
        
        return total_similarity

    def find_similar_images(self, target_image, top_k=5):
        if target_image not in self.demographic_data:
            target_data = self.analyze_image(os.path.join(self.images_dir, target_image))
            if not target_data:
                return []
        else:
            target_data = self.demographic_data[target_image]
        
        similarities = []
        for img, data in self.demographic_data.items():
            if img != target_image:
                similarity = self.calculate_similarity(target_data, data)
                similarities.append((img, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def main():
    matcher = DemographicMatcher('AllFaces')
    
    # Process all images and save demographic data
    matcher.process_all_images()
    
    # Example: Find similar images for a specific target
    example_image = os.listdir('AllFaces')[0]  # Use first image as example
    similar_images = matcher.find_similar_images(example_image)
    
    print(f"\nMost similar images to {example_image}:")
    for img, score in similar_images:
        print(f"{img}: Similarity score = {score:.3f}")

if __name__ == "__main__":
    main()
