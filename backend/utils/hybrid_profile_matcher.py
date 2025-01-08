import os
from dotenv import load_dotenv
from openai import OpenAI
import torch
import clip
from PIL import Image
import numpy as np
from pymongo import MongoClient
import re
from typing import List, Dict, Union, Optional
from pathlib import Path

# Load environment variables
load_dotenv()

class HybridProfileMatcher:
    def __init__(self):
        """Initialize OpenAI, CLIP, and MongoDB connections"""
        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))
        
        # CLIP setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # MongoDB setup
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['profile_matching']
        self.profiles_collection = self.db['profiles']

        # Physical attributes that CLIP is good at detecting
        self.physical_attributes = {
            'body_type': [
                'athletic', 'muscular', 'fit', 'slim', 'thin', 'curvy', 
                'plus-size', 'heavy', 'toned', 'built', 'stocky', 'lean'
            ],
            'height': [
                'tall', 'short', 'average height'
            ],
            'style': [
                'well-dressed', 'casual', 'formal', 'professional',
                'trendy', 'fashionable', 'sporty'
            ],
            'features': [
                'bearded', 'clean-shaven', 'long hair', 'short hair',
                'tattoos', 'glasses'
            ]
        }

    def extract_preferences(self, prompt: str) -> Dict:
        """Extract age range, gender, and ethnicity preferences from prompt"""
        preferences = {}
        
        # Extract age range
        age_match = re.search(r'aged? (\d+)-(\d+)|(\d+)-(\d+) \w+', prompt)
        if age_match:
            groups = age_match.groups()
            if groups[0] and groups[1]:
                preferences['min_age'] = int(groups[0])
                preferences['max_age'] = int(groups[1])
            elif groups[2] and groups[3]:
                preferences['min_age'] = int(groups[2])
                preferences['max_age'] = int(groups[3])

        # Extract gender
        prompt_lower = prompt.lower()
        if 'female' in prompt_lower:
            preferences['gender'] = 'female'
        elif 'male' in prompt_lower:
            preferences['gender'] = 'male'

        # Extract ethnicity
        ethnicities = ['white', 'black', 'asian', 'hispanic', 'latin', 'pacific islander', 'indian', 'middle eastern', 'native american']
        found_ethnicities = [eth for eth in ethnicities if eth in prompt_lower]
        if found_ethnicities:
            preferences['ethnicities'] = found_ethnicities

        return preferences

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    def calculate_image_similarity(self, image_path: str, prompt: str) -> float:
        """Calculate similarity between image and prompt using CLIP"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text = clip.tokenize([prompt]).to(self.device)
            
            # Calculate similarities
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            return float(similarity[0][0])
        except Exception as e:
            print(f"Error calculating image similarity: {str(e)}")
            return 0.0

    def detect_physical_attributes(self, prompt: str) -> float:
        """
        Detect mentions of physical attributes in prompt and return
        a weight multiplier for image similarity
        
        Returns:
            float: Weight multiplier between 1.0 and 2.0
        """
        prompt_lower = prompt.lower()
        attribute_count = 0
        total_attributes = 0
        
        # Check each category of physical attributes
        for category, attributes in self.physical_attributes.items():
            for attr in attributes:
                total_attributes += 1
                if attr in prompt_lower:
                    attribute_count += 1
        
        # Calculate weight multiplier:
        # - Base weight: 1.0
        # - Each physical attribute mention adds up to 1.0 extra weight
        # - Maximum multiplier is 2.0 (when many physical attributes are mentioned)
        weight_multiplier = 1.0 + min(1.0, attribute_count / (total_attributes / 4))
        
        return weight_multiplier

    def filter_profiles_by_preferences(self, preferences: Dict) -> List[Dict]:
        """Filter profiles based on extracted preferences"""
        query = {}
        
        # Add age filter
        if 'min_age' in preferences and 'max_age' in preferences:
            query['age'] = {
                '$gte': preferences['min_age'],
                '$lte': preferences['max_age']
            }
        
        # Add gender filter
        if 'gender' in preferences:
            query['gender'] = preferences['gender']
        
        # Add ethnicity filter
        if 'ethnicities' in preferences:
            query['ethnicity'] = {
                '$in': preferences['ethnicities']
            }
        
        return list(self.profiles_collection.find(query))

    def find_matching_profiles(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """
        Find profiles that best match the given prompt using both text and image similarity
        
        Args:
            prompt: User's preference description
            top_k: Number of top matches to return
            
        Returns:
            List of matching profiles with scores
        """
        try:
            # Extract preferences for filtering
            preferences = self.extract_preferences(prompt)
            
            # Filter profiles based on preferences
            filtered_profiles = self.filter_profiles_by_preferences(preferences)
            
            if not filtered_profiles:
                return []
            
            # Generate embedding for the prompt
            prompt_embedding = self.generate_text_embedding(prompt)
            
            # Determine image weight based on physical attributes in prompt
            image_weight_multiplier = self.detect_physical_attributes(prompt)
            base_image_weight = 0.4
            adjusted_image_weight = min(0.8, base_image_weight * image_weight_multiplier)
            text_weight = 1 - adjusted_image_weight
            
            # Calculate similarities for each profile
            results = []
            for profile in filtered_profiles:
                # Calculate text similarity
                profile_text = f"{profile.get('aboutMe', '')} {profile.get('interests', '')} {profile.get('relationshipGoals', '')}"
                profile_embedding = self.generate_text_embedding(profile_text)
                text_similarity = np.dot(prompt_embedding, profile_embedding)
                
                # Calculate image similarity for each photo
                image_similarities = []
                for photo in profile.get('photos', []):
                    image_path = Path(os.getcwd()) / 'backend' / photo['url'].lstrip('/')
                    if image_path.exists():
                        similarity = self.calculate_image_similarity(str(image_path), prompt)
                        image_similarities.append(similarity)
                
                # Calculate average image similarity
                avg_image_similarity = np.mean(image_similarities) if image_similarities else 0
                
                # Calculate combined score with dynamic weights
                combined_score = (text_weight * text_similarity + 
                                adjusted_image_weight * avg_image_similarity)
                
                results.append({
                    'profile': profile,
                    'score': combined_score,
                    'text_similarity': text_similarity,
                    'image_similarity': avg_image_similarity,
                    'weights': {
                        'text': float(text_weight),
                        'image': float(adjusted_image_weight)
                    }
                })
            
            # Sort by combined score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in find_matching_profiles: {str(e)}")
            return []

    def close(self):
        """Close MongoDB connection"""
        self.mongo_client.close()

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='Hybrid Profile Matcher')
    parser.add_argument('--prompt', type=str, help='Search prompt for matching profiles')
    parser.add_argument('--check', action='store_true', help='Check if the matcher is working')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top matches to return')
    args = parser.parse_args()

    try:
        matcher = HybridProfileMatcher()

        if args.check:
            # Just check if everything is initialized properly
            print(json.dumps({
                "status": "ok",
                "message": "Profile matcher is ready"
            }))
            sys.exit(0)

        if not args.prompt:
            print(json.dumps({
                "error": "No prompt provided"
            }))
            sys.exit(1)

        # Find matches
        matches = matcher.find_matching_profiles(args.prompt, args.top_k)
        
        # Convert matches to JSON-serializable format
        results = []
        for match in matches:
            profile = match['profile']
            results.append({
                "profile": {
                    "name": profile.get('name'),
                    "age": profile.get('age'),
                    "gender": profile.get('gender'),
                    "ethnicity": profile.get('ethnicity'),
                    "location": profile.get('location'),
                    "aboutMe": profile.get('aboutMe'),
                    "interests": profile.get('interests'),
                    "photos": profile.get('photos', [])
                },
                "scores": {
                    "total": float(match['score']),
                    "text": float(match['text_similarity']),
                    "image": float(match['image_similarity'])
                }
            })
        
        # Print results as JSON
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({
            "error": str(e)
        }))
        sys.exit(1)
    finally:
        if 'matcher' in locals():
            matcher.close()
