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

    def calculate_image_similarity(self, image_path: str, prompt: str, physical_attributes: List[str] = None) -> Dict[str, float]:
        """
        Calculate similarity between image and prompt using CLIP, including physical attribute detection
        
        Args:
            image_path: Path to the image
            prompt: User's search prompt
            physical_attributes: List of physical attributes to specifically check for
            
        Returns:
            Dict containing general similarity and attribute detection confidences
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Calculate general prompt similarity
            text = clip.tokenize([prompt]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                general_similarity = float((100.0 * image_features @ text_features.T).softmax(dim=-1)[0][0])
            
            # If no physical attributes to check, return general similarity
            if not physical_attributes:
                return {
                    'general_similarity': general_similarity,
                    'attribute_confidence': 0.0,
                    'clarity_score': 0.0,
                    'attribute_scores': {}
                }
            
            # Check for specific physical attributes
            attribute_prompts = [
                f"a photo clearly showing a person with {attr}" for attr in physical_attributes
            ]
            attribute_prompts += [
                f"a clear full-body photo of a person",
                f"a clear photo showing someone's physical appearance"
            ]
            
            text_tokens = clip.tokenize(attribute_prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity for each attribute prompt
                attribute_similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get individual attribute confidences
                attribute_confidences = [float(s) for s in attribute_similarities[0][:len(physical_attributes)]]
                
                # Get photo clarity confidence (how well it shows the person)
                clarity_confidence = float(attribute_similarities[0][-2:].mean())
            
            # Calculate weighted attribute confidence
            avg_attribute_confidence = np.mean(attribute_confidences) if attribute_confidences else 0
            
            # Final attribute confidence is affected by both specific attribute detection
            # and how clearly the photo shows the person
            final_attribute_confidence = avg_attribute_confidence * clarity_confidence
            
            return {
                'general_similarity': general_similarity,
                'attribute_confidence': final_attribute_confidence,
                'clarity_score': clarity_confidence,
                'attribute_scores': dict(zip(physical_attributes, attribute_confidences))
            }
            
        except Exception as e:
            print(f"Error calculating image similarity: {str(e)}")
            return {
                'general_similarity': 0.0,
                'attribute_confidence': 0.0,
                'clarity_score': 0.0,
                'attribute_scores': {}
            }

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
            
            # Get physical attributes mentioned in prompt
            detected_attributes = []
            for category, attributes in self.physical_attributes.items():
                for attr in attributes:
                    if attr in prompt.lower():
                        detected_attributes.append(attr)
            
            # Determine base image weight based on physical attributes in prompt
            image_weight_multiplier = self.detect_physical_attributes(prompt)
            base_image_weight = 0.4
            max_image_weight = min(0.8, base_image_weight * image_weight_multiplier)
            
            # Calculate similarities for each profile
            results = []
            for profile in filtered_profiles:
                # Calculate text similarity
                profile_text = f"{profile.get('aboutMe', '')} {profile.get('interests', '')} {profile.get('relationshipGoals', '')}"
                profile_embedding = self.generate_text_embedding(profile_text)
                text_similarity = np.dot(prompt_embedding, profile_embedding)
                
                # Calculate image similarity for each photo
                image_results = []
                for photo in profile.get('photos', []):
                    image_path = Path(os.getcwd()) / 'backend' / photo['url'].lstrip('/')
                    if image_path.exists():
                        image_result = self.calculate_image_similarity(
                            str(image_path), 
                            prompt,
                            detected_attributes
                        )
                        image_results.append(image_result)
                
                if image_results:
                    # Calculate average similarities and confidences
                    avg_general_similarity = np.mean([r['general_similarity'] for r in image_results])
                    avg_attribute_confidence = np.mean([r['attribute_confidence'] for r in image_results])
                    avg_clarity = np.mean([r['clarity_score'] for r in image_results])
                    
                    # Adjust image weight based on detection confidence
                    # If we can't clearly see the attributes, reduce the image weight
                    confidence_factor = avg_attribute_confidence * avg_clarity
                    adjusted_image_weight = max_image_weight * confidence_factor
                else:
                    avg_general_similarity = 0
                    avg_attribute_confidence = 0
                    avg_clarity = 0
                    adjusted_image_weight = base_image_weight
                
                # Ensure text weight + image weight = 1
                text_weight = 1 - adjusted_image_weight
                
                # Calculate combined score with adjusted weights
                combined_score = (text_weight * text_similarity + 
                                adjusted_image_weight * avg_general_similarity)
                
                results.append({
                    'profile': profile,
                    'score': combined_score,
                    'text_similarity': text_similarity,
                    'image_analysis': {
                        'similarity': avg_general_similarity,
                        'attribute_confidence': avg_attribute_confidence,
                        'clarity': avg_clarity
                    },
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
                    "image": {
                        "similarity": float(match['image_analysis']['similarity']),
                        "attribute_confidence": float(match['image_analysis']['attribute_confidence']),
                        "clarity": float(match['image_analysis']['clarity'])
                    }
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
