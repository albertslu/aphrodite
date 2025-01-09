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
import json
import logging
import argparse
import traceback
from bson import ObjectId
import sys

# Load environment variables
load_dotenv(dotenv_path=str(Path(__file__).parent.parent.parent / '.env'))

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

class HybridProfileMatcher:
    def __init__(self):
        """Initialize OpenAI, CLIP, and MongoDB connections"""
        try:
            # OpenAI setup
            api_key = os.getenv("SECRET_API_KEY")
            if not api_key:
                raise ValueError("SECRET_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            
            # CLIP setup
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # MongoDB setup
            mongodb_uri = os.getenv('MONGODB_URI')
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable not set")
            self.mongo_client = MongoClient(mongodb_uri)
            self.db = self.mongo_client['profile_matching']
            self.profiles_collection = self.db['profiles']
            
        except Exception as e:
            logger.error(f"Error initializing matcher: {str(e)}")
            raise

    def get_photo_path(self, photo):
        """Convert photo object or string to a full path."""
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        
        # Handle both string and dict photo formats
        if isinstance(photo, dict):
            photo_url = photo.get('url', '')
        else:
            photo_url = str(photo)
        
        # Clean up the path
        photo_name = os.path.basename(photo_url)
        photo_path = os.path.join(uploads_dir, photo_name)
        
        return os.path.normpath(photo_path)

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
        if 'female' in prompt_lower or 'woman' in prompt_lower or 'girl' in prompt_lower:
            preferences['gender'] = 'female'
        elif 'male' in prompt_lower or 'man' in prompt_lower or 'guy' in prompt_lower:
            preferences['gender'] = 'male'

        # Extract ethnicity with more context awareness
        ethnicities = {
            'white': ['white', 'caucasian'],
            'black': ['black', 'african', 'african american'],
            'asian': ['asian', 'east asian', 'southeast asian'],
            'hispanic': ['hispanic', 'latina', 'latino'],
            'latin': ['latin', 'latina', 'latino'],
            'pacific islander': ['pacific islander', 'polynesian'],
            'indian': ['indian', 'south asian'],
            'middle eastern': ['middle eastern', 'arab'],
            'native american': ['native american', 'indigenous', 'first nations']
        }
        
        found_ethnicities = []
        words = prompt_lower.split()
        
        # Check for ethnicity mentions in context
        for ethnicity, variants in ethnicities.items():
            # Check for exact matches
            if any(variant in prompt_lower for variant in variants):
                found_ethnicities.append(ethnicity)
                continue
                
            # Check for ethnicity mentions near relevant words
            context_words = ['person', 'woman', 'man', 'people', 'actress', 'actor']
            for i, word in enumerate(words):
                if any(variant == word for variant in variants):
                    # Check if there's a context word nearby (within 2 words)
                    nearby_words = words[max(0, i-2):min(len(words), i+3)]
                    if any(context in nearby_words for context in context_words):
                        found_ethnicities.append(ethnicity)
                        break
        
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

    def calculate_text_similarity(self, profile: Dict, prompt: str) -> float:
        """Calculate text similarity between profile and prompt"""
        try:
            # Combine relevant profile text fields
            profile_text = f"{profile.get('occupation', '')} {profile.get('aboutMe', '')} {profile.get('interests', '')}"
            
            # Calculate similarity using embeddings
            return self.calculate_embedding_similarity(profile_text, prompt)
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0

    def calculate_image_similarity(self, photo, prompt: str) -> Dict[str, float]:
        """Calculate similarity between image and prompt using CLIP"""
        try:
            # Get full path to image
            image_path = self.get_photo_path(photo)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return {
                    'general_similarity': 0.0,
                    'clarity_score': 0.0
                }
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Prepare text inputs for CLIP
            text_inputs = [prompt]
            
            # Add body type and athletic prompts if relevant
            if any(word in prompt.lower() for word in ['athletic', 'fit', 'muscular', 'strong', 'tall']):
                text_inputs.extend([
                    "a photo of an athletic person",
                    "a photo of someone with muscular build",
                    "a photo of a tall person"
                ])
            
            text_tokens = clip.tokenize(text_inputs).to(self.device)
            
            # Get feature vectors
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize feature vectors
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                
                # Take the maximum similarity score
                general_similarity = float(max(similarities[0]))
                
                # Calculate clarity score
                clarity_tokens = clip.tokenize(["a clear photo of a person", "a blurry or unclear photo"]).to(self.device)
                clarity_features = self.clip_model.encode_text(clarity_tokens)
                clarity_features /= clarity_features.norm(dim=-1, keepdim=True)
                clarity_sim = (image_features @ clarity_features.T).softmax(dim=-1)
                clarity_score = float(clarity_sim[0][0])
                
                return {
                    'general_similarity': general_similarity,
                    'clarity_score': clarity_score
                }
                
        except Exception as e:
            logger.error(f"Error calculating image similarity: {str(e)}")
            return {
                'general_similarity': 0.0,
                'clarity_score': 0.0
            }

    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text embeddings"""
        try:
            # Generate embeddings
            embedding1 = self.generate_text_embedding(text1)
            embedding2 = self.generate_text_embedding(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {str(e)}")
            return 0.0

    def filter_profiles_by_preferences(self, preferences: Dict) -> List[Dict]:
        """Filter profiles based on extracted preferences"""
        try:
            # Build query based on preferences
            query = {}
            
            # Always apply gender filter if specified
            if preferences.get('gender'):
                query['gender'] = preferences['gender']
            
            # Apply age filter if both min and max are specified
            if preferences.get('min_age') and preferences.get('max_age'):
                query['age'] = {
                    '$gte': preferences['min_age'],
                    '$lte': preferences['max_age']
                }
            
            # Apply ethnicity filter if specified, with proper capitalization
            if preferences.get('ethnicities'):
                query['ethnicity'] = {
                    '$in': [eth.capitalize() for eth in preferences['ethnicities']]  # Capitalize to match DB format
                }
            
            # Get filtered profiles
            profiles = list(self.profiles_collection.find(query))
            
            # Post-query filters for more complex criteria
            filtered_profiles = []
            for profile in profiles:
                # Fix photo paths
                if 'photos' not in profile or not profile['photos']:
                    profile['photos'] = []
                else:
                    profile['photos'] = [photo for photo in profile['photos'] if photo]
                
                filtered_profiles.append(profile)
            
            return filtered_profiles

        except Exception as e:
            logger.error(f"Error filtering profiles: {str(e)}")
            return []

    def find_matching_profiles(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """
        Find profiles that match the given prompt.
        
        Args:
            prompt (str): User's search prompt
            top_k (int): Number of top matches to return
        
        Returns:
            List[Dict]: List of matching profiles with scores
        """
        try:
            # Extract preferences from prompt
            preferences = self.extract_preferences(prompt)
            preferences['prompt'] = prompt  # Store original prompt for complex filters
            logger.debug(f"Extracted preferences: {preferences}")
            
            # Get filtered profiles
            filtered_profiles = self.filter_profiles_by_preferences(preferences)
            logger.debug(f"Found {len(filtered_profiles)} profiles after filtering")
            
            if not filtered_profiles:
                logger.warning("No profiles found after filtering")
                return []
            
            # Calculate similarity for each profile
            profile_scores = []
            for profile in filtered_profiles:
                try:
                    # Calculate text similarity
                    text_score = self.calculate_text_similarity(profile, prompt)
                    
                    # Calculate image similarity for each photo
                    image_scores = []
                    for photo in profile.get('photos', []):
                        if not photo:
                            continue
                        image_result = self.calculate_image_similarity(photo, prompt)
                        if image_result['general_similarity'] > 0:
                            image_scores.append(image_result['general_similarity'])
                    
                    # Use best photo score if available, otherwise 0
                    image_score = max(image_scores) if image_scores else 0
                    
                    # Boost scores for athletic/tall profiles if those terms are in prompt
                    boost = 1.0
                    if any(word in prompt.lower() for word in ['athletic', 'fit', 'muscular', 'strong', 'tall']):
                        if any(word in profile.get('interests', '').lower() for word in ['weightlifting', 'crossfit', 'gym', 'fitness']):
                            boost = 1.5
                    
                    # Combine scores with boost
                    final_score = (text_score * 0.7 + image_score * 0.3) * boost
                    
                    # Scale to 0-100 range and ensure it stays within bounds
                    final_score = min(100, max(0, final_score))
                    
                    profile_scores.append({
                        'profile': profile,
                        'matchScore': round(final_score, 1)  # Round to 1 decimal place
                    })
                
                except Exception as e:
                    logger.error(f"Error processing profile {profile.get('name', 'Unknown')}: {str(e)}")
                    continue
            
            # Sort by score in descending order
            profile_scores.sort(key=lambda x: x['matchScore'], reverse=True)
            
            # Return top k results
            return profile_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding matching profiles: {str(e)}")
            return []

    def format_matches_for_json(self, matches: List[Dict]) -> List[Dict]:
        """Format matches for JSON response"""
        try:
            json_matches = []
            for match in matches:
                # Format photos to keep full paths
                photos = []
                for photo in match['profile'].get('photos', []):
                    if isinstance(photo, dict):
                        photos.append(photo.get('url', ''))
                    else:
                        photos.append(str(photo))

                # Create JSON match object with string conversion for ObjectId
                json_match = {
                    'profile': {
                        '_id': str(match['profile'].get('_id', '')),
                        'name': match['profile'].get('name', ''),
                        'bio': match['profile'].get('bio', ''),
                        'interests': match['profile'].get('interests', ''),
                        'occupation': match['profile'].get('occupation', ''),
                        'photos': photos
                    },
                    'matchScore': float(match['matchScore'])
                }
                json_matches.append(json_match)
            
            return json_matches
        except Exception as e:
            logger.error(f"Error formatting matches for JSON: {str(e)}")
            return []

    def close(self):
        """Close MongoDB connection"""
        self.mongo_client.close()

if __name__ == "__main__":
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Profile Matching Script')
        parser.add_argument('--prompt', type=str, required=True, help='Search prompt')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize matcher
        matcher = HybridProfileMatcher()

        # Find matches
        matches = matcher.find_matching_profiles(args.prompt)
        
        if args.debug:
            logging.debug(f"Found {len(matches)} matches after similarity calculation")
        
        # Format matches for JSON response
        json_matches = matcher.format_matches_for_json(matches)
        
        # Convert to JSON string with custom encoder
        json_str = json.dumps(json_matches, cls=JSONEncoder)
        print(json_str)
        sys.stdout.flush()
        
    except Exception as e:
        logger.error(f"Error in profile matching: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty list on error to prevent frontend crashes
        print(json.dumps([]))
        sys.stdout.flush()
    finally:
        if 'matcher' in locals():
            matcher.close()
