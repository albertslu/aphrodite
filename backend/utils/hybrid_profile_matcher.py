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
        """Extract age range, gender, and location preferences from prompt"""
        preferences = {}
        prompt_lower = prompt.lower()

        # Extract gender
        if 'female' in prompt_lower or 'woman' in prompt_lower:
            preferences['gender'] = 'female'
        elif 'male' in prompt_lower or 'man' in prompt_lower:
            preferences['gender'] = 'male'

        # Extract age range
        age_match = re.search(r'aged?\s*(\d+)\s*-\s*(\d+)', prompt_lower)
        if age_match:
            preferences['minAge'] = int(age_match.group(1))
            preferences['maxAge'] = int(age_match.group(2))

        # Extract location - look for common location patterns
        location_patterns = [
            r'from\s+(.*?)(?:\.|$|\s+(?:who|and|,))',  # "from Los Angeles." or "from LA who"
            r'in\s+(.*?)(?:\.|$|\s+(?:who|and|,))',    # "in New York." or "in NYC who"
            r'near\s+(.*?)(?:\.|$|\s+(?:who|and|,))',  # "near Chicago." or "near CHI who"
            r'around\s+(.*?)(?:\.|$|\s+(?:who|and|,))', # "around Miami." or "around MIA who"
        ]

        for pattern in location_patterns:
            location_match = re.search(pattern, prompt_lower)
            if location_match:
                preferences['location'] = location_match.group(1).strip()
                break

        # Store original prompt for complex filtering
        preferences['prompt'] = prompt

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
            # Create a comprehensive profile text including occupation and interests
            profile_text = f"{profile.get('occupation', '')} {profile.get('bio', '')} {' '.join(profile.get('interests', []))}"
            
            # Calculate semantic similarity using embeddings
            similarity_score = self.calculate_embedding_similarity(profile_text, prompt)
            
            # Return score
            return similarity_score
            
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
            query = {}
            
            # Basic filters
            if preferences.get('gender'):
                query['gender'] = preferences['gender']
            
            if preferences.get('minAge') and preferences.get('maxAge'):
                query['age'] = {
                    '$gte': preferences['minAge'],
                    '$lte': preferences['maxAge']
                }
            
            # Location filter
            if preferences.get('location'):
                query['location'] = {
                    '$regex': preferences['location'],
                    '$options': 'i'
                }
            
            # Get matching profiles
            return list(self.profiles_collection.find(query))
            
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
            
            # Get filtered profiles based on hierarchical matching (age, gender, location)
            filtered_profiles = self.filter_profiles_by_preferences(preferences)
            
            if not filtered_profiles:
                return []
            
            # Calculate match scores for filtered profiles
            profile_scores = []
            
            for profile in filtered_profiles:
                try:
                    # Calculate text similarity with emphasis on occupation matching
                    occupation_text = profile.get('occupation', '')
                    bio_text = profile.get('bio', '')
                    interests = ' '.join(profile.get('interests', []))
                    
                    # Calculate occupation similarity separately
                    occupation_score = self.calculate_embedding_similarity(occupation_text, prompt) if occupation_text else 0
                    
                    # Calculate general profile text similarity
                    profile_text = f"{bio_text} {interests}"
                    profile_score = self.calculate_embedding_similarity(profile_text, prompt)
                    
                    # Combine text scores (60% occupation, 40% general profile)
                    text_score = (0.6 * occupation_score + 0.4 * profile_score) if occupation_text else profile_score
                    
                    # Calculate image similarity if photos exist
                    image_scores = []
                    for photo in profile.get('photos', []):
                        try:
                            score = self.calculate_image_similarity(photo, prompt)
                            if score is not None:
                                image_scores.append(score)
                        except Exception as e:
                            logger.warning(f"Error calculating image score: {str(e)}")
                            continue
                    
                    # Average image scores if any exist
                    image_score = sum(image_scores) / len(image_scores) if image_scores else 0
                    
                    # Combine scores (70% text, 30% image if images exist)
                    if image_scores:
                        final_score = (0.7 * float(text_score)) + (0.3 * float(image_score))
                    else:
                        final_score = float(text_score)
                    
                    profile_scores.append({
                        'profile': profile,
                        'matchScore': round(final_score, 2),
                        'prompt': prompt
                    })
                
                except Exception as e:
                    logger.error(f"Error processing profile {profile.get('_id', '')}: {str(e)}")
                    continue
            
            # Sort by match score and return top k
            profile_scores.sort(key=lambda x: x['matchScore'], reverse=True)
            return profile_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding matching profiles: {str(e)}")
            return []

    def format_matches_for_json(self, matches: List[Dict]) -> List[Dict]:
        """Format matches for JSON response"""
        try:
            formatted_matches = []
            for match in matches:
                # Get photo paths
                photos = []
                for photo in match['profile'].get('photos', []):
                    if not photo:
                        continue
                    # Handle both string and object photo formats
                    photo_url = photo.get('url', photo) if isinstance(photo, dict) else photo
                    photos.append('/uploads/' + os.path.basename(photo_url))

                # Generate explanation based on profile attributes and match score
                explanation = self.generate_match_explanation(match['profile'], match['matchScore'], match['prompt'])

                # Create JSON match object with string conversion for ObjectId
                json_match = {
                    'profile': {
                        '_id': str(match['profile'].get('_id', '')),
                        'name': match['profile'].get('name', ''),
                        'occupation': match['profile'].get('occupation', ''),
                        'aboutMe': match['profile'].get('aboutMe', ''),
                        'interests': match['profile'].get('interests', ''),
                        'photos': photos,
                        'location': match['profile'].get('location', ''),
                        'aiJustification': {
                            'overallScore': round(match['matchScore'] * 100),
                            'explanation': explanation
                        }
                    },
                    'matchScore': float(match['matchScore'])
                }
                formatted_matches.append(json_match)

            return formatted_matches
            
        except Exception as e:
            logger.error(f"Error formatting matches: {str(e)}")
            return []

    def generate_match_explanation(self, profile: Dict, match_score: float, prompt: str) -> str:
        """Generate a human-readable explanation for why this profile matched"""
        try:
            # Get key profile attributes
            occupation = profile.get('occupation', '')
            location = profile.get('location', '')
            
            # Build explanation based on match criteria
            reasons = []
            
            # Check if prompt is asking about location
            location_check = f"Is this prompt asking about location or area: {prompt}"
            is_location_search = self.calculate_embedding_similarity(location_check, "Yes, this prompt is asking about location") > 0.7
            
            if is_location_search and location:
                # Check if the location matches what's asked for
                location_match = f"Is {location} the same area that this prompt is asking about: {prompt}"
                location_score = self.calculate_embedding_similarity(
                    "Yes, it's the same area",
                    location_match
                )
                if location_score > 0.7:
                    reasons.append(f"located in {location}")

            # Check if prompt is asking about occupation
            occupation_check = f"Is this prompt asking about someone's job or occupation: {prompt}"
            is_occupation_search = self.calculate_embedding_similarity(occupation_check, "Yes, this prompt is asking about occupation") > 0.7
            
            if is_occupation_search and occupation:
                # Check if the occupation matches what's asked for
                occupation_match = f"Is being {occupation} relevant to what this prompt is looking for: {prompt}"
                occupation_score = self.calculate_embedding_similarity(
                    "Yes, it's very relevant",
                    occupation_match
                )
                if occupation_score > 0.7:
                    reasons.append(f"works as {occupation}")
                elif occupation_score > 0.5:
                    reasons.append(f"has a related role as {occupation}")
            
            # Match quality
            if match_score > 0.8:
                confidence = "strong"
            elif match_score > 0.6:
                confidence = "moderate"
            else:
                confidence = "partial"
                
            # Combine into natural language explanation
            if reasons:
                explanation = f"{confidence.capitalize()} match: Profile is {' and '.join(reasons)}"
            else:
                explanation = f"{confidence.capitalize()} match based on overall profile similarity"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Match based on profile similarity"

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
