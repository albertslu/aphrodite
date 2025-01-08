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
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

class HybridProfileMatcher:
    def __init__(self):
        """Initialize OpenAI, CLIP, and MongoDB connections"""
        # OpenAI setup
        self.client = OpenAI(api_key=os.getenv("SECRET_API_KEY"))
        
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

    def get_photo_path(self, photo):
        """Convert photo object or string to a full path."""
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        
        if isinstance(photo, dict):
            photo_url = photo.get('url', '').lstrip('/')
        else:
            photo_url = str(photo).lstrip('/')
        
        # Try to find a matching file in uploads directory
        photo_name = os.path.basename(photo_url)
        if os.path.exists(os.path.join(uploads_dir, photo_name)):
            return os.path.normpath(os.path.join(uploads_dir, photo_name))
        
        # If exact match not found, try to find a file with similar name
        for filename in os.listdir(uploads_dir):
            # Skip temporary files
            if filename.startswith('.') or filename.startswith('~'):
                continue
                
            # Try to match the profile type (athletic, artistic, etc.)
            profile_type = photo_name.split('_')[0]
            if filename.startswith(profile_type):
                if photo_name.endswith('1.jpg') and filename.endswith('1.jpg'):
                    return os.path.normpath(os.path.join(uploads_dir, filename))
                elif photo_name.endswith('2.jpg') and filename.endswith('2.jpg'):
                    return os.path.normpath(os.path.join(uploads_dir, filename))
        
        # If no match found, return original path
        return os.path.normpath(os.path.join(uploads_dir, photo_name))

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
            # Ensure image_path is a proper path
            if isinstance(image_path, dict):
                image_path = self.get_photo_path(image_path)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return {
                    'general_similarity': 0.0,
                    'attribute_confidence': 0.0,
                    'clarity_score': 0.0,
                    'attribute_scores': {}
                }
            
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

    def calculate_text_similarity(self, profile: Dict, prompt: str) -> float:
        """Calculate text similarity between profile and prompt"""
        try:
            # Combine relevant profile text fields
            profile_text = " ".join(filter(None, [
                profile.get('name', ''),
                profile.get('aboutMe', ''),
                profile.get('interests', ''),
                profile.get('occupation', ''),
                profile.get('height', ''),
                profile.get('ethnicity', '')
            ]))
            
            if not profile_text.strip():
                return 0.0
            
            # Generate embeddings
            profile_embedding = self.generate_text_embedding(profile_text)
            prompt_embedding = self.generate_text_embedding(prompt)
            
            # Calculate cosine similarity
            similarity = float(np.dot(profile_embedding, prompt_embedding) / 
                            (np.linalg.norm(profile_embedding) * np.linalg.norm(prompt_embedding)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
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

    def height_to_inches(self, height_str: str) -> Optional[int]:
        """Convert height string (e.g. "5'10"") to inches"""
        try:
            if not height_str:
                return None
                
            # Remove any whitespace and double quotes
            height_str = height_str.strip().replace('"', '')
            
            # Try to parse X'Y" format
            match = re.match(r"(\d+)'(\d+)", height_str)
            if match:
                feet, inches = map(int, match.groups())
                return feet * 12 + inches
                
            # If it's already in inches (just a number), return it
            if height_str.isdigit():
                return int(height_str)
                
            return None
        except Exception:
            return None

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
            
            # Apply ethnicity filter if specified
            if preferences.get('ethnicities'):
                query['ethnicity'] = {
                    '$in': preferences['ethnicities']
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
                
                # Height requirements
                if "tall" in preferences.get('prompt', '').lower():
                    height_str = profile.get('height')
                    height_inches = self.height_to_inches(height_str)
                    if height_inches is not None:  # Only check if height is specified and valid
                        if profile['gender'] == 'male' and height_inches < 70:  # 5'10" (allowing 2 inches below 6'0")
                            continue
                        elif profile['gender'] == 'female' and height_inches < 66:  # 5'6" (allowing 2 inches below 5'8")
                            continue
                elif "short" in preferences.get('prompt', '').lower():
                    height_str = profile.get('height')
                    height_inches = self.height_to_inches(height_str)
                    if height_inches is not None:  # Only check if height is specified and valid
                        if profile['gender'] == 'male' and height_inches > 67:  # 5'7"
                            continue
                        elif profile['gender'] == 'female' and height_inches > 63:  # 5'3"
                            continue
                
                # Education requirements
                if any(term in preferences.get('prompt', '').lower() for term in ['college', 'university', 'degree']):
                    education = profile.get('education', '').lower()
                    if not any(term in education for term in ['college', 'university', 'degree', 'masters', 'phd']):
                        continue
                
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
                    
                    # Calculate image similarity if photos exist
                    image_score = 0.0
                    photo_count = 0
                    
                    if profile.get('photos'):
                        image_results = []
                        physical_attrs = [attr for attrs in self.physical_attributes.values() for attr in attrs]
                        
                        for photo in profile['photos']:
                            try:
                                image_path = self.get_photo_path(photo)
                                logger.debug(f"Processing image: {image_path}")
                                
                                if os.path.exists(image_path):
                                    result = self.calculate_image_similarity(image_path, prompt, physical_attrs)
                                    if result['general_similarity'] > 0:
                                        # Weight image scores by both general similarity and specific attribute detection
                                        weighted_score = (
                                            result['general_similarity'] * 0.4 +  # General visual match
                                            result['attribute_confidence'] * 0.4 +  # Specific attribute detection
                                            result['clarity_score'] * 0.2  # Photo quality/clarity
                                        )
                                        image_results.append(weighted_score)
                                        photo_count += 1
                            except Exception as e:
                                logger.error(f"Error processing image {photo}: {str(e)}")
                                continue
                        
                        if image_results:
                            image_score = max(image_results)  # Use best matching photo
                    
                    # Weight scores based on whether physical attributes were mentioned
                    physical_weight = self.detect_physical_attributes(prompt)
                    
                    # If physical attributes mentioned heavily, prioritize image matching
                    if physical_weight > 1.5:
                        final_score = (text_score * 0.3) + (image_score * 0.7 if photo_count > 0 else 0)
                    else:
                        final_score = (text_score * 0.7) + (image_score * 0.3 if photo_count > 0 else 0)
                    
                    profile_scores.append({
                        'profile': profile,
                        'matchScore': float(final_score)
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing profile {profile.get('_id')}: {str(e)}")
                    continue
            
            # Sort by score and return top k
            matches = sorted(profile_scores, key=lambda x: x['matchScore'], reverse=True)[:top_k]
            logger.info(f"Found {len(matches)} matches out of {len(filtered_profiles)} filtered profiles")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in find_matching_profiles: {str(e)}")
            return []

    def close(self):
        """Close MongoDB connection"""
        self.mongo_client.close()

if __name__ == "__main__":
    import argparse
    import json
    import sys
    import logging

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
        logging.basicConfig(level=logging.DEBUG, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        matcher = HybridProfileMatcher()
        preferences = matcher.extract_preferences(args.prompt)
        
        if args.debug:
            logging.debug(f"Extracted preferences: {preferences}")
        
        filtered_profiles = matcher.filter_profiles_by_preferences(preferences)
        
        if args.debug:
            logging.debug(f"Found {len(filtered_profiles)} profiles after filtering")
        
        matches = matcher.find_matching_profiles(args.prompt)
        
        if args.debug:
            logging.debug(f"Found {len(matches)} matches after similarity calculation")
        
        # Convert matches to JSON-serializable format
        json_matches = []
        for match in matches:
            # Keep photo URLs as they are in the database
            photos = []
            for photo in match['profile'].get('photos', []):
                if isinstance(photo, dict):
                    photos.append(photo['url'])
                else:
                    photos.append(str(photo))

            json_match = {
                'profile': {
                    '_id': str(match['profile']['_id']),
                    'name': match['profile'].get('name', ''),
                    'bio': match['profile'].get('aboutMe', ''),
                    'interests': match['profile'].get('interests', []),
                    'occupation': match['profile'].get('occupation', ''),
                    'photos': photos
                },
                'matchScore': float(match['matchScore'])
            }
            json_matches.append(json_match)
            
        print(json.dumps(json_matches))
        sys.stdout.flush()
        
        matcher.close()
        
    except Exception as e:
        logging.error(f"Error in profile matching: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
