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
load_dotenv(dotenv_path=str(Path(__file__).parent.parent.parent / '.env'))

class HybridProfileMatcher:
    def __init__(self):
        """Initialize OpenAI, CLIP, and MongoDB connections"""
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

        # Physical attributes that CLIP is good at detecting
        self.physical_attributes = {
            'body_type': [
                'athletic', 'muscular', 'fit', 'slim', 'thin', 'curvy', 
                'plus-size', 'heavy', 'toned', 'built', 'stocky', 'lean'
            ],
            'height': [
                'tall', 'short', 'average height'
            ],
            'hair_color': [
                'blonde hair', 'brown hair', 'black hair', 'red hair',
                'dark hair', 'light hair', 'platinum blonde'
            ],
            'eye_color': [
                'blue eyes', 'brown eyes', 'green eyes', 'hazel eyes',
                'dark eyes', 'light eyes'
            ],
            'facial_features': [
                'bearded', 'clean-shaven', 'long hair', 'short hair',
                'wavy hair', 'straight hair', 'curly hair'
            ],
            'style': [
                'well-dressed', 'casual', 'formal', 'professional',
                'trendy', 'fashionable', 'sporty'
            ],
            'other_features': [
                'tattoos', 'glasses', 'makeup', 'natural look'
            ]
        }

        # Weights for different attribute categories when mentioned in prompt
        self.attribute_weights = {
            'body_type': 1.0,
            'height': 0.75,     # Lower weight since height is hard to determine from photos
            'hair_color': 1.5,  # Higher weight for hair color
            'eye_color': 1.5,   # Higher weight for eye color
            'facial_features': 1.2,
            'style': 0.8,
            'other_features': 0.8
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
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Prepare text inputs for CLIP
            prompt_tokens = prompt.replace(',', ' ').split()
            
            # Add athletic-specific prompts if relevant
            if any(word in prompt.lower() for word in ['athletic', 'fit', 'muscular', 'strong']):
                athletic_prompts = [
                    "a photo of an athletic person",
                    "a photo of someone with muscular build",
                    "a photo showing fitness and strength",
                    "a photo of someone who works out",
                ]
                text_inputs = [prompt] + athletic_prompts
            else:
                text_inputs = [prompt]
                
            if physical_attributes:
                text_inputs.extend([f"a photo of a person with {attr}" for attr in physical_attributes])
                
            text_tokens = clip.tokenize(text_inputs).to(self.device)
            
            # Get feature vectors
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize feature vectors
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                general_similarity = float(similarities[0][0])
                
                # For athletic prompts, take the max similarity with any athletic-specific prompt
                if any(word in prompt.lower() for word in ['athletic', 'fit', 'muscular', 'strong']):
                    athletic_similarity = float(max(similarities[0][1:5]))  # Indices 1-4 are athletic prompts
                    general_similarity = max(general_similarity, athletic_similarity)
            
            # Calculate attribute confidences with category weights
            attribute_confidences = []
            attribute_weights = []
            
            for attr in (physical_attributes or []):
                # Find which category this attribute belongs to
                category = None
                for cat, attrs in self.physical_attributes.items():
                    if any(a in attr.lower() for a in attrs):
                        category = cat
                        break
                
                # Get the weight for this category
                weight = self.attribute_weights.get(category, 1.0)
                
                # Calculate confidence score for this attribute
                attr_text = f"a photo of a person with {attr}"
                attr_tokens = clip.tokenize([attr_text]).to(self.device)
                with torch.no_grad():
                    attr_features = self.clip_model.encode_text(attr_tokens)
                    attr_features /= attr_features.norm(dim=-1, keepdim=True)
                    attr_similarity = float((100.0 * image_features @ attr_features.T).softmax(dim=-1)[0][0])
                
                attribute_confidences.append(attr_similarity * weight)
                attribute_weights.append(weight)
        
            # Calculate clarity confidence (how clearly the person is visible)
            clarity_tokens = clip.tokenize(["a clear photo of a person's face", "a blurry or unclear photo"]).to(self.device)
            with torch.no_grad():
                clarity_features = self.clip_model.encode_text(clarity_tokens)
                clarity_features /= clarity_features.norm(dim=-1, keepdim=True)
                clarity_similarities = (100.0 * image_features @ clarity_features.T).softmax(dim=-1)
                clarity_confidence = float(clarity_similarities[0][0])
            
            # Calculate weighted attribute confidence
            if attribute_confidences:
                weighted_confidence = sum(conf * weight for conf, weight in zip(attribute_confidences, attribute_weights))
                avg_attribute_confidence = weighted_confidence / sum(attribute_weights)
            else:
                avg_attribute_confidence = 0
            
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
            profile_text = f"{profile.get('occupation', '')} {profile.get('aboutMe', '')} {profile.get('interests', '')}"
            
            # Add extra weight for athletic/fitness-related content when relevant
            athletic_keywords = {
                # Strong athletic indicators (3.0x boost)
                'personal trainer': 3.0,
                'athlete': 3.0,
                'wrestler': 3.0,
                'bodybuilder': 3.0,
                'fitness instructor': 3.0,
                'crossfit': 3.0,
                
                # Clear fitness focus (2.0x boost)
                'weightlifting': 2.0,
                'fitness': 2.0,
                'gym': 2.0,
                'sports': 2.0,
                'athletic': 2.0,
                'muscular': 2.0,
                'exercise science': 2.0,
                
                # Fitness-related activities (1.5x boost)
                'workout': 1.5,
                'training': 1.5,
                'nutrition': 1.5,
                'strength': 1.5,
                'hiking': 1.5,
                'outdoor sports': 1.5
            }
            
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ['athletic', 'fit', 'muscular', 'strong', 'gym', 'fitness']):
                # Check profile text for athletic keywords
                score_multiplier = 1.0
                profile_lower = profile_text.lower()
                
                # Check for keyword matches
                for keyword, weight in athletic_keywords.items():
                    if keyword in profile_lower:
                        score_multiplier = max(score_multiplier, weight)
                
                # Extra boost for athletic education
                education = profile.get('education', '').lower()
                if any(term in education for term in ['exercise', 'sports', 'physical education', 'kinesiology']):
                    score_multiplier = max(score_multiplier, 2.0)
                
                # Extra boost for athletic occupation
                occupation = profile.get('occupation', '').lower()
                if any(term in occupation for term in ['trainer', 'athlete', 'fitness', 'coach']):
                    score_multiplier = max(score_multiplier, 3.0)
                
                # Generate and adjust similarity score
                base_similarity = self.calculate_embedding_similarity(profile_text, prompt)
                return base_similarity * score_multiplier
            
            return self.calculate_embedding_similarity(profile_text, prompt)
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0

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
                    
                    # Combine scores - text similarity has more weight for attribute matching
                    final_score = (text_score * 0.7 + image_score * 0.3) * 100
                    
                    # Ensure score is between 0 and 100
                    final_score = min(100, max(0, final_score))
                    
                    # Filter out very low scoring profiles
                    if final_score >= 20:  # Only include profiles with at least 20% match
                        profile_scores.append({
                            'profile': profile,
                            'score': final_score
                        })
                
                except Exception as e:
                    logger.error(f"Error processing profile {profile.get('name', 'Unknown')}: {str(e)}")
                    continue
            
            # Sort by score in descending order
            profile_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top k results
            return profile_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding matching profiles: {str(e)}")
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
                'score': float(match['score'])
            }
            json_matches.append(json_match)
            
        print(json.dumps(json_matches))
        sys.stdout.flush()
        
        matcher.close()
        
    except Exception as e:
        logging.error(f"Error in profile matching: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
