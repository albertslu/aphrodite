import os
from dotenv import load_dotenv
from openai import OpenAI
import torch
import clip
from PIL import Image
import numpy as np
import re
from typing import List, Dict, Union, Optional
from pathlib import Path
import json
import logging
import argparse
import traceback
from bson import ObjectId
import sys
from pymongo import MongoClient
from torch.nn.functional import cosine_similarity as F

# Load environment variables
load_dotenv(dotenv_path=str(Path(__file__).parent.parent.parent / '.env'))

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

class HybridProfileMatcher:
    # Class-level variables for caching
    _instance = None
    _clip_model = None
    _preprocess = None
    _mongo_client = None
    _openai_client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridProfileMatcher, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize OpenAI, CLIP, and MongoDB connections"""
        # Skip if already initialized
        if getattr(self, '_initialized', False):
            return
            
        try:
            # OpenAI setup - only create client once
            if self._openai_client is None:
                api_key = os.getenv("SECRET_API_KEY")
                if not api_key:
                    raise ValueError("SECRET_API_KEY environment variable not set")
                self._openai_client = OpenAI(api_key=api_key)
            self.client = self._openai_client
            
            # CLIP setup - only load model once
            if self._clip_model is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._clip_model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model = self._clip_model
            self.preprocess = self._preprocess
            
            # Add embeddings caches
            self.image_embeddings_cache = {}
            self.text_embeddings_cache = {}
            print("Initialized caches", file=sys.stderr)
            
            # MongoDB setup - only connect once
            if self._mongo_client is None:
                mongodb_uri = os.getenv('MONGODB_URI')
                if not mongodb_uri:
                    raise ValueError("MONGODB_URI environment variable not set")
                    
                # Simple MongoDB connection with shorter timeouts
                self._mongo_client = MongoClient(
                    mongodb_uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                # Test connection once
                self._mongo_client.admin.command('ping')
                
            self.mongo_client = self._mongo_client
            self.db = self.mongo_client['profile_matching']
            self.profiles_collection = self.db['profiles']
            
            self._initialized = True
            
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
        elif 'any age' in prompt_lower:
            # Don't set age preferences for "any age"
            pass

        # Extract ethnicity
        ethnicity_keywords = {
            'white': ['white', 'caucasian', 'european'],
            'asian': ['asian', 'oriental'],
            'black': ['black', 'african'],
            'hispanic': ['hispanic', 'latina', 'latino'],
            'south asian': ['indian', 'south asian'],
            'middle eastern': ['middle eastern', 'arab'],
        }
        
        for ethnicity, keywords in ethnicity_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                preferences['ethnicity'] = ethnicity
                break

        # Extract location - look for common location patterns
        if 'anywhere' in prompt_lower or 'any location' in prompt_lower:
            preferences['location'] = 'anywhere'
        else:
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

        # Extract height preferences
        if 'tall' in prompt_lower:
            # Men: tall >= 72 inches (6'0"), Women: tall >= 68 inches (5'8")
            # Allow -2 inches flexibility
            preferences['height_requirement'] = 'tall'
        elif 'short' in prompt_lower:
            # Men: short <= 67 inches (5'7"), Women: short <= 63 inches (5'3")
            preferences['height_requirement'] = 'short'

        logger.debug(f"Extracted preferences: {preferences}")
        return preferences

    def extract_physical_traits(self, prompt: str) -> Dict[str, str]:
        """Extract physical traits from the prompt more intelligently"""
        traits = {}
        prompt_lower = prompt.lower()
        
        # Hair color detection
        hair_colors = {
            'blonde': ['blonde hair', 'blond hair', 'golden hair'],
            'black': ['black hair', 'dark hair'],
            'brown': ['brown hair', 'brunette'],
            'red': ['red hair', 'ginger hair', 'redhead']
        }
        
        for color, patterns in hair_colors.items():
            if any(pattern in prompt_lower for pattern in patterns):
                traits['hair_color'] = color
                break
        
        # Eye color detection
        eye_colors = {
            'blue': ['blue eyes', 'light blue eyes'],
            'brown': ['brown eyes', 'dark eyes'],
            'green': ['green eyes'],
            'hazel': ['hazel eyes']
        }
        
        for color, patterns in eye_colors.items():
            if any(pattern in prompt_lower for pattern in patterns):
                traits['eye_color'] = color
                break
                
        return traits

    def get_text_embedding(self, text):
        """Get text embedding with caching."""
        if text in self.text_embeddings_cache:
            print(f"TEXT CACHE HIT: {text[:30]}...", file=sys.stderr)
            return self.text_embeddings_cache[text]
        
        print(f"TEXT CACHE MISS: {text[:30]}...", file=sys.stderr)
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response.data[0].embedding
        self.text_embeddings_cache[text] = embedding
        return embedding

    def calculate_text_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts using embeddings."""
        try:
            if not text1 or not text2:
                return 0.0
                
            # Get embeddings using cache
            embedding1 = self.get_text_embedding(text1)
            embedding2 = self.get_text_embedding(text2)
            
            # Convert to numpy arrays and calculate similarity
            embedding1_np = np.array(embedding1)
            embedding2_np = np.array(embedding2)
            
            # Normalize the vectors
            embedding1_normalized = embedding1_np / np.linalg.norm(embedding1_np)
            embedding2_normalized = embedding2_np / np.linalg.norm(embedding2_np)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI's API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two text embeddings"""
        try:
            # Generate embeddings
            embedding1 = self.generate_text_embedding(text1)
            embedding2 = self.generate_text_embedding(text2)
            
            # Calculate cosine similarity
            similarity = float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {str(e)}")
            return 0.0

    def calculate_image_similarity(self, photo, prompt: str) -> float:
        """Calculate similarity between image and prompt using CLIP"""
        try:
            # Get full path to image
            image_path = self.get_photo_path(photo)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return 0.0
            
            # Check cache first
            cached_features = None
            if image_path in self.image_embeddings_cache:
                cached_features = self.image_embeddings_cache[image_path]
                print(f"CACHE HIT: Using cached embedding for {os.path.basename(image_path)}", file=sys.stderr)
                logger.info(f"CACHE HIT: Using cached embedding for {os.path.basename(image_path)}")
            else:
                print(f"CACHE MISS: Generating new embedding for {os.path.basename(image_path)}", file=sys.stderr)
                logger.info(f"CACHE MISS: Generating new embedding for {os.path.basename(image_path)}")
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Generate and cache embedding
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    self.image_embeddings_cache[image_path] = image_features
                    cached_features = image_features
                print(f"CACHED: New embedding generated and cached for {os.path.basename(image_path)}", file=sys.stderr)
                logger.info(f"CACHED: New embedding generated and cached for {os.path.basename(image_path)}")
            
            # Extract specific physical traits from prompt
            traits = self.extract_physical_traits(prompt)
            scores = []
            
            if traits:
                # Build specific CLIP prompts based on detected traits
                clip_prompts = []
                
                if 'hair_color' in traits:
                    color = traits['hair_color']
                    clip_prompts.extend([
                        f"person with {color} colored hair",
                        f"head shot showing {color} hair",
                        f"close up of {color} hair color"
                    ])
                    logger.debug(f"Using hair color prompts for {os.path.basename(image_path)}: {clip_prompts}")
                
                if 'eye_color' in traits:
                    color = traits['eye_color']
                    clip_prompts.extend([
                        f"person with {color} colored eyes",
                        f"face with {color} eyes",
                        f"close up of {color} eyes"
                    ])
                    logger.debug(f"Using eye color prompts for {os.path.basename(image_path)}: {clip_prompts}")
                
                if clip_prompts:
                    text_inputs = clip.tokenize(clip_prompts).to(self.device)
                    with torch.no_grad():
                        text_features = self.clip_model.encode_text(text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        
                        # Use cached image features
                        similarities = cached_features @ text_features.T
                        
                        # Convert to range 0-1 with enhanced scaling
                        similarities = (similarities + 1) / 2  # Now between 0 and 1
                        
                        # Apply non-linear scaling to boost high scores while keeping low scores low
                        similarities = torch.pow(similarities, 0.5)  # Square root to boost scores
                        similarities = (similarities - 0.4) / 0.6  # Rescale above threshold
                        similarities = torch.clamp(similarities, 0, 1)  # Ensure 0-1 range
                        
                        # Log raw similarities before processing
                        for i, prompt in enumerate(clip_prompts):
                            logger.debug(f"Raw CLIP score for '{prompt}' on {os.path.basename(image_path)}: {float(similarities[0][i])}")
                        
                        # Group scores by trait
                        if 'hair_color' in traits:
                            hair_scores = similarities[:, :3]  # First 3 prompts are for hair
                            hair_score = float(hair_scores.max().item())
                            scores.append(hair_score)
                            logger.debug(f"Final hair color score for {os.path.basename(image_path)}: {hair_score}")
                        
                        if 'eye_color' in traits and len(similarities[0]) > 3:
                            eye_scores = similarities[:, 3:]  # Last 3 prompts are for eyes
                            eye_score = float(eye_scores.max().item())
                            scores.append(eye_score)
                            logger.debug(f"Final eye color score for {os.path.basename(image_path)}: {eye_score}")
                
                if scores:
                    # Use minimum score to ensure all requested traits must match
                    final_score = min(scores)
                    # Apply threshold - if score is too low, return 0
                    if final_score < 0.4:  # Threshold for trait matching
                        final_score = 0.0
                    else:
                        # Boost scores above threshold to spread them out more
                        final_score = 0.4 + (final_score - 0.4) * 1.5
                    logger.debug(f"Final trait score for {os.path.basename(image_path)}: {final_score}")
                    return final_score
            
            # For non-physical trait prompts or if no traits detected, use general matching
            text_inputs = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Calculate similarity with enhanced scaling using cached features
                similarity = float((cached_features @ text_features.T + 1) / 2)
                if similarity >= 0.4:
                    similarity = 0.4 + (similarity - 0.4) * 1.5
                else:
                    similarity = 0.0
                logger.debug(f"General similarity score for {os.path.basename(image_path)}: {similarity}")
                return similarity
            
        except Exception as e:
            logger.error(f"Error calculating image similarity: {str(e)}")
            return 0.0

    def filter_profiles_by_preferences(self, preferences: Dict) -> List[Dict]:
        """Filter profiles based on extracted preferences"""
        try:
            filtered_profiles = []
            
            for profile in self.profiles_collection.find():
                matches_criteria = True
                
                # Basic filters
                if preferences.get('gender') and profile.get('gender') != preferences['gender']:
                    matches_criteria = False
                    continue
                
                # Only filter age if specific age range is provided
                if preferences.get('minAge') is not None and preferences.get('maxAge') is not None:
                    age = profile.get('age')
                    if not age or not (preferences['minAge'] <= age <= preferences['maxAge']):
                        matches_criteria = False
                        continue
                
                # Only filter location if specific location is provided
                if preferences.get('location') and preferences['location'].lower() not in ['any', 'anywhere']:
                    profile_location = profile.get('location', '').lower()
                    search_location = preferences['location'].lower()
                    
                    # Split locations into parts to match partial locations
                    search_parts = set(search_location.replace(',', ' ').split())
                    profile_parts = set(profile_location.replace(',', ' ').split())
                    
                    # Check if any major part matches (city or state)
                    if not any(part in profile_parts for part in search_parts):
                        matches_criteria = False
                        continue

                # Filter by height if specified
                if preferences.get('height_requirement'):
                    height = profile.get('height')
                    gender = profile.get('gender', '').lower()
                    
                    if height is not None and gender:
                        # Convert height to int if it's a string
                        try:
                            height = int(height) if isinstance(height, str) else height
                        except (ValueError, TypeError):
                            continue
                            
                        if preferences['height_requirement'] == 'tall':
                            # Men: tall >= 72 inches (6'0"), Women: tall >= 68 inches (5'8")
                            tall_threshold = 72 if gender == 'male' else 68
                            if height < (tall_threshold - 2):  # Allow 2 inches flexibility
                                matches_criteria = False
                                continue
                        elif preferences['height_requirement'] == 'short':
                            # Men: short <= 67 inches (5'7"), Women: short <= 63 inches (5'3")
                            short_threshold = 67 if gender == 'male' else 63
                            if height > short_threshold:
                                matches_criteria = False
                                continue
                
                # Filter by ethnicity if specified
                if preferences.get('ethnicity'):
                    profile_ethnicity = profile.get('ethnicity', '').lower()
                    search_ethnicity = preferences['ethnicity'].lower()
                    
                    if profile_ethnicity != search_ethnicity:
                        matches_criteria = False
                        continue
                
                if matches_criteria:
                    filtered_profiles.append(profile)
            
            logger.debug(f"Found {len(filtered_profiles)} profiles after filtering")
            return filtered_profiles
            
        except Exception as e:
            logger.error(f"Error filtering profiles: {str(e)}")
            return []

    def find_matching_profiles(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """Find profiles that match the given prompt."""
        try:
            # Extract preferences from prompt
            preferences = self.extract_preferences(prompt)
            logger.debug(f"Extracted preferences: {preferences}")
            
            # Get filtered profiles based on hierarchical matching
            filtered_profiles = self.filter_profiles_by_preferences(preferences)
            logger.debug(f"Found {len(filtered_profiles)} profiles after filtering")
            
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
                    occupation_score = self.calculate_text_similarity(occupation_text, prompt) if occupation_text else 0.0
                    
                    # Calculate general profile text similarity
                    profile_text = f"{bio_text} {interests}"
                    profile_score = self.calculate_text_similarity(profile_text, prompt)
                    
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
                    image_score = sum(image_scores) / len(image_scores) if image_scores else 0.0
                    
                    # Check if prompt contains physical trait descriptors
                    physical_traits = ['hair', 'eyes', 'tall', 'short', 'blonde', 'brunette', 'redhead', 'black hair']
                    has_physical_traits = any(trait in prompt.lower() for trait in physical_traits)
                    
                    # Adjust weights based on whether physical traits are mentioned
                    # If physical traits are mentioned, image similarity becomes more important
                    if has_physical_traits:
                        final_score = (0.2 * text_score) + (0.8 * image_score)
                    else:
                        final_score = (0.7 * text_score) + (0.3 * image_score)
                    
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
            logger.debug(f"Found {len(profile_scores)} matches after similarity calculation")
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
                        'age': match['profile'].get('age'),
                        'gender': match['profile'].get('gender', ''),
                        'location': match['profile'].get('location', ''),
                        'height': match['profile'].get('height', ''),
                        'ethnicity': match['profile'].get('ethnicity', ''),
                        'education': match['profile'].get('education', ''),
                        'sexualOrientation': match['profile'].get('sexualOrientation', ''),
                        'occupation': match['profile'].get('occupation', ''),
                        'aboutMe': match['profile'].get('aboutMe', ''),
                        'interests': match['profile'].get('interests', []),
                        'relationshipGoals': match['profile'].get('relationshipGoals', ''),
                        'photos': photos,
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
            # Create a comprehensive profile description
            profile_text = f"Name: {profile.get('name', '')}\n"
            profile_text += f"Occupation: {profile.get('occupation', '')}\n"
            profile_text += f"Location: {profile.get('location', '')}\n"
            profile_text += f"Interests: {', '.join(profile.get('interests', []))}\n"
            profile_text += f"Bio: {profile.get('bio', '')}"

            # Ask GPT to explain why this profile matches the user's prompt
            system_message = """You are an AI matchmaker. Your task is to explain profile matches based on the user's search criteria.
When the user's search includes specific physical traits (like hair color, eye color, height, etc.):
1. Focus primarily on evaluating those specific traits the user asked for
2. Only mention physical traits that were explicitly searched for
3. Base your response on the match score - a high score means the traits likely match, a low score means they likely don't
4. If unsure about a specific trait, don't mention it

For all matches, also consider personality, interests, and other relevant attributes that align with the search criteria."""
            
            user_message = f"User's search: '{prompt}'\nMatch score: {match_score}\n\nProfile:\n{profile_text}\n\nExplain why this might be a good match (in 1-2 sentences)."
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Add confidence level based on match score
            if match_score > 0.8:
                confidence = "Strong"
            elif match_score > 0.6:
                confidence = "Moderate"
            else:
                confidence = "Partial"
                
            return f"{confidence} match: {explanation}"
            
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

        # Debug: Print environment variables
        logger.info("Environment check:")
        logger.info(f"OPENAI_API_KEY exists: {'SECRET_API_KEY' in os.environ}")
        logger.info(f"MONGODB_URI exists: {'MONGODB_URI' in os.environ}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
