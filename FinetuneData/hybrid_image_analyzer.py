import torch
import clip
from PIL import Image
import base64
import requests
import json
from typing import List, Dict, Union, Optional
import io
import os
from datetime import datetime

class ProfileMatcher:
    def __init__(self, openai_api_key: str):
        """
        Initialize both CLIP and GPT-4 Vision analyzers
        """
        # Initialize CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # GPT-4 Vision settings
        self.api_key = openai_api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

    def verify_physical_traits(self, image_path: str, stated_traits: Dict[str, str]) -> Dict[str, float]:
        """
        Verify stated physical traits using CLIP
        Args:
            image_path: Path to profile image
            stated_traits: Dict of traits from profile, e.g., {"body_type": "athletic", "height": "tall"}
        Returns:
            Dict of verification scores for each trait
        """
        verification_prompts = []
        for trait_type, value in stated_traits.items():
            if trait_type == "body_type":
                verification_prompts.append(f"a person with {value} build")
            elif trait_type == "height":
                verification_prompts.append(f"a {value} person")
            # Add more trait types as needed

        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text descriptions
            text_tokens = clip.tokenize(verification_prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Create verification results
                results = {
                    trait: float(score)
                    for trait, score in zip(stated_traits.keys(), similarity[0])
                }
                
            return results
            
        except Exception as e:
            print(f"Error in physical trait verification for {image_path}: {str(e)}")
            return {}

    def analyze_lifestyle_traits(self, image_path: str, trait_prompts: List[str]) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Analyze lifestyle and personality traits using GPT-4 Vision
        Args:
            image_path: Path to profile image
            trait_prompts: List of traits to analyze, e.g., ["travel enthusiasm", "social activity"]
        """
        try:
            base64_image = self._encode_image(image_path)
            
            analysis_prompt = (
                "Analyze this dating profile photo for the following traits. "
                "For each trait, provide a score (0-100) and a brief explanation:\n\n"
                + "\n".join(trait_prompts)
                + "\n\nFormat your response as JSON: "
                "{'trait': {'score': number, 'explanation': 'brief explanation'}}"
            )

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            return json.loads(response.json()['choices'][0]['message']['content'])

        except Exception as e:
            print(f"Error in lifestyle trait analysis for {image_path}: {str(e)}")
            return {}

    def calculate_profile_match(self, 
                              profile_data: Dict,
                              preference_data: Dict,
                              profile_images: List[str]) -> Dict:
        """
        Calculate overall profile match combining stated data and image analysis
        Args:
            profile_data: Dict containing profile information
            preference_data: Dict containing preference criteria
            profile_images: List of paths to profile images
        """
        match_results = {
            "demographic_match": self._calculate_demographic_match(profile_data, preference_data),
            "trait_verification": {},
            "lifestyle_analysis": {},
            "overall_score": 0.0
        }

        # Verify physical traits from images
        physical_verifications = []
        for image_path in profile_images:
            verification = self.verify_physical_traits(
                image_path,
                {
                    "body_type": profile_data.get("body_type", ""),
                    "height": profile_data.get("height", "")
                }
            )
            if verification:
                physical_verifications.append(verification)

        # Aggregate physical trait verifications
        if physical_verifications:
            match_results["trait_verification"] = {
                trait: sum(v[trait] for v in physical_verifications) / len(physical_verifications)
                for trait in physical_verifications[0].keys()
            }

        # Analyze lifestyle traits
        lifestyle_traits = [
            "travel enthusiasm",
            "social activity level",
            "outdoor adventure interest",
            "fitness lifestyle",
            "artistic/creative interests"
        ]

        lifestyle_analyses = []
        for image_path in profile_images:
            analysis = self.analyze_lifestyle_traits(image_path, lifestyle_traits)
            if analysis:
                lifestyle_analyses.append(analysis)

        # Aggregate lifestyle analyses
        if lifestyle_analyses:
            match_results["lifestyle_analysis"] = {
                trait: {
                    "score": sum(a[trait]["score"] for a in lifestyle_analyses) / len(lifestyle_analyses),
                    "explanations": [a[trait]["explanation"] for a in lifestyle_analyses]
                }
                for trait in lifestyle_traits
                if all(trait in a for a in lifestyle_analyses)
            }

        # Calculate overall match score
        match_results["overall_score"] = self._calculate_overall_score(match_results)
        
        return match_results

    def _calculate_demographic_match(self, profile: Dict, preferences: Dict) -> Dict:
        """Calculate match score for demographic criteria"""
        scores = {}
        
        # Age match
        if "age" in profile and "age_range" in preferences:
            age = profile["age"]
            min_age, max_age = preferences["age_range"]
            if min_age <= age <= max_age:
                scores["age"] = 1.0
            else:
                distance = min(abs(age - min_age), abs(age - max_age))
                scores["age"] = max(0, 1 - (distance / 10))  # Decay by distance

        # Gender match
        if "gender" in profile and "preferred_gender" in preferences:
            scores["gender"] = 1.0 if profile["gender"] == preferences["preferred_gender"] else 0.0

        # Location match
        if "location" in profile and "max_distance" in preferences:
            distance = self._calculate_distance(profile["location"], preferences["location"])
            scores["location"] = max(0, 1 - (distance / preferences["max_distance"]))

        return scores

    def _calculate_overall_score(self, match_results: Dict) -> float:
        """
        Calculate overall match score with weighted components
        """
        weights = {
            "demographic_match": 0.4,
            "trait_verification": 0.3,
            "lifestyle_analysis": 0.3
        }

        scores = []
        
        # Demographic score (weighted average of demographic components)
        if match_results["demographic_match"]:
            demographic_score = sum(match_results["demographic_match"].values()) / len(match_results["demographic_match"])
            scores.append(weights["demographic_match"] * demographic_score)

        # Trait verification score
        if match_results["trait_verification"]:
            verification_score = sum(match_results["trait_verification"].values()) / len(match_results["trait_verification"])
            scores.append(weights["trait_verification"] * verification_score)

        # Lifestyle score
        if match_results["lifestyle_analysis"]:
            lifestyle_score = sum(trait["score"]/100 for trait in match_results["lifestyle_analysis"].values()) / len(match_results["lifestyle_analysis"])
            scores.append(weights["lifestyle_analysis"] * lifestyle_score)

        return sum(scores) / sum(weights[k] for k, v in match_results.items() if v)

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string with size checking"""
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @staticmethod
    def _calculate_distance(loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
        """Calculate distance between two locations (simplified)"""
        # Implement your preferred distance calculation here
        # This is a placeholder that assumes locations are dictionaries with 'lat' and 'lon' keys
        return ((loc1['lat'] - loc2['lat'])**2 + (loc1['lon'] - loc2['lon'])**2)**0.5

    def match_flickr_images(self, flickr_dir: str, prompts_file: str, output_file: str = "flickr_matches.json") -> None:
        """
        Match Flickr images with prompts using CLIP model
        Args:
            flickr_dir: Directory containing Flickr images
            prompts_file: Path to JSONL file containing prompts
            output_file: Path to save the matches
        """
        # Load prompts from JSONL
        prompts = []
        with open(prompts_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data['prompt'])
        
        # Get all image paths
        image_paths = []
        for root, _, files in os.walk(flickr_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images and {len(prompts)} prompts")
        
        # Process images in batches
        batch_size = 32
        all_matches = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess images
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(self.preprocess(image))
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # Stack images into a batch tensor
            image_input = torch.stack(batch_images).to(self.device)
            
            # Process text prompts
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            # Calculate similarities
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T)
                
                # Find top matches for each image
                for img_idx, img_path in enumerate(valid_paths):
                    scores, indices = similarity[img_idx].topk(5)  # Get top 5 matches
                    
                    matches = {
                        'image_path': img_path,
                        'matches': [
                            {
                                'prompt': prompts[idx],
                                'score': float(score)
                            }
                            for score, idx in zip(scores.cpu().numpy(), indices.cpu().numpy())
                        ]
                    }
                    all_matches.append(matches)
            
            print(f"Processed {len(all_matches)} images so far...")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_matches, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Print some example matches
        print("\nExample matches:")
        for match in all_matches[:3]:
            print(f"\nImage: {os.path.basename(match['image_path'])}")
            print("Top matching prompts:")
            for prompt_match in match['matches']:
                print(f"- Score: {prompt_match['score']:.3f}")
                print(f"  Prompt: {prompt_match['prompt']}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profile Matching System')
    parser.add_argument('--openai_key', type=str, help='OpenAI API Key')
    parser.add_argument('--mode', type=str, choices=['verify', 'analyze', 'match_flickr'], 
                      default='verify', help='Operation mode')
    parser.add_argument('--flickr_dir', type=str, help='Directory containing Flickr images')
    parser.add_argument('--prompts_file', type=str, help='Path to prompts JSONL file')
    parser.add_argument('--output_file', type=str, default='flickr_matches.json',
                      help='Output file for Flickr matches')
    
    args = parser.parse_args()
    
    matcher = ProfileMatcher(args.openai_key)
    
    if args.mode == 'match_flickr':
        if not args.flickr_dir or not args.prompts_file:
            print("Error: flickr_dir and prompts_file are required for match_flickr mode")
            return
        matcher.match_flickr_images(args.flickr_dir, args.prompts_file, args.output_file)
    else:
        # Original verification and analysis code here
        pass

if __name__ == "__main__":
    main()
