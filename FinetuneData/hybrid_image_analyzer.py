import torch
import clip
from PIL import Image
import base64
import requests
import json
from typing import List, Dict, Tuple
import io
import os
from datetime import datetime

class HybridImageAnalyzer:
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

    def analyze_physical_traits(self, image_path: str, trait_prompts: List[str]) -> Dict:
        """
        Use CLIP for analyzing physical and demographic traits
        Examples of trait_prompts:
        - "a tall person"
        - "a person with athletic build"
        - "a person with curly hair"
        - "a plus-size person"
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text descriptions
            text_tokens = clip.tokenize(trait_prompts).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Create results dictionary
                results = {
                    prompt: float(score)
                    for prompt, score in zip(trait_prompts, similarity[0])
                }
                
            return {"physical_traits": results}
            
        except Exception as e:
            print(f"Error in CLIP analysis for {image_path}: {str(e)}")
            return None

    def analyze_contextual_traits(self, image_path: str, context_prompts: List[str]) -> Dict:
        """
        Use GPT-4 Vision for analyzing contextual and lifestyle traits
        Examples of context_prompts:
        - "Evidence of love for traveling"
        - "Signs of being a foodie or cooking enthusiast"
        - "Indicators of being social and outgoing"
        - "Fashion sense and style preferences"
        """
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Create analysis prompt
            analysis_prompt = (
                "Analyze this dating profile photo for the following lifestyle and contextual traits. "
                "For each trait, provide a score (0-100) and a brief explanation:\n\n"
                + "\n".join(context_prompts)
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
            
            analysis = json.loads(response.json()['choices'][0]['message']['content'])
            return {"contextual_traits": analysis}

        except Exception as e:
            print(f"Error in GPT-4 Vision analysis for {image_path}: {str(e)}")
            return None

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string with size checking"""
        with Image.open(image_path) as img:
            # Resize if image is too large
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze_profile(self, image_paths: List[str], 
                       physical_prompts: List[str], 
                       contextual_prompts: List[str]) -> Dict:
        """
        Analyze multiple profile images using both CLIP and GPT-4 Vision
        """
        all_results = []
        for image_path in image_paths:
            physical_results = self.analyze_physical_traits(image_path, physical_prompts)
            contextual_results = self.analyze_contextual_traits(image_path, contextual_prompts)
            
            if physical_results and contextual_results:
                all_results.append({
                    "image": image_path,
                    **physical_results,
                    **contextual_results
                })

        return self._aggregate_profile_results(all_results)

    def _aggregate_profile_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple images
        """
        if not results:
            return {}

        aggregated = {
            "physical_traits": {},
            "contextual_traits": {},
            "overall_summary": {}
        }

        # Aggregate physical traits (CLIP results)
        for trait in results[0]["physical_traits"]:
            scores = [r["physical_traits"][trait] for r in results]
            aggregated["physical_traits"][trait] = {
                "average_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "num_images": len(scores)
            }

        # Aggregate contextual traits (GPT-4 Vision results)
        for trait in results[0]["contextual_traits"]:
            scores = [r["contextual_traits"][trait]["score"] for r in results]
            explanations = [r["contextual_traits"][trait]["explanation"] for r in results]
            
            aggregated["contextual_traits"][trait] = {
                "average_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "num_images": len(scores),
                "explanations": explanations
            }

        return aggregated

def main():
    # Load API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    analyzer = HybridImageAnalyzer(api_key)

    # Example prompts for each model
    physical_prompts = [
        "a tall person",
        "a person with athletic build",
        "a person with slim build",
        "a person with curly hair",
        "a plus-size person",
        "a person with tattoos",
        "a person with glasses"
    ]

    contextual_prompts = [
        "Evidence of being a travel enthusiast",
        "Signs of being social and outgoing",
        "Indicators of being into fitness and healthy lifestyle",
        "Fashion sense and style preferences",
        "Evidence of outdoor adventure activities",
        "Signs of being a foodie or cooking enthusiast",
        "Indicators of creative or artistic interests"
    ]

    # Example usage
    profile_images = [
        "path/to/profile_image1.jpg",
        "path/to/profile_image2.jpg"
    ]

    profile_analysis = analyzer.analyze_profile(
        profile_images, 
        physical_prompts,
        contextual_prompts
    )

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'profile_hybrid_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(profile_analysis, f, indent=2)

if __name__ == "__main__":
    main()
