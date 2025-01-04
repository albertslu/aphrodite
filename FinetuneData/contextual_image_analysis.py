import torch
from PIL import Image
import clip
from typing import List, Dict
import json

class ContextualImageAnalyzer:
    def __init__(self):
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def analyze_image(self, image_path: str, prompts: List[str]) -> Dict:
        """
        Analyze an image against any set of textual descriptions/prompts
        Args:
            image_path: Path to the image
            prompts: List of text descriptions to check against the image
                    e.g. ["person playing sports", "someone who loves traveling",
                          "professional photographer", "fashion enthusiast"]
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text descriptions
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                # Get image and text features
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Create results dictionary
                results = {
                    prompt: float(score)  # Convert tensor to float
                    for prompt, score in zip(prompts, similarity[0])
                }
                
            return results
            
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None

    def batch_analyze_profile(self, image_paths: List[str], prompts: List[str]) -> Dict:
        """
        Analyze multiple images from a profile against given prompts
        """
        all_results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path, prompts)
            if result:
                all_results.append(result)
        
        # Aggregate results across all images
        return self._aggregate_profile_results(all_results)

    def _aggregate_profile_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple images
        """
        if not results:
            return {}
            
        # Calculate average scores across all images
        aggregated = {}
        for key in results[0].keys():
            scores = [result[key] for result in results]
            aggregated[key] = {
                'average_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'num_images': len(scores)
            }
        
        return aggregated

def main():
    analyzer = ContextualImageAnalyzer()
    
    # Example prompts - these can be dynamically generated from user preferences
    example_prompts = [
        "person who loves traveling",
        "athletic person playing sports",
        "someone at a social gathering with friends",
        "professional photographer with camera",
        "fashion enthusiast with stylish outfit",
        "person with pets",
        "someone who enjoys outdoor activities",
        "person in a professional setting"
    ]
    
    # Example usage
    profile_images = [
        "path/to/profile_image1.jpg",
        "path/to/profile_image2.jpg"
    ]
    
    profile_analysis = analyzer.batch_analyze_profile(profile_images, example_prompts)
    
    # Save results
    with open('profile_contextual_analysis.json', 'w') as f:
        json.dump(profile_analysis, f, indent=2)

if __name__ == "__main__":
    main()
