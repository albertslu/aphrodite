import base64
import requests
import json
from typing import List, Dict
from PIL import Image
import io
import os
from datetime import datetime

class GPT4VisionAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the GPT-4 Vision analyzer
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with Image.open(image_path) as img:
            # Resize if image is too large (GPT-4 Vision has 16MB limit)
            max_size = (1024, 1024)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze_image(self, image_path: str, prompts: List[str]) -> Dict:
        """
        Analyze an image using GPT-4 Vision API
        Args:
            image_path: Path to the image
            prompts: List of aspects to analyze in the image
        """
        try:
            base64_image = self._encode_image(image_path)
            
            # Create analysis prompt
            analysis_prompt = (
                "Analyze this image in the context of a dating profile. "
                "For each of the following aspects, rate how well the image demonstrates them on a scale of 0-100 "
                "and provide a brief explanation:\n\n"
                + "\n".join(prompts)
                + "\n\nProvide your response in JSON format with the following structure for each aspect:"
                "{'aspect_name': {'score': number, 'explanation': 'brief explanation'}}"
            )

            # Prepare the API request
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
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

            # Make API request
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            analysis = json.loads(result['choices'][0]['message']['content'])
            
            return analysis

        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None

    def batch_analyze_profile(self, image_paths: List[str], prompts: List[str]) -> Dict:
        """
        Analyze multiple images from a profile
        """
        all_results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path, prompts)
            if result:
                all_results.append(result)

        return self._aggregate_profile_results(all_results)

    def _aggregate_profile_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple images with explanations
        """
        if not results:
            return {}

        aggregated = {}
        for aspect in results[0].keys():
            scores = [result[aspect]['score'] for result in results]
            explanations = [result[aspect]['explanation'] for result in results]
            
            aggregated[aspect] = {
                'average_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'num_images': len(scores),
                'explanations': explanations,
                'summary': self._generate_aspect_summary(aspect, scores, explanations)
            }

        return aggregated

    def _generate_aspect_summary(self, aspect: str, scores: List[float], explanations: List[str]) -> str:
        """
        Generate a summary for an aspect across multiple images
        """
        avg_score = sum(scores) / len(scores)
        if avg_score >= 80:
            confidence = "strongly"
        elif avg_score >= 60:
            confidence = "moderately"
        elif avg_score >= 40:
            confidence = "somewhat"
        else:
            confidence = "minimally"

        return f"Profile {confidence} demonstrates {aspect} with an average score of {avg_score:.1f}"

def main():
    # Load API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    analyzer = GPT4VisionAnalyzer(api_key)

    # Example prompts that GPT-4 Vision can analyze
    example_prompts = [
        "Athletic lifestyle and fitness level",
        "Travel enthusiasm and adventurousness",
        "Social nature and friendliness",
        "Professional appearance and career focus",
        "Fashion sense and style",
        "Outdoor activity involvement",
        "Artistic or creative interests",
        "Pet ownership or animal lover",
        "Hobbies and interests visible in the image",
        "Overall profile photo quality and appeal"
    ]

    # Example usage
    profile_images = [
        "path/to/profile_image1.jpg",
        "path/to/profile_image2.jpg"
    ]

    profile_analysis = analyzer.batch_analyze_profile(profile_images, example_prompts)

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'profile_gpt4_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(profile_analysis, f, indent=2)

if __name__ == "__main__":
    main()
