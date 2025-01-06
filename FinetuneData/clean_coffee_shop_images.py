import os
import base64
import json
from PIL import Image
import shutil
from tqdm import tqdm
import imagehash
from typing import Dict, List, Tuple
import numpy as np
from openai import OpenAI
import time

class ImageCleaner:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str) -> Dict:
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and tell me:\n1. Estimated age range of any person in the image\n2. Is this a good quality photo suitable for a dating app?\n3. Is this taken in a coffee shop setting?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Error analyzing {image_path}: {str(e)}")
            return None

    def compute_image_hash(self, image_path: str) -> str:
        """Compute perceptual hash of image for duplicate detection"""
        try:
            with Image.open(image_path) as img:
                return str(imagehash.average_hash(img))
        except Exception as e:
            print(f"Error computing hash for {image_path}: {str(e)}")
            return None

    def find_duplicates(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Find duplicate images using perceptual hashing"""
        hash_dict = {}
        for img_path in tqdm(image_paths, desc="Computing image hashes"):
            img_hash = self.compute_image_hash(img_path)
            if img_hash:
                if img_hash in hash_dict:
                    hash_dict[img_hash].append(img_path)
                else:
                    hash_dict[img_hash] = [img_path]
        
        return {h: paths for h, paths in hash_dict.items() if len(paths) > 1}

    def clean_coffee_shop_images(self, input_dir: str, removed_dir: str = "dating_app_dataset/removed_images"):
        """Clean coffee shop images based on age and duplicates"""
        os.makedirs(removed_dir, exist_ok=True)

        # Get all images
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        print(f"Found {len(image_files)} images to process")

        # First, find duplicates
        image_paths = [os.path.join(input_dir, f) for f in image_files]
        duplicates = self.find_duplicates(image_paths)
        
        # Keep track of processed images and their analysis
        processed_images = {}
        removed_count = 0
        kept_count = 0

        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            
            # Skip if this is a duplicate and we've already processed its group
            skip = False
            for dup_group in duplicates.values():
                if image_path in dup_group and any(p in processed_images for p in dup_group):
                    skip = True
                    shutil.move(image_path, os.path.join(removed_dir, f"duplicate_{image_file}"))
                    removed_count += 1
                    print(f"\nRemoved duplicate: {image_file}")
                    break
            
            if skip:
                continue

            # Analyze image
            analysis = self.analyze_image(image_path)
            if not analysis:
                continue

            processed_images[image_path] = analysis
            
            # Check if image should be kept
            should_keep = (
                analysis['suitable_for_dating_app'] and
                analysis['age_range']['min'] >= 18 and
                analysis['age_range']['max'] <= 60 and
                analysis['image_quality'] != 'low'
            )

            if should_keep:
                # Keep the image where it is
                kept_count += 1
            else:
                # Move to removed_images folder
                shutil.move(image_path, os.path.join(removed_dir, image_file))
                removed_count += 1
                print(f"\nRemoved {image_file}:")
                print(f"Age range: {analysis['age_range']}")
                print(f"Quality: {analysis['image_quality']}")
                print(f"Reasons: {', '.join(analysis['reasons'])}")
                
            # Add a small delay to avoid rate limits
            time.sleep(0.5)

        print(f"\nProcessing complete!")
        print(f"Kept {kept_count} images in {input_dir}")
        print(f"Removed {removed_count} images to {removed_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clean coffee shop person images')
    parser.add_argument('--openai_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--input_dir', type=str, default='dating_app_dataset/coffee_shop_person',
                      help='Input directory containing images')
    parser.add_argument('--removed_dir', type=str, default='dating_app_dataset/removed_images',
                      help='Directory for removed images')
    
    args = parser.parse_args()
    
    cleaner = ImageCleaner(args.openai_key)
    cleaner.clean_coffee_shop_images(args.input_dir, args.removed_dir)

if __name__ == "__main__":
    main()
