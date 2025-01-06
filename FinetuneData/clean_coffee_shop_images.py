import os
import shutil
import argparse
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.nn.functional import cosine_similarity
import clip
from tqdm import tqdm
import imagehash
from pathlib import Path

class ImageCleaner:
    def __init__(self, image_dir, removed_images_dir):
        self.image_dir = image_dir
        self.removed_images_dir = removed_images_dir
        os.makedirs(self.removed_images_dir, exist_ok=True)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Simplified prompts focusing on people presence
        self.text_prompts = [
            "a photo of a person",
            "a photo of a face",
            "a photo without any people",
            "an empty room",
            "just food or drinks",
            "a landscape or scenery"
        ]
        
        # Pre-encode text prompts
        text_tokens = clip.tokenize(self.text_prompts).to(self.device)
        self.text_features = self.model.encode_text(text_tokens)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
    def encode_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
            return image_features.float()
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def analyze_image(self, image_path):
        try:
            # Encode image
            image_features = self.encode_image(image_path)
            if image_features is None:
                print(f"Error encoding image: {image_path}")
                return False

            # Calculate similarity scores with text prompts
            text_features = self.text_features
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Print scores for debugging
            for i, (prompt, score) in enumerate(zip(self.text_prompts, similarity[0])):
                print(f"{prompt}: {score:.2f}")

            # Extract scores
            person_score = similarity[0][0].item()  # "a photo of a person"
            face_score = similarity[0][1].item()    # "a photo of a face"
            no_people_score = similarity[0][2].item()  # "a photo without any people"
            empty_score = similarity[0][3].item()    # "an empty room"
            food_score = similarity[0][4].item()     # "just food or drinks"
            landscape_score = similarity[0][5].item() # "a landscape or scenery"
            
            # Calculate person presence and absence scores
            person_presence = max(person_score, face_score)
            no_person_signals = max(no_people_score, empty_score, food_score, landscape_score)
            
            # Keep image if it has clear person presence and low negative signals
            if person_presence > 0.3 and no_person_signals < 0.5:
                return True
            else:
                print(f"Image {image_path} does not meet criteria (person: {person_presence:.2f}, no_person: {no_person_signals:.2f})")
                # Move to removed_images folder
                removed_path = os.path.join(self.removed_images_dir, os.path.basename(image_path))
                shutil.move(image_path, removed_path)
                return False

        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return False

    def process_images(self):
        # Get all image files
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(Path(self.image_dir).glob(ext))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        kept_count = 0
        removed_count = 0
        
        # Process images
        print("Processing images")
        for image_path in tqdm(image_files):
            if self.analyze_image(str(image_path)):
                kept_count += 1
            else:
                removed_count += 1
        
        print("\nProcessing complete!")
        print(f"Kept {kept_count} images in {self.image_dir}")
        print(f"Removed {removed_count} images to {self.removed_images_dir}")

def main():
    parser = argparse.ArgumentParser(description='Clean coffee shop images dataset')
    parser.add_argument('--image_dir', default='dating_app_dataset/coffee_shop_person', help='Directory containing the images')
    parser.add_argument('--removed_images_dir', default='dating_app_dataset/removed_images', help='Directory to move removed images')
    args = parser.parse_args()
    
    cleaner = ImageCleaner(args.image_dir, args.removed_images_dir)
    cleaner.process_images()

if __name__ == '__main__':
    main()
