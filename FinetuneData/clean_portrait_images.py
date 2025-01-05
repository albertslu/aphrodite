import os
import torch
import clip
from PIL import Image
import shutil
from tqdm import tqdm

def setup_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def check_is_person(image_path, model, preprocess, device):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Define text descriptions
        text_descriptions = [
            "a photo of a person",
            "a portrait of a person",
            "a full body photo of a person",
            "a photo of a human",
            "a landscape",
            "a building",
            "an object",
            "an animal"
        ]
        
        text_tokens = clip.tokenize(text_descriptions).to(device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get scores for person vs non-person categories
            person_score = sum(float(similarity[0][i]) for i in range(4))  # First 4 are person-related
            non_person_score = sum(float(similarity[0][i]) for i in range(4, 8))  # Last 4 are non-person
            
            return person_score > non_person_score and person_score > 0.6
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def main():
    # Setup paths
    portraits_dir = "dating_app_dataset/full_body_portrait"
    backup_dir = "dating_app_dataset/removed_images"
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Setup CLIP
    print("Loading CLIP model...")
    model, preprocess, device = setup_clip()
    
    # Get all images
    image_files = [f for f in os.listdir(portraits_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    removed_count = 0
    kept_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(portraits_dir, image_file)
        
        if check_is_person(image_path, model, preprocess, device):
            kept_count += 1
        else:
            # Move non-person images to backup directory
            shutil.move(image_path, os.path.join(backup_dir, image_file))
            removed_count += 1
            print(f"Removed: {image_file}")
    
    print(f"\nProcessing complete!")
    print(f"Kept {kept_count} images")
    print(f"Removed {removed_count} images")
    print(f"Removed images are backed up in: {backup_dir}")

if __name__ == "__main__":
    main()
