import os
import shutil
from tqdm import tqdm

def collect_images():
    # Source directory with categorized images
    source_dir = "FaceImages"
    
    # Create target directory for all images
    target_dir = "AllFaces"
    os.makedirs(target_dir, exist_ok=True)
    
    # Counter for duplicate names
    name_counter = {}
    
    # Get all image files
    total_images = 0
    for root, _, files in os.walk(source_dir):
        total_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Copy files with progress bar
    copied = 0
    errors = []
    
    with tqdm(total=total_images, desc="Copying images") as pbar:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # Get source path
                        source_path = os.path.join(root, file)
                        
                        # Create unique target filename
                        base_name = os.path.splitext(file)[0]
                        extension = os.path.splitext(file)[1]
                        
                        # If this name was already used, add a counter
                        if base_name in name_counter:
                            name_counter[base_name] += 1
                            target_name = f"{base_name}_{name_counter[base_name]}{extension}"
                        else:
                            name_counter[base_name] = 0
                            target_name = file
                        
                        target_path = os.path.join(target_dir, target_name)
                        
                        # Copy the file
                        shutil.copy2(source_path, target_path)
                        copied += 1
                        
                    except Exception as e:
                        errors.append({
                            "file": file,
                            "error": str(e)
                        })
                    
                    pbar.update(1)
    
    print(f"\nSuccessfully copied {copied} images to {target_dir}")
    if errors:
        print(f"Failed to copy {len(errors)} images:")
        for error in errors:
            print(f"  {error['file']}: {error['error']}")

if __name__ == "__main__":
    collect_images()
