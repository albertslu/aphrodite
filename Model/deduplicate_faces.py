import os
import shutil
from collections import defaultdict

def deduplicate_faces(directory):
    # Dictionary to store first image for each name
    name_to_image = {}
    
    # Get all image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} total images")
    
    # Group files by person's name (everything before first underscore)
    for image_file in image_files:
        name = image_file.split('_')[0]
        if name not in name_to_image:
            name_to_image[name] = image_file
    
    print(f"Found {len(name_to_image)} unique people")
    
    # Create backup directory
    backup_dir = os.path.join(os.path.dirname(directory), "AllFaces_Backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Move all files to backup first
    print("Creating backup...")
    for image_file in image_files:
        src = os.path.join(directory, image_file)
        dst = os.path.join(backup_dir, image_file)
        shutil.move(src, dst)
    
    # Move back only one image per person
    print("Moving back one image per person...")
    for image_file in name_to_image.values():
        src = os.path.join(backup_dir, image_file)
        dst = os.path.join(directory, image_file)
        shutil.copy2(src, dst)
    
    print("Deduplication complete!")
    print(f"Original image count: {len(image_files)}")
    print(f"Deduplicated image count: {len(name_to_image)}")
    print(f"Backup of all original images saved in: {backup_dir}")

if __name__ == "__main__":
    directory = "AllFaces"
    deduplicate_faces(directory)
