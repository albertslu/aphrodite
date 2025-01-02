import face_recognition
import os
import numpy as np
import json
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def load_image_encodings(image_path):
    """
    Load and encode faces from an image.
    Returns a tuple of (encodings, error_message).
    If successful, error_message is None.
    """
    try:
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        # If no faces found, return None
        if not face_locations:
            return None, "No faces detected"
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # If multiple faces found, use the largest face (assumed to be the main subject)
        if len(face_encodings) > 1:
            # Calculate face areas
            areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in face_locations]
            largest_face_idx = np.argmax(areas)
            return face_encodings[largest_face_idx], None
        
        return face_encodings[0], None
    
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def process_image_batch(args):
    """Process a batch of images for parallel processing"""
    image_path, name = args
    encoding, error = load_image_encodings(image_path)
    return {
        "name": name,
        "path": image_path,
        "encoding": encoding.tolist() if encoding is not None else None,
        "error": error
    }

def calculate_face_similarity():
    # Directory containing face images
    face_dir = "FaceImages"
    
    # Get all image files and their names
    image_files = []
    for root, _, files in os.walk(face_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract name from filename (assuming format "name.jpg")
                name = os.path.splitext(file)[0]
                image_files.append((os.path.join(root, file), name))
    
    print(f"Found {len(image_files)} images")
    
    # Process images in parallel
    results = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Process images with progress bar
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            future_to_image = {executor.submit(process_image_batch, args): args for args in image_files}
            
            for future in concurrent.futures.as_completed(future_to_image):
                image_path, name = future_to_image[future]
                try:
                    result = future.result()
                    if result["encoding"] is not None:
                        results.append(result)
                    else:
                        errors.append({
                            "path": image_path,
                            "name": name,
                            "error": result["error"]
                        })
                except Exception as e:
                    errors.append({
                        "path": image_path,
                        "name": name,
                        "error": str(e)
                    })
                pbar.update(1)
    
    print(f"\nSuccessfully processed {len(results)} images")
    print(f"Failed to process {len(errors)} images")
    
    # Calculate similarity matrix
    print("\nCalculating similarity matrix...")
    similarity_results = []
    
    for i, img1 in enumerate(tqdm(results)):
        matches = []
        encoding1 = np.array(img1["encoding"])
        
        # Compare with all other faces
        distances = []
        for j, img2 in enumerate(results):
            if i != j:  # Don't compare with self
                encoding2 = np.array(img2["encoding"])
                distance = face_recognition.face_distance([encoding1], encoding2)[0]
                distances.append((j, distance))
        
        # Sort by similarity (lower distance = more similar)
        distances.sort(key=lambda x: x[1])
        
        # Get top 3 matches
        for j, distance in distances[:3]:
            similarity_score = 1 - distance  # Convert distance to similarity score
            matches.append({
                "name": results[j]["name"],
                "path": results[j]["path"],
                "similarity_score": float(similarity_score)
            })
        
        similarity_results.append({
            "source": {
                "name": img1["name"],
                "path": img1["path"]
            },
            "matches": matches
        })
    
    # Save results
    output = {
        "similarity_results": similarity_results,
        "errors": errors,
        "statistics": {
            "total_images": len(image_files),
            "successful": len(results),
            "failed": len(errors)
        }
    }
    
    with open("face_similarity_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to face_similarity_results.json")

if __name__ == "__main__":
    calculate_face_similarity()
