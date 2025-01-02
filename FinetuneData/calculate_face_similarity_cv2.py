import cv2
import os
import numpy as np
import json
from tqdm import tqdm
import concurrent.futures

def init_face_detector():
    # Load YuNet face detector
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),  # Input size
        0.9,         # Score threshold
        0.3,         # NMS threshold
        5000         # Top k
    )
    
    return face_detector

def extract_face_features(image_path, face_detector):
    """Extract face features from an image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, f"Could not read image: {image_path}"
        
        # Set input size for detector
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = face_detector.detect(image)
        
        if faces is None:
            return None, "No faces detected"
        
        # Get the face with highest confidence
        best_face = faces[np.argmax([face[-1] for face in faces])]
        x, y, w, h = map(int, best_face[:4])
        
        # Extract the face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Resize face to standard size
        face_roi = cv2.resize(face_roi, (112, 112))
        
        # Convert to grayscale and flatten for feature vector
        face_features = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).flatten()
        
        # Normalize features
        face_features = face_features / np.linalg.norm(face_features)
        
        return face_features, None
        
    except Exception as e:
        return None, str(e)

def process_image_batch(args):
    """Process a batch of images"""
    image_path, name, face_detector = args
    features, error = extract_face_features(image_path, face_detector)
    return {
        "name": name,
        "path": image_path,
        "features": features.tolist() if features is not None else None,
        "error": error
    }

def calculate_face_similarity():
    # Initialize face detection model
    face_detector = init_face_detector()
    
    # Directory containing face images
    face_dir = "FaceImages"
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(face_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(file)[0]
                image_files.append((os.path.join(root, file), name))
    
    print(f"Found {len(image_files)} images")
    
    # Process images in parallel
    results = []
    errors = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Add face detector to each task
        tasks = [(path, name, face_detector) for path, name in image_files]
        
        # Process images with progress bar
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            future_to_image = {executor.submit(process_image_batch, args): args for args in tasks}
            
            for future in concurrent.futures.as_completed(future_to_image):
                image_path, name, _ = future_to_image[future]
                try:
                    result = future.result()
                    if result["features"] is not None:
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
        features1 = np.array(img1["features"])
        
        # Compare with all other faces
        distances = []
        for j, img2 in enumerate(results):
            if i != j:  # Don't compare with self
                features2 = np.array(img2["features"])
                # Calculate cosine similarity
                similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
                distances.append((j, similarity))
        
        # Sort by similarity (higher is better)
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 matches
        for j, similarity in distances[:3]:
            matches.append({
                "name": results[j]["name"],
                "path": results[j]["path"],
                "similarity_score": float(similarity)
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
