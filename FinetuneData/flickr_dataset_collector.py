import flickrapi
import requests
import os
from typing import List, Dict
import json
from datetime import datetime
from tqdm import tqdm
import time
import random

class FlickrDatasetCollector:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Flickr API client
        Get API credentials from: https://www.flickr.com/services/apps/create/
        """
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
        self.search_categories = {
            "portrait": [
                "portrait photography", "professional portrait", 
                "outdoor portrait", "natural light portrait"
            ],
            "full_body": [
                "full body portrait", "fashion photography",
                "model photography", "street style photography"
            ],
            "travel": [
                "travel photography people", "tourist photo",
                "vacation selfie", "travel portrait",
                "landmark tourist", "beach vacation people"
            ],
            "social": [
                "friends group photo", "social gathering",
                "party people", "friends hanging out",
                "group celebration", "friends dinner"
            ],
            "activities": [
                "hiking people", "yoga outdoor",
                "beach sports", "rock climbing person",
                "cycling people", "running athletes"
            ],
            "lifestyle": [
                "coffee shop person", "restaurant dining",
                "cooking kitchen", "shopping fashion",
                "concert crowd", "museum visitor"
            ],
            "pets": [
                "person with dog", "cat owner",
                "pet photography", "dog walking"
            ]
        }
        
        # License types we want (Creative Commons)
        self.licenses = [
            '4',  # CC BY
            '5',  # CC BY-SA
            '7',  # No known copyright restrictions
        ]

    def search_photos(self, query: str, max_photos: int = 100) -> List[Dict]:
        """
        Search for photos matching query with specific criteria
        """
        try:
            photos = []
            page = 1
            
            # Parameters for high-quality, recent photos
            search_params = {
                'text': query,
                'license': ','.join(self.licenses),
                'content_type': 1,  # Photos only
                'media': 'photos',
                'sort': 'relevance',
                'privacy_filter': 1,  # Public photos only
                'safe_search': 1,  # Safe content only
                'per_page': 100,
                'extras': 'url_l,license,owner_name,date_taken,tags'
            }

            while len(photos) < max_photos:
                search_params['page'] = page
                results = self.flickr.photos.search(**search_params)
                
                if not results['photos']['photo']:
                    break
                
                for photo in results['photos']['photo']:
                    if 'url_l' in photo:  # Only get photos with large size available
                        photos.append({
                            'id': photo['id'],
                            'title': photo['title'],
                            'url': photo['url_l'],
                            'owner': photo['owner'],
                            'owner_name': photo.get('owner_name', ''),
                            'date_taken': photo.get('date_taken', ''),
                            'tags': photo.get('tags', ''),
                            'license': photo['license'],
                            'category': query
                        })
                        
                        if len(photos) >= max_photos:
                            break
                
                page += 1
                time.sleep(0.5)  # Respect rate limits
            
            return photos

        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return []

    def download_photo(self, photo: Dict, output_dir: str) -> bool:
        """
        Download a single photo and save its metadata
        """
        try:
            # Create category subdirectory
            category_dir = os.path.join(output_dir, photo['category'].replace(' ', '_'))
            os.makedirs(category_dir, exist_ok=True)
            
            # Generate filename
            filename = f"{photo['id']}_{photo['owner']}"
            image_path = os.path.join(category_dir, f"{filename}.jpg")
            meta_path = os.path.join(category_dir, f"{filename}.json")
            
            # Download image if it doesn't exist
            if not os.path.exists(image_path):
                response = requests.get(photo['url'])
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Save metadata
                    with open(meta_path, 'w') as f:
                        json.dump(photo, f, indent=2)
                    
                    return True
            
            return False

        except Exception as e:
            print(f"Error downloading photo {photo['id']}: {str(e)}")
            return False

    def collect_dataset(self, output_dir: str, photos_per_category: int = 100):
        """
        Collect dataset from all categories
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset_stats = {
            'total_photos': 0,
            'categories': {}
        }

        for category, queries in tqdm(self.search_categories.items()):
            category_photos = []
            photos_needed = photos_per_category
            
            print(f"\nCollecting {photos_needed} photos for category: {category}")
            
            # Try each query until we have enough photos
            for query in queries:
                if photos_needed <= 0:
                    break
                    
                print(f"Searching for: {query}")
                photos = self.search_photos(query, photos_needed)
                
                # Download each photo
                for photo in tqdm(photos):
                    if self.download_photo(photo, output_dir):
                        category_photos.append(photo['id'])
                        photos_needed -= 1
                        
                    if photos_needed <= 0:
                        break
                        
                time.sleep(1)  # Respect rate limits
            
            # Update stats
            dataset_stats['categories'][category] = {
                'photos_collected': len(category_photos),
                'photo_ids': category_photos
            }
            dataset_stats['total_photos'] += len(category_photos)

        # Save dataset statistics
        stats_file = os.path.join(output_dir, 'dataset_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"\nDataset collection completed!")
        print(f"Total photos collected: {dataset_stats['total_photos']}")
        for category, stats in dataset_stats['categories'].items():
            print(f"{category}: {stats['photos_collected']} photos")

def main():
    # Hardcode API credentials
    api_key = '93c73cb77a77eaaa1d53f121aaae07f7'
    api_secret = 'f29e176df48bd8c9'
    
    collector = FlickrDatasetCollector(api_key, api_secret)
    
    # Set output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'dating_app_dataset')
    
    # Collect dataset
    collector.collect_dataset(output_dir, photos_per_category=100)

if __name__ == "__main__":
    main()
