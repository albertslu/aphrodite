# Profile Seeding Script

This script helps populate the database with sample profiles from the extracted OkCupid dataset.

## Required Images

Based on the profiles in the dataset, we need the following images in the `scripts/images` directory:

1. Profile 1 (22-year-old Asian/White male):
   - male_22_asian_1.jpg
   - male_22_asian_2.jpg
   (Find stock photos of a young Asian male with "a little extra" body type)

2. Profile 2 (35-year-old White male):
   - male_35_white_1.jpg
   - male_35_white_2.jpg
   (Find stock photos of an average-build white male chef/hospitality worker)

3. Profile 3 (32-year-old Female):
   - female_32_white_1.jpg
   - female_32_white_2.jpg
   (Find appropriate stock photos matching the profile)

## Image Guidelines

1. Use stock photos or AI-generated images that match these characteristics:
   - Age
   - Gender
   - Ethnicity
   - Body type (when specified)
   - Professional context (when specified)

2. Image Requirements:
   - JPG format
   - Max size: 5MB per image
   - Recommended dimensions: 800x800 pixels
   - Professional or high-quality casual photos
   - No inappropriate content

3. Naming Convention:
   ```
   [gender]_[age]_[ethnicity]_[number].jpg
   ```
   Example: male_22_asian_1.jpg

## Running the Script

1. Add the required images to `backend/scripts/images/`
2. From the backend directory, run:
   ```bash
   node scripts/seedProfiles.js
   ```

The script will:
1. Clear existing profiles
2. Copy images to the uploads directory
3. Create new profiles with the image references
4. Map the OkCupid data format to our schema
