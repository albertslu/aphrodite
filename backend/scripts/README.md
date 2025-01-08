# Profile Seeding Script

This script helps populate the database with sample profiles including images for testing purposes.

## Setup

1. Create an `images` directory inside the `scripts` folder:
```bash
mkdir scripts/images
```

2. Add your profile images to the `scripts/images` directory. Name them according to the URLs in the profiles data (e.g., emma1.jpg, emma2.jpg, etc.)

3. Install dependencies if not already installed:
```bash
npm install mongoose fs path
```

## Running the Script

1. Make sure MongoDB is running locally

2. From the backend directory, run:
```bash
node scripts/seedProfiles.js
```

## Adding More Profiles

To add more profiles:

1. Add new profile objects to the `profiles` array in `seedProfiles.js`
2. Add corresponding images to the `scripts/images` directory
3. Make sure image filenames match the URLs in the profile data

## Image Guidelines

- Keep images under 5MB each
- Use JPG or PNG format
- Recommended dimensions: 800x800 pixels
- Name format: [profilename][number].jpg (e.g., emma1.jpg)
