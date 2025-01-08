const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const Profile = require('../models/Profile');

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/profile_matching', {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

// Convert height from inches to readable format
const formatHeight = (inches) => {
    const feet = Math.floor(inches / 12);
    const remainingInches = inches % 12;
    return `${feet}'${remainingInches}"`;
};

// Convert profile data to our schema format
const formatProfileData = (rawProfile) => {
    return {
        name: `User${Math.floor(Math.random() * 10000)}`, // Generate random username since original data doesn't have names
        gender: rawProfile.sex === 'm' ? 'male' : 'female',
        sexualOrientation: rawProfile.orientation,
        age: rawProfile.age,
        height: formatHeight(rawProfile.height),
        ethnicity: rawProfile.ethnicity.split(',')[0].trim(), // Take first ethnicity if multiple
        occupation: rawProfile.job,
        location: rawProfile.location,
        education: rawProfile.education,
        aboutMe: rawProfile.essay0 || 'No description provided',
        interests: rawProfile.essay2 || 'No interests provided',
        relationshipGoals: rawProfile.essay9 || 'Looking for meaningful connections',
        partnerPreferences: rawProfile.essay0?.split('about you:')[1] || 'Open to meeting new people',
        photos: [
            {
                url: `/uploads/${rawProfile.sex === 'm' ? 'male' : 'female'}_${rawProfile.age}_${rawProfile.ethnicity.split(',')[0].trim()}_1.jpg`,
                caption: "Profile photo",
                order: 1
            },
            {
                url: `/uploads/${rawProfile.sex === 'm' ? 'male' : 'female'}_${rawProfile.age}_${rawProfile.ethnicity.split(',')[0].trim()}_2.jpg`,
                caption: "Additional photo",
                order: 2
            }
        ]
    };
};

// Read the original profile data
const rawProfiles = JSON.parse(fs.readFileSync(path.join(__dirname, '../../Model/extracted_10_profiles.json')));

// Convert profiles to our format
const profiles = rawProfiles.slice(0, 3).map(formatProfileData); // Only use first 3 profiles for now

// Function to copy images
const copyImages = async () => {
    const sourceDir = path.join(__dirname, 'images');
    const targetDir = path.join(__dirname, '../uploads');

    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
    }

    // Copy each image
    const images = fs.readdirSync(sourceDir);
    for (const image of images) {
        const sourcePath = path.join(sourceDir, image);
        const targetPath = path.join(targetDir, image);
        fs.copyFileSync(sourcePath, targetPath);
        console.log(`Copied ${image} to uploads directory`);
    }
};

// Function to seed profiles
const seedProfiles = async () => {
    try {
        // Clear existing profiles
        await Profile.deleteMany({});
        console.log('Cleared existing profiles');

        // Copy images
        await copyImages();
        console.log('Copied images to uploads directory');

        // Create new profiles
        const createdProfiles = await Profile.insertMany(profiles);
        console.log(`Created ${createdProfiles.length} profiles`);

        // Log created profiles for verification
        createdProfiles.forEach(profile => {
            console.log(`Created profile: ${profile.name} (${profile.gender}, ${profile.age}, ${profile.ethnicity})`);
        });

        console.log('Seeding completed successfully');
    } catch (error) {
        console.error('Error seeding profiles:', error);
    } finally {
        mongoose.disconnect();
    }
};

// Run the seeding
seedProfiles();
