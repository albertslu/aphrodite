const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const Profile = require('../models/Profile');

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/profile_matching', {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

// Sample profiles data
const profiles = [
    {
        name: "Emma Watson",
        gender: "female",
        sexualOrientation: "straight",
        age: 33,
        height: "5'5\"",
        ethnicity: "White",
        occupation: "Actress",
        location: "London",
        education: "Brown University",
        aboutMe: "Passionate about acting, women's rights, and sustainable fashion.",
        interests: "Acting, reading, yoga, environmental activism",
        relationshipGoals: "Looking for a meaningful long-term relationship",
        partnerPreferences: "Someone who is passionate about their work and cares about social causes",
        photos: [
            {
                url: "/uploads/emma1.jpg",
                caption: "Professional headshot",
                order: 1
            },
            {
                url: "/uploads/emma2.jpg",
                caption: "Casual photo",
                order: 2
            }
        ]
    },
    {
        name: "Chris Evans",
        gender: "male",
        sexualOrientation: "straight",
        age: 42,
        height: "6'0\"",
        ethnicity: "White",
        occupation: "Actor",
        location: "Los Angeles",
        education: "Lincoln-Sudbury Regional High School",
        aboutMe: "Love acting, staying fit, and spending time with my dog Dodger.",
        interests: "Fitness, acting, directing, playing piano",
        relationshipGoals: "Looking for someone genuine and down-to-earth",
        partnerPreferences: "Someone who values family and has a good sense of humor",
        photos: [
            {
                url: "/uploads/chris1.jpg",
                caption: "Professional photo",
                order: 1
            },
            {
                url: "/uploads/chris2.jpg",
                caption: "With my dog",
                order: 2
            }
        ]
    }
    // Add more profiles as needed
];

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

        console.log('Seeding completed successfully');
    } catch (error) {
        console.error('Error seeding profiles:', error);
    } finally {
        mongoose.disconnect();
    }
};

// Run the seeding
seedProfiles();
