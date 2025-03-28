const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const Profile = require('../models/Profile');
const config = require('../config');

console.log('Attempting to connect to MongoDB...');
console.log('MongoDB URI:', config.mongodb.uri.replace(/:[^:]*@/, ':****@'));

// MongoDB Atlas connection with proper options
mongoose.connect(config.mongodb.uri, {
    serverSelectionTimeoutMS: 10000,
    socketTimeoutMS: 45000,
    family: 4  // Force IPv4
})
.then(() => {
    console.log('=================================');
    console.log('✅ MongoDB connected successfully!');
    console.log('Database:', mongoose.connection.name);
    console.log('Host:', mongoose.connection.host);
    console.log('=================================');
    // Only start seeding after connection is established
    seedProfiles();
})
.catch(err => {
    console.error('❌ MongoDB connection error:', err);
    if (err.cause) {
        console.error('Cause:', err.cause.message);
    }
    process.exit(1);
});

// Convert height from inches to readable format
const formatHeight = (inches) => {
    const feet = Math.floor(inches / 12);
    const remainingInches = inches % 12;
    return `${feet}'${remainingInches}"`;
};

// Synthetic profile templates for testing different matching scenarios
const syntheticProfiles = [
    {
        name: "Athletic_Test_Male",
        gender: "male",
        sexualOrientation: "straight",
        age: 28,
        height: "6'1\"",
        ethnicity: "white",
        occupation: "Personal Trainer",
        location: "Los Angeles, California",
        education: "Bachelor's in Exercise Science",
        aboutMe: "Dedicated fitness enthusiast and personal trainer. I spend most of my time at the gym, helping others achieve their fitness goals. My athletic build comes from years of dedication to weightlifting and proper nutrition.",
        interests: "Weightlifting, CrossFit, nutrition, outdoor sports, hiking",
        relationshipGoals: "Looking for someone who shares my passion for fitness and healthy living",
        photos: [
            {
                url: "/uploads/athletic_male_1.jpg",
                order: 1
            },
            {
                url: "/uploads/athletic_male_2.jpg",
                order: 2
            }
        ]
    },
    {
        name: "Professional_Test_Female",
        gender: "female",
        sexualOrientation: "straight",
        age: 32,
        height: "5'6\"",
        ethnicity: "asian",
        occupation: "Software Engineer",
        location: "San Francisco, California",
        education: "Master's in Computer Science",
        aboutMe: "Tech professional by day, foodie by night. I have a slim build and maintain an active lifestyle despite long hours coding. Love exploring new restaurants and traveling.",
        interests: "Coding, traveling, fine dining, yoga, photography",
        relationshipGoals: "Seeking a meaningful connection with someone ambitious and curious about life",
        photos: [
            {
                url: "/uploads/professional_female_1.jpg",
                order: 1
            },
            {
                url: "/uploads/professional_female_2.jpg",
                order: 2
            }
        ]
    },
    {
        name: "Artistic_Test_Male",
        gender: "male",
        sexualOrientation: "straight",
        age: 25,
        height: "5'10\"",
        ethnicity: "hispanic",
        occupation: "Graphic Designer",
        location: "Brooklyn, New York",
        education: "BFA in Design",
        aboutMe: "Creative soul with a lean build. Covered in tattoos that tell my life story. When I'm not designing, you'll find me at art galleries or working on my photography portfolio.",
        interests: "Art, photography, tattoos, indie music, vintage fashion",
        relationshipGoals: "Looking for someone who appreciates creativity and alternative lifestyle",
        photos: [
            {
                url: "/uploads/artistic_male_1.jpg",
                order: 1
            },
            {
                url: "/uploads/artistic_male_2.jpg",
                order: 2
            }
        ]
    }
];

// Helper function to generate profile photo URLs based on characteristics
const generatePhotoUrls = (profile) => {
    const gender = profile.sex === 'm' ? 'male' : 'female';
    const ageGroup = profile.age < 25 ? 'young' : profile.age < 35 ? 'mid' : 'mature';
    const ethnicity = profile.ethnicity.split(',')[0].trim().toLowerCase();
    
    return [
        {
            url: `/uploads/${gender}_${ageGroup}_${ethnicity}_1.jpg`,
            order: 1
        },
        {
            url: `/uploads/${gender}_${ageGroup}_${ethnicity}_2.jpg`,
            order: 2
        }
    ];
};

// Read and format profiles from the extracted dataset
const rawProfiles = JSON.parse(fs.readFileSync(path.join(__dirname, '../../Model/extracted_10_profiles.json')));

// Convert profile data from extracted dataset to our schema format
const formatProfileData = (rawProfile) => {
    // Clean and format the text fields
    const cleanText = (text) => {
        if (!text) return '';
        return text.replace(/[^\w\s.,!?-]/g, ' ')  // Remove special characters
                  .replace(/\s+/g, ' ')            // Remove extra spaces
                  .trim();
    };

    return {
        name: `User${Math.floor(Math.random() * 10000)}`,
        gender: rawProfile.sex === 'm' ? 'male' : 'female',
        sexualOrientation: rawProfile.orientation || 'straight',
        age: rawProfile.age,
        height: formatHeight(rawProfile.height),
        ethnicity: rawProfile.ethnicity.split(',')[0].trim(),
        occupation: cleanText(rawProfile.job) || 'Not specified',
        location: cleanText(rawProfile.location) || 'Not specified',
        education: cleanText(rawProfile.education) || 'Not specified',
        aboutMe: cleanText(rawProfile.essay0) || 'No description provided',
        interests: cleanText(rawProfile.essay2) || 'No interests provided',
        relationshipGoals: cleanText(rawProfile.essay9) || 'Looking for meaningful connections',
        photos: generatePhotoUrls(rawProfile)
    };
};

// Select and format extracted profiles
const extractedProfiles = rawProfiles
    .filter(profile => {
        // Filter out profiles with missing critical data
        return profile.age && 
               profile.sex && 
               profile.ethnicity && 
               profile.height && 
               profile.essay0; // Ensure we have at least some text content
    })
    .slice(0, 7) // Take 7 profiles from the extracted dataset
    .map(formatProfileData);

// Combine synthetic and extracted profiles
const allProfiles = [...syntheticProfiles, ...extractedProfiles];

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
        // Instead of clearing profiles, check which ones already exist
        console.log('Checking existing profiles...');
        const existingProfiles = await Profile.find({});
        console.log(`Found ${existingProfiles.length} existing profiles`);

        // Copy images
        console.log('Copying images...');
        await copyImages();
        console.log('Copied images to uploads directory');

        // Filter out profiles that already exist (by name)
        const existingNames = new Set(existingProfiles.map(p => p.name));
        const profilesToAdd = allProfiles.filter(p => !existingNames.has(p.name));

        if (profilesToAdd.length === 0) {
            console.log('All profiles already exist in the database');
        } else {
            // Create new profiles
            console.log(`Adding ${profilesToAdd.length} new profiles...`);
            const createdProfiles = await Profile.insertMany(profilesToAdd);
            console.log(`Created ${createdProfiles.length} new profiles`);

            // Log created profiles for verification
            createdProfiles.forEach(profile => {
                console.log(`Created profile: ${profile.name} (${profile.gender}, ${profile.age}, ${profile.ethnicity})`);
            });
        }

        console.log('Seeding completed successfully');
        mongoose.disconnect();
    } catch (error) {
        console.error('Error seeding profiles:', error);
        mongoose.disconnect();
        process.exit(1);
    }
};
