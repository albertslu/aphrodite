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
        partnerPreferences: "Athletic and health-conscious individual who enjoys an active lifestyle",
        photos: [
            {
                url: "/uploads/athletic_male_1.jpg",
                caption: "Working out at the gym",
                order: 1
            },
            {
                url: "/uploads/athletic_male_2.jpg",
                caption: "Full body shot showing athletic build",
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
        partnerPreferences: "Well-educated professional who values both career and personal growth",
        photos: [
            {
                url: "/uploads/professional_female_1.jpg",
                caption: "Professional headshot",
                order: 1
            },
            {
                url: "/uploads/professional_female_2.jpg",
                caption: "Full body photo in business casual",
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
        partnerPreferences: "Creative and open-minded person who loves art and self-expression",
        photos: [
            {
                url: "/uploads/artistic_male_1.jpg",
                caption: "Showing tattoos and artistic style",
                order: 1
            },
            {
                url: "/uploads/artistic_male_2.jpg",
                caption: "Working in design studio",
                order: 2
            }
        ]
    }
];

// Convert profile data from extracted dataset to our schema format
const formatProfileData = (rawProfile) => {
    return {
        name: `User${Math.floor(Math.random() * 10000)}`,
        gender: rawProfile.sex === 'm' ? 'male' : 'female',
        sexualOrientation: rawProfile.orientation,
        age: rawProfile.age,
        height: formatHeight(rawProfile.height),
        ethnicity: rawProfile.ethnicity.split(',')[0].trim(),
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

// Read and format profiles from the extracted dataset
const rawProfiles = JSON.parse(fs.readFileSync(path.join(__dirname, '../../Model/extracted_10_profiles.json')));
const extractedProfiles = rawProfiles.slice(0, 3).map(formatProfileData);

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
        // Clear existing profiles
        await Profile.deleteMany({});
        console.log('Cleared existing profiles');

        // Copy images
        await copyImages();
        console.log('Copied images to uploads directory');

        // Create new profiles
        const createdProfiles = await Profile.insertMany(allProfiles);
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
