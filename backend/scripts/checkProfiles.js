const mongoose = require('mongoose');
const config = require('../config');
const Profile = require('../models/Profile');

mongoose.connect(config.mongodb.uri, {
    serverSelectionTimeoutMS: 10000,
    socketTimeoutMS: 45000,
    family: 4
})
.then(async () => {
    try {
        const profiles = await Profile.find({});
        console.log('Current profiles in database:', profiles.length);
        profiles.forEach(profile => {
            console.log(`Profile: ${profile.name} (${profile.gender}, ${profile.age}, ${profile.ethnicity})`);
            console.log('Photos:', profile.photos.map(p => p.url).join(', '));
            console.log('---');
        });
    } catch (error) {
        console.error('Error:', error);
    } finally {
        mongoose.disconnect();
    }
})
.catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
});
