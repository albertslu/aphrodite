const mongoose = require('mongoose');

const profileSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
        // Removed required: true so admin can create profiles without users
    },
    name: {
        type: String,
        required: true
    },
    gender: {
        type: String,
        required: true,
        enum: ['male', 'female', 'non-binary', 'other']
    },
    sexualOrientation: {
        type: String,
        required: true,
        enum: ['straight', 'gay', 'lesbian', 'bisexual', 'pansexual', 'other']
    },
    age: {
        type: Number,
        required: true
    },
    height: {
        type: String,
        required: true
    },
    ethnicity: String,
    occupation: {
        type: String
    },
    location: {
        type: String,
        required: true
    },
    education: String,
    photos: [{
        url: String,
        caption: String,
        order: Number
    }],
    aboutMe: {
        type: String,
        default: ''
    },
    interests: {
        type: String,
        default: ''
    },
    relationshipGoals: {
        type: String,
        default: ''
    },
    partnerPreferences: {
        type: String,
        default: ''
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Validate photos array length
profileSchema.path('photos').validate(function(photos) {
    return photos.length <= 3;
}, 'You can only upload up to 3 photos');

module.exports = mongoose.model('Profile', profileSchema);
