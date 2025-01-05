const mongoose = require('mongoose');

const profileSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    name: {
        type: String,
        required: true
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
        type: String,
        required: true
    },
    location: {
        type: String,
        required: true
    },
    education: String,
    photos: [{
        url: String,
        caption: String
    }],
    aboutMe: {
        type: String,
        required: true
    },
    interests: {
        type: String,
        required: true
    },
    idealDate: {
        type: String,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Profile', profileSchema);
