const express = require('express');
const router = express.Router();
const User = require('../models/User');
const Profile = require('../models/Profile');

// Get all users (for development/testing only)
router.get('/users', async (req, res) => {
    try {
        const users = await User.find({}, { password: 0 }); // Exclude password field
        res.json(users);
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).json({ message: 'Error fetching users' });
    }
});

// Get all profiles (for development/testing only)
router.get('/profiles', async (req, res) => {
    try {
        const profiles = await Profile.find({}).populate('user', 'username -_id');
        res.json(profiles);
    } catch (error) {
        console.error('Error fetching profiles:', error);
        res.status(500).json({ message: 'Error fetching profiles' });
    }
});

module.exports = router;
