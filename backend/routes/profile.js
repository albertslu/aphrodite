const express = require('express');
const router = express.Router();
const Profile = require('../models/Profile');
const authenticateToken = require('../middleware/authenticateToken');

// Get current user's profile
router.get('/', authenticateToken, async (req, res) => {
    try {
        console.log('Getting profile for user:', req.user.userId);
        const profile = await Profile.findOne({ user: req.user.userId });
        
        if (!profile) {
            console.log('No profile found for user:', req.user.userId);
            return res.status(404).json({ message: 'Profile not found' });
        }
        
        console.log('Profile found:', profile);
        res.json(profile);
    } catch (error) {
        console.error('Error fetching profile:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// Create new profile
router.post('/create', authenticateToken, async (req, res) => {
    try {
        console.log('Creating profile for user:', req.user.userId);
        // Check if profile already exists
        let profile = await Profile.findOne({ user: req.user.userId });
        if (profile) {
            return res.status(400).json({ message: 'Profile already exists' });
        }

        // Create new profile
        profile = new Profile({
            ...req.body,
            user: req.user.userId
        });

        await profile.save();
        console.log('Profile created successfully:', profile);
        res.status(201).json(profile);
    } catch (error) {
        console.error('Error creating profile:', error);
        if (error.name === 'ValidationError') {
            return res.status(400).json({ message: error.message });
        }
        res.status(500).json({ message: 'Server error' });
    }
});

// Update user's profile
router.put('/', authenticateToken, async (req, res) => {
    try {
        console.log('Updating profile for user:', req.user.userId);
        const profile = await Profile.findOne({ user: req.user.userId });
        if (!profile) {
            return res.status(404).json({ message: 'Profile not found' });
        }

        // Update profile fields
        Object.keys(req.body).forEach(key => {
            if (key !== 'user' && key !== '_id') { // Don't allow updating user ID or MongoDB _id
                profile[key] = req.body[key];
            }
        });

        await profile.save();
        console.log('Profile updated successfully:', profile);
        res.json(profile);
    } catch (error) {
        console.error('Error updating profile:', error);
        if (error.name === 'ValidationError') {
            return res.status(400).json({ message: error.message });
        }
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;
