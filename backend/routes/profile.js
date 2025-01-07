const express = require('express');
const router = express.Router();
const Profile = require('../models/Profile');
const User = require('../models/User');
const jwt = require('jsonwebtoken');
const config = require('../config');

// Middleware to authenticate token
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ message: 'Authentication required' });
    }

    jwt.verify(token, config.jwt.secret, (err, user) => {
        if (err) {
            return res.status(403).json({ message: 'Invalid or expired token' });
        }
        req.user = user;
        next();
    });
};

// Create profile
router.post('/', authenticateToken, async (req, res) => {
    try {
        console.log('Creating profile for user:', req.user.userId);
        console.log('Profile data:', req.body);

        // Check if profile already exists
        const existingProfile = await Profile.findOne({ user: req.user.userId });
        if (existingProfile) {
            return res.status(400).json({ message: 'Profile already exists for this user' });
        }

        // Create new profile
        const profileData = {
            ...req.body,
            user: req.user.userId
        };

        const profile = new Profile(profileData);
        await profile.save();

        // Update user with profile reference
        await User.findByIdAndUpdate(req.user.userId, { profile: profile._id });

        console.log('Profile created successfully');
        res.status(201).json({ 
            message: 'Profile created successfully',
            profile 
        });
    } catch (error) {
        console.error('Error creating profile:', error);
        if (error.name === 'ValidationError') {
            return res.status(400).json({ 
                message: 'Invalid profile data', 
                errors: Object.values(error.errors).map(err => err.message)
            });
        }
        res.status(500).json({ 
            message: 'Error creating profile',
            error: error.message
        });
    }
});

// Get profile
router.get('/', authenticateToken, async (req, res) => {
    try {
        const profile = await Profile.findOne({ user: req.user.userId });
        if (!profile) {
            return res.status(404).json({ message: 'Profile not found' });
        }
        res.json(profile);
    } catch (error) {
        console.error('Error fetching profile:', error);
        res.status(500).json({ 
            message: 'Error fetching profile',
            error: error.message
        });
    }
});

// Update profile
router.put('/', authenticateToken, async (req, res) => {
    try {
        const profile = await Profile.findOneAndUpdate(
            { user: req.user.userId },
            req.body,
            { new: true, runValidators: true }
        );
        
        if (!profile) {
            return res.status(404).json({ message: 'Profile not found' });
        }
        
        res.json({ 
            message: 'Profile updated successfully',
            profile 
        });
    } catch (error) {
        console.error('Error updating profile:', error);
        if (error.name === 'ValidationError') {
            return res.status(400).json({ 
                message: 'Invalid profile data', 
                errors: Object.values(error.errors).map(err => err.message)
            });
        }
        res.status(500).json({ 
            message: 'Error updating profile',
            error: error.message
        });
    }
});

module.exports = router;
