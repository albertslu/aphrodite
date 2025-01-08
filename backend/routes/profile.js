const express = require('express');
const router = express.Router();
const Profile = require('../models/Profile');
const User = require('../models/User');
const jwt = require('jsonwebtoken');
const config = require('../config');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);

        if (extname && mimetype) {
            return cb(null, true);
        } else {
            cb('Error: Images only!');
        }
    }
});

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

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
router.post('/', authenticateToken, upload.none(), async (req, res) => {
    try {
        console.log('Creating profile with data:', req.body);
        const user = await User.findById(req.user.userId);
        
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        // If admin user, create profile without linking
        if (user.isAdmin) {
            console.log('Creating unlinked profile as admin');
            const profile = new Profile({
                ...req.body,
                age: Number(req.body.age) // Ensure age is a number
            });
            await profile.save();
            return res.status(201).json({ 
                message: 'Profile created successfully', 
                profile,
                isAdmin: true
            });
        }

        // For regular users, check if profile exists
        const existingProfile = await Profile.findOne({ user: req.user.userId });
        if (existingProfile) {
            return res.status(400).json({ message: 'Profile already exists for this user' });
        }

        // Create new profile linked to user
        const profileData = {
            ...req.body,
            user: req.user.userId,
            age: Number(req.body.age) // Ensure age is a number
        };

        console.log('Creating profile with data:', profileData);
        const profile = new Profile(profileData);
        await profile.save();

        // Update user with profile reference
        user.profile = profile._id;
        await user.save();

        console.log('Profile created successfully');
        res.status(201).json({ 
            message: 'Profile created successfully',
            profile,
            isAdmin: false
        });
    } catch (error) {
        console.error('Error creating profile:', error);
        if (error.name === 'ValidationError') {
            return res.status(400).json({ 
                message: 'Invalid profile data', 
                errors: error.errors
            });
        }
        res.status(500).json({ 
            message: 'Error creating profile',
            error: error.message,
            stack: error.stack
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
