const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const config = require('../config');

// Signup
router.post('/signup', async (req, res) => {
    try {
        const { username, password, isAdmin } = req.body;

        // Check if username already exists
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            return res.status(400).json({ message: 'Username already exists' });
        }

        // Create new user
        const user = new User({
            username,
            password,
            isAdmin: isAdmin || false // Only set to true if explicitly provided
        });

        await user.save();

        // Create and sign JWT token
        const token = jwt.sign(
            { userId: user._id, username: user.username, isAdmin: user.isAdmin },
            config.jwt.secret,
            { expiresIn: '24h' }
        );

        res.status(201).json({
            message: 'User created successfully',
            token
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: 'Error creating user' });
    }
});

// Special admin signup route (temporary)
router.post('/admin-signup', async (req, res) => {
    try {
        const { username, password } = req.body;

        // Check if username already exists
        const existingUser = await User.findOne({ username });
        if (existingUser) {
            return res.status(400).json({ message: 'Username already exists' });
        }

        // Create admin user
        const user = new User({
            username,
            password,
            isAdmin: true,
            createdAt: new Date()
        });

        await user.save();

        res.status(201).json({
            message: 'Admin user created successfully',
            userId: user._id
        });
    } catch (error) {
        console.error('Admin signup error:', error);
        res.status(500).json({ message: 'Error creating admin user' });
    }
});

// Login
router.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body;

        // Find user
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Check password
        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Generate token with isAdmin flag
        const token = jwt.sign(
            { 
                userId: user._id, 
                username: user.username,
                isAdmin: user.isAdmin 
            },
            config.jwt.secret,
            { expiresIn: '24h' }
        );

        res.json({ 
            token,
            isAdmin: user.isAdmin
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Error logging in' });
    }
});

module.exports = router;
