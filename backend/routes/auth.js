const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const config = require('../config');

// Signup
router.post('/signup', async (req, res) => {
    try {
        console.log('Received signup request:', req.body);
        const { username, password, phoneNumber } = req.body;

        if (!username || !password) {
            return res.status(400).json({ message: 'Username and password are required' });
        }

        try {
            // Check if user exists
            const existingUser = await User.findOne({ username });
            console.log('Existing user check:', existingUser ? 'found' : 'not found');
            
            if (existingUser) {
                return res.status(400).json({ message: 'Username already exists' });
            }

            // Create new user
            const user = new User({
                username,
                password,
                phoneNumber: phoneNumber || ''
            });

            console.log('Attempting to save user...');
            await user.save();
            console.log('User saved successfully:', username);

            // Generate token
            const token = jwt.sign(
                { userId: user._id },
                config.jwt.secret,
                { expiresIn: '24h' }
            );

            res.status(201).json({ 
                message: 'User created successfully',
                token 
            });
        } catch (dbError) {
            console.error('Database operation error:', dbError);
            throw new Error(`Database error: ${dbError.message}`);
        }
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ 
            message: 'Error creating user', 
            error: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

// Login
router.post('/login', async (req, res) => {
    try {
        console.log('Received login request for username:', req.body.username);
        const { username, password } = req.body;

        if (!username || !password) {
            return res.status(400).json({ message: 'Username and password are required' });
        }

        try {
            // Find user
            const user = await User.findOne({ username });
            if (!user) {
                console.log('User not found:', username);
                return res.status(401).json({ message: 'Invalid username or password' });
            }

            // Check password
            const isMatch = await user.comparePassword(password);
            if (!isMatch) {
                console.log('Invalid password for user:', username);
                return res.status(401).json({ message: 'Invalid username or password' });
            }

            console.log('Login successful for user:', username);

            // Generate token
            const token = jwt.sign(
                { userId: user._id },
                config.jwt.secret,
                { expiresIn: '24h' }
            );

            res.json({ 
                message: 'Login successful',
                token 
            });
        } catch (dbError) {
            console.error('Database operation error:', dbError);
            throw new Error(`Database error: ${dbError.message}`);
        }
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ 
            message: 'Error logging in', 
            error: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

module.exports = router;
