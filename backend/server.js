const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const authRoutes = require('./routes/auth');
const profileRoutes = require('./routes/profile');
const jwt = require('jsonwebtoken');
const config = require('./config');

// Uncaught error handling
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    process.exit(1);
});

process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err);
    process.exit(1);
});

const app = express();
const port = config.server.port;

// Middleware
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true
}));

app.use(express.json());

// Request logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
});

// MongoDB connection status endpoint
app.get('/api/status', (req, res) => {
    const status = mongoose.connection.readyState;
    const statusMap = {
        0: 'disconnected',
        1: 'connected',
        2: 'connecting',
        3: 'disconnecting'
    };
    res.json({
        status: statusMap[status],
        timestamp: new Date().toISOString()
    });
});

// MongoDB connection with retry
const connectWithRetry = async () => {
    try {
        console.log('Attempting to connect to MongoDB...');
        console.log('MongoDB URI:', config.mongodb.uri.replace(/:[^:]*@/, ':****@'));

        const options = {
            serverSelectionTimeoutMS: 10000,
            socketTimeoutMS: 45000,
            family: 4
        };

        await mongoose.connect(config.mongodb.uri, options);
        
        console.log('=================================');
        console.log('✅ MongoDB connected successfully!');
        console.log('Database:', mongoose.connection.name);
        console.log('Host:', mongoose.connection.host);
        console.log('=================================');
    } catch (err) {
        console.error('❌ MongoDB connection error:', err.message);
        if (err.cause) {
            console.error('Cause:', err.cause.message);
        }
        console.log('Retrying in 5 seconds...');
        setTimeout(connectWithRetry, 5000);
    }
};

// Create uploads directory if it doesn't exist
const fs = require('fs');
if (!fs.existsSync('./uploads')) {
    fs.mkdirSync('./uploads');
}

// Profile Schema
const profileSchema = new mongoose.Schema({
    user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    name: { type: String, required: true },
    age: { type: Number, required: true },
    height: String,
    ethnicity: String,
    occupation: String,
    location: String,
    education: String,
    photos: [{ url: String }],
    aboutMe: String,
    interests: String,
    idealDate: String,
    dateCreated: { type: Date, default: Date.now }
});

const Profile = mongoose.model('Profile', profileSchema);

// User Schema (assuming it's defined in another file)
const User = require('./models/User'); // Assuming User model is defined in another file

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/profile', profileRoutes);

// Basic route for testing
app.get('/api/test', (req, res) => {
    res.json({ 
        message: 'Backend server is running',
        mongoStatus: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        timestamp: new Date().toISOString()
    });
});

// File upload configuration
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: function(req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 10000000 }, // 10MB limit
    fileFilter: function(req, file, cb) {
        checkFileType(file, cb);
    }
}).array('photos', 3); // Allow up to 3 photos

// Check file type
function checkFileType(file, cb) {
    const filetypes = /jpeg|jpg|png|gif/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);

    if (mimetype && extname) {
        return cb(null, true);
    } else {
        cb('Error: Images Only!');
    }
}

// Auth middleware
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

// Profile routes
app.post('/api/profiles', authenticateToken, upload, async (req, res) => {
    try {
        const photoUrls = req.files ? req.files.map(file => ({
            url: `/uploads/${file.filename}`
        })) : [];

        const profileData = {
            user: req.user.userId,
            name: req.body.name,
            age: req.body.age,
            height: req.body.height,
            ethnicity: req.body.ethnicity,
            occupation: req.body.occupation,
            location: req.body.location,
            education: req.body.education,
            photos: photoUrls,
            aboutMe: req.body.aboutMe,
            interests: req.body.interests,
            idealDate: req.body.idealDate
        };

        const profile = new Profile(profileData);
        await profile.save();

        // Update user with profile reference
        await User.findByIdAndUpdate(req.user.userId, { profile: profile._id });

        res.status(201).json(profile);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

app.get('/api/profiles', authenticateToken, async (req, res) => {
    try {
        const profiles = await Profile.find().populate('user', 'username');
        res.json(profiles);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

// Get user's own profile
app.get('/api/profile', authenticateToken, async (req, res) => {
    try {
        const profile = await Profile.findOne({ user: req.user.userId });
        if (!profile) {
            return res.status(404).json({ message: 'Profile not found' });
        }
        res.json(profile);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

// Serve uploaded files
app.use('/uploads', express.static('uploads'));

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({ 
        message: 'Something went wrong!', 
        error: err.message,
        stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
    });
});

// Start server and connect to MongoDB
const server = app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
    console.log('Connecting to MongoDB...');
    connectWithRetry();
});

// Handle server errors
server.on('error', (err) => {
    console.error('Server error:', err);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received. Shutting down gracefully...');
    server.close(() => {
        console.log('Server closed');
        mongoose.connection.close(false, () => {
            console.log('MongoDB connection closed');
            process.exit(0);
        });
    });
});
