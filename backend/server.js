const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const authRoutes = require('./routes/auth');
const jwt = require('jsonwebtoken'); // Added jwt module
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/profile_matching')
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.log('MongoDB connection error:', err));

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

    jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key', (err, user) => {
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

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});
