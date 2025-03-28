const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const jwt = require('jsonwebtoken');
const config = require('./config');

// Import routes
const authRoutes = require('./routes/auth');
const profileRoutes = require('./routes/profile');
const adminRoutes = require('./routes/admin');
const uploadRoutes = require('./routes/upload');
const matchRoutes = require('./routes/match');

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
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    console.log('Request Headers:', req.headers);
    console.log('Request Body:', req.body);
    next();
});

app.use(cors({
    origin: ['https://frontend-zeta-amber.vercel.app', 'http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
    optionsSuccessStatus: 200
}));

// Handle OPTIONS requests explicitly
app.options('*', cors());

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

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
        console.log('NODE_ENV:', process.env.NODE_ENV);
        console.log('MongoDB URI:', config.mongodb.uri);

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
        console.log('Full connection details:', mongoose.connection.client.s.url);
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

// Routes
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Use the existing routes
app.use('/api/auth', authRoutes);
app.use('/api/profile', profileRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/match', matchRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({ 
        message: 'Internal server error',
        error: err.message 
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
