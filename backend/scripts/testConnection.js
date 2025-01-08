const mongoose = require('mongoose');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../../.env') });

console.log('Testing MongoDB connection...');

mongoose.connect(process.env.MONGODB_URI)
    .then(() => {
        console.log('Successfully connected to MongoDB!');
        mongoose.disconnect();
    })
    .catch(err => {
        console.error('Connection error:', err);
        process.exit(1);
    });
