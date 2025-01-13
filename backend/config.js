const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const config = {
    mongodb: {
        uri: process.env.MONGODB_URI || 'mongodb+srv://albertlu:test@aphrodite.nr8sj.mongodb.net/profile_matching?retryWrites=true&w=majority&appName=Aphrodite&tls=true'
    },
    jwt: {
        secret: process.env.JWT_SECRET || 'aphrodite_secret_key_2025'
    },
    server: {
        port: process.env.PORT || 5000
    }
};

module.exports = config;
