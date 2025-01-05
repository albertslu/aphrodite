require('dotenv').config();

const config = {
    mongodb: {
        uri: process.env.MONGODB_URI || 'mongodb+srv://albertlu:test@aphrodite.nr8sj.mongodb.net/profile_matching?tls=true&authSource=admin'
    },
    jwt: {
        secret: process.env.JWT_SECRET || 'aphrodite_secret_key_2025'
    },
    server: {
        port: process.env.PORT || 5000
    }
};

module.exports = config;
