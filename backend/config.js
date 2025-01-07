const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const config = {
    mongodb: {
        uri: process.env.MONGODB_URI || 'mongodb://albertlu:test@ac-6qvt5dn-shard-00-00.nr8sj.mongodb.net:27017,ac-6qvt5dn-shard-00-01.nr8sj.mongodb.net:27017,ac-6qvt5dn-shard-00-02.nr8sj.mongodb.net:27017/profile_matching?ssl=true&replicaSet=atlas-yzjm6q-shard-0&authSource=admin'
    },
    jwt: {
        secret: process.env.JWT_SECRET || 'aphrodite_secret_key_2025'
    },
    server: {
        port: process.env.PORT || 5000
    }
};

module.exports = config;
