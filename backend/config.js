const config = {
    mongodb: {
        uri: 'mongodb+srv://albertlu:test@aphrodite.nr8sj.mongodb.net/profile_matching?retryWrites=true&w=majority&appName=Aphrodite'
    },
    jwt: {
        secret: 'aphrodite_secret_key_2025'
    },
    server: {
        port: 5000
    }
};

module.exports = config;
