const jwt = require('jsonwebtoken');
const config = require('../config');

const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ message: 'Access denied. No token provided.' });
    }

    try {
        const decoded = jwt.verify(token, config.jwt.secret);
        req.user = {
            userId: decoded.userId,  // Make sure this matches what's set in the login route
            isAdmin: decoded.isAdmin
        };
        next();
    } catch (error) {
        console.error('Token verification error:', error);
        res.status(401).json({ message: 'Invalid token.' });
        
    }
};

module.exports = authenticateToken;
