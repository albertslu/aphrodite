// src/App.js
import React from 'react';
import './App.css';
import ProfileCreation from './ProfileCreation';
import Login from './Login';
import Signup from './Signup';
import Preferences from './Preferences';
import Matches from './Matches';
import FullProfile from './FullProfile';
import { BrowserRouter as Router, Route, Routes, Navigate, useLocation } from 'react-router-dom';

function App() {
    const PrivateRoute = ({ children }) => {
        const token = localStorage.getItem('token');
        const isAdmin = localStorage.getItem('isAdmin') === 'true';
        const location = useLocation();
        
        if (!token) {
            return <Navigate to="/login" state={{ from: location }} />;
        }

        // Admin users can access all routes
        if (isAdmin) {
            return children;
        }

        // For regular users, check if they're trying to access profile creation
        // after already having created a profile
        if (location.pathname === '/create-profile') {
            // You might want to add an API check here to see if they already have a profile
            // For now, we'll let them through
            return children;
        }

        return children;
    };

    return (
        <Router>
            <div className="App">
                <Routes>
                    <Route path="/" element={<Login />} />
                    <Route path="/signup" element={<Signup />} />
                    <Route path="/create-profile" element={<PrivateRoute><ProfileCreation /></PrivateRoute>} />
                    <Route path="/edit-profile" element={<PrivateRoute><ProfileCreation isEditing={true} /></PrivateRoute>} />
                    <Route path="/preferences" element={<PrivateRoute><Preferences /></PrivateRoute>} />
                    <Route path="/matches" element={<PrivateRoute><Matches /></PrivateRoute>} />
                    <Route path="/profile/:id" element={<FullProfile />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
