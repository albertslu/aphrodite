// src/App.js
import React from 'react';
import './App.css';
import ProfileCreation from './ProfileCreation';
import Login from './Login';
import Signup from './Signup';
import Preferences from './Preferences';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';

function App() {
    return (
        <Router>
            <div className="App">
                <Routes>
                    <Route path="/" element={<Login />} />
                    <Route path="/signup" element={<Signup />} />
                    <Route path="/create-profile" element={<ProfileCreation />} />
                    <Route path="/preferences" element={<Preferences />} />
                    <Route path="/matches" element={<div>Matches Page (Coming Soon)</div>} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
