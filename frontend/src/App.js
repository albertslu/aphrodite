// src/App.js
import React from 'react';
import './App.css';
import ProfileCreation from './ProfileCreation';
import PreferenceInput from './PreferenceInput';
import ProfileBrowsing from './ProfileBrowsing';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
    return (
        <Router>
            <div className="App   ">
                <header className="App-header">
                    <h1>Profile Matching App</h1>
                    <Routes>
                        <Route path="/" element={<ProfileCreation />} />
                        <Route path="/preferences" element={<PreferenceInput />} />
                        <Route path="/browse" element={<ProfileBrowsing />} />
                    </Routes>
                </header>
            </div>
        </Router>
    );
}

export default App;
