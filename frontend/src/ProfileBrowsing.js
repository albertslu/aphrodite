// src/ProfileBrowsing.js
import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import './App.css';

const ProfileBrowsing = () => {
    const location = useLocation();
    const profiles = location.state?.profiles || [];

    const [currentProfileIndex, setCurrentProfileIndex] = useState(0);

    const handleLike = () => {
        console.log('Liked:', profiles[currentProfileIndex]);
        nextProfile();
    };

    const handlePass = () => {
        console.log('Passed:', profiles[currentProfileIndex]);
        nextProfile();
    };

    const nextProfile = () => {
        setCurrentProfileIndex((prevIndex) => (prevIndex + 1) % profiles.length);
    };

    const currentProfile = profiles[currentProfileIndex];

    return (
        <div>
            <h1>Profile Browsing</h1>
            {currentProfile ? (
                <div className="profile-card">
                    <img src={currentProfile.picture} alt={`${currentProfile.name}'s profile`} />
                    <h2>{currentProfile.name}, {currentProfile.age}</h2>
                    <p>{currentProfile.description}</p>
                    <div className="profile-actions">
                        <button onClick={handlePass} className="pass-button">Pass</button>
                        <button onClick={handleLike} className="like-button">Like</button>
                    </div>
                </div>
            ) : (
                <p>No profiles available.</p>
            )}
        </div>
    );
};

export default ProfileBrowsing;
