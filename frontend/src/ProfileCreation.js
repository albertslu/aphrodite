// src/ProfileCreation.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css'; // Ensure you import the CSS file

const ProfileCreation = () => {
    const [name, setName] = useState('');
    const [age, setAge] = useState('');
    const [profilePicture, setProfilePicture] = useState(null);
    const [preferences, setPreferences] = useState([]);
    const [currentPreference, setCurrentPreference] = useState('');
    const navigate = useNavigate();

    const handleNameChange = (e) => setName(e.target.value);
    const handleAgeChange = (e) => setAge(e.target.value);
    const handlePictureChange = (e) => setProfilePicture(URL.createObjectURL(e.target.files[0]));
    const handleCurrentPreferenceChange = (e) => setCurrentPreference(e.target.value);
    const handlePreferencesAdd = () => {
        if (currentPreference && !preferences.includes(currentPreference)) {
            setPreferences([...preferences, currentPreference]);
            setCurrentPreference('');
        }
    };
    const handlePreferencesKeyPress = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handlePreferencesAdd();
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        // Handle form submission logic here
        console.log({ name, age, profilePicture, preferences });
        // Navigate to preference input screen
        navigate('/preferences');
    };

    return (
        <div>
            <h1>Create Your Profile</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Name: </label>
                    <input type="text" value={name} onChange={handleNameChange} required />
                </div>
                <div>
                    <label>Age: </label>
                    <input type="number" value={age} onChange={handleAgeChange} required />
                </div>
                <div>
                    <label>Profile Picture: </label>
                    <input type="file" onChange={handlePictureChange} required />
                    {profilePicture && <img src={profilePicture} alt="Profile Preview" width="100" />}
                </div>
                <div>
                    <label>Preferences: </label>
                    <input type="text" value={currentPreference} onChange={handleCurrentPreferenceChange} onKeyPress={handlePreferencesKeyPress} placeholder="Add a preference tag" />
                    <button type="button" onClick={handlePreferencesAdd}>Add</button>
                    <div>
                        {preferences.map((preference, index) => (
                            <span key={index}>
                                {preference}
                            </span>
                        ))}
                    </div>
                </div>
                <button type="submit">Save Profile</button>
            </form>
        </div>
    );
};

export default ProfileCreation;
