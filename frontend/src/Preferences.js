import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const Preferences = () => {
    const [preferences, setPreferences] = useState({
        ageRange: '',
        location: '',
        lookingFor: '',
        dealBreakers: '',
        interests: '',
        idealMatch: ''
    });
    const navigate = useNavigate();

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setPreferences(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:5000/api/preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(preferences)
            });

            if (!response.ok) {
                throw new Error('Failed to save preferences');
            }

            // Navigate to matches page (to be created)
            navigate('/matches');
        } catch (error) {
            console.error('Error saving preferences:', error);
        }
    };

    return (
        <div className="preferences-container">
            <h2>Tell Us Your Preferences</h2>
            <p className="subtitle">Help our AI find your perfect match</p>
            
            <form onSubmit={handleSubmit} className="preferences-form">
                <div className="form-section">
                    <div className="form-group">
                        <label>Desired Age Range</label>
                        <input
                            type="text"
                            name="ageRange"
                            placeholder="e.g., 25-35"
                            value={preferences.ageRange}
                            onChange={handleInputChange}
                            className="form-control"
                        />
                    </div>
                    
                    <div className="form-group">
                        <label>Preferred Location</label>
                        <input
                            type="text"
                            name="location"
                            placeholder="e.g., New York City or within 50 miles"
                            value={preferences.location}
                            onChange={handleInputChange}
                            className="form-control"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>What are you looking for?</label>
                        <textarea
                            name="lookingFor"
                            placeholder="Describe what you're looking for in a relationship..."
                            value={preferences.lookingFor}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="4"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>Deal Breakers</label>
                        <textarea
                            name="dealBreakers"
                            placeholder="What are your absolute deal breakers?"
                            value={preferences.dealBreakers}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="3"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>Interests You Want to Share</label>
                        <textarea
                            name="interests"
                            placeholder="What interests or hobbies would you like your match to share?"
                            value={preferences.interests}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="3"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>Describe Your Ideal Match</label>
                        <textarea
                            name="idealMatch"
                            placeholder="Tell us about your ideal match in detail..."
                            value={preferences.idealMatch}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="4"
                        />
                    </div>
                </div>

                <button type="submit" className="btn btn-primary">
                    Find My Matches
                </button>
            </form>
        </div>
    );
};

export default Preferences;
