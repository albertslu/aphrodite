import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const Preferences = () => {
    const [preferences, setPreferences] = useState({
        ageRange: '',
        location: '',
        preferredGender: '',
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
                        <label>Preferred Gender</label>
                        <select
                            name="preferredGender"
                            value={preferences.preferredGender}
                            onChange={handleInputChange}
                            className="form-control"
                            required
                        >
                            <option value="">Select Gender</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="non-binary">Non-binary</option>
                            <option value="any">Any</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label>Desired Age Range</label>
                        <input
                            type="text"
                            name="ageRange"
                            placeholder="e.g., 25-35"
                            value={preferences.ageRange}
                            onChange={handleInputChange}
                            className="form-control"
                            required
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
                            required
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>Describe Your Ideal Match</label>
                        <textarea
                            name="idealMatch"
                            placeholder="Describe your ideal match in detail. What qualities, values, and characteristics are you looking for in a partner? Feel free to be specific about personality traits, lifestyle, ambitions, and anything else that matters to you."
                            value={preferences.idealMatch}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="6"
                            required
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
