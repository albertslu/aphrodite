import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './App.css';

const Preferences = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [preferences, setPreferences] = useState({
        preferredGender: '',
        ageRange: '',
        location: '',
        idealMatch: ''
    });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState(location.state?.message || '');
    const isAdmin = localStorage.getItem('isAdmin') === 'true';

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setPreferences(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        
        try {
            const token = localStorage.getItem('token');
            
            // Call the matching endpoint
            const response = await fetch('http://localhost:5000/api/match/match-profiles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    prompt: `Looking for a ${preferences.preferredGender} aged ${preferences.ageRange} from ${preferences.location}. ${preferences.idealMatch}`
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to find matches');
            }

            const data = await response.json();
            // Store the matches in localStorage to display them on the next page
            const matches = data.matches || [];
            localStorage.setItem('matches', JSON.stringify(matches));
            navigate('/matches', { state: { matches } });
        } catch (error) {
            console.error('Error finding matches:', error);
            setError(error.message || 'Failed to find matches. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('isAdmin');
        navigate('/');
    };

    return (
        <div className="preferences-container">
            <div className="header-section">
                <h2>Tell Us Your Preferences</h2>
                <button onClick={handleLogout} className="logout-btn">
                    Logout
                </button>
            </div>
            {error && <div className="error">{error}</div>}
            {successMessage && <div className="success">{successMessage}</div>}
            
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
                            placeholder="Anywhere"
                            value={preferences.location}
                            onChange={handleInputChange}
                            className="form-control"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <div className="form-group">
                        <label>Describe Your Ideal Match</label>
                        <textarea
                            name="idealMatch"
                            placeholder="Describe your ideal match in detail. What qualities, values, and characteristics are you looking for in a partner?"
                            value={preferences.idealMatch}
                            onChange={handleInputChange}
                            className="form-control"
                            rows="6"
                            required
                        />
                    </div>
                </div>

                <button 
                    type="submit" 
                    className="btn btn-primary"
                    disabled={loading}
                >
                    {loading ? 'Finding Matches...' : 'Find My Matches'}
                </button>
            </form>
            
            {isAdmin && (
                <button 
                    onClick={() => navigate('/create-profile')}
                    className="create-profile-btn"
                >
                    Create New Profile
                </button>
            )}
        </div>
    );
};

export default Preferences;
