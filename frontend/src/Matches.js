import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './App.css';

const Matches = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [matches, setMatches] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        // Get matches from location state or localStorage
        const matchData = location.state?.matches || JSON.parse(localStorage.getItem('matches') || '[]');
        setMatches(matchData);
        setLoading(false);
    }, [location.state]);

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('isAdmin');
        navigate('/');
    };

    const handleBackToPreferences = () => {
        navigate('/preferences');
    };

    if (loading) {
        return (
            <div className="matches-container">
                <div className="loading">Loading matches...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="matches-container">
                <div className="error">{error}</div>
                <button onClick={handleBackToPreferences}>Back to Preferences</button>
            </div>
        );
    }

    return (
        <div className="matches-container">
            <div className="header-section">
                <h2>Your Matches</h2>
                <div className="header-buttons">
                    <button onClick={handleBackToPreferences} className="back-btn">
                        Back to Preferences
                    </button>
                    <button onClick={handleLogout} className="logout-btn">
                        Logout
                    </button>
                </div>
            </div>

            {matches.length === 0 ? (
                <div className="no-matches">
                    <p>No matches found. Try adjusting your preferences.</p>
                    <button onClick={handleBackToPreferences}>Update Preferences</button>
                </div>
            ) : (
                <div className="matches-grid">
                    {matches.map((match, index) => (
                        <div key={index} className="match-card">
                            <div className="match-photos">
                                {match.photos && match.photos.length > 0 ? (
                                    <div className="photo-carousel">
                                        {match.photos.map((photo, photoIndex) => (
                                            <img
                                                key={photoIndex}
                                                src={photo.url}
                                                alt={`${match.name}'s photo ${photoIndex + 1}`}
                                                className="profile-photo"
                                            />
                                        ))}
                                    </div>
                                ) : (
                                    <div className="no-photo">No photos available</div>
                                )}
                            </div>
                            <div className="match-info">
                                <h3>{match.name}</h3>
                                <div className="basic-info">
                                    <p>{match.age} years old • {match.gender}</p>
                                    <p>{match.location}</p>
                                    <p>{match.ethnicity}</p>
                                </div>
                                <div className="match-details">
                                    <h4>About Me</h4>
                                    <p>{match.aboutMe}</p>
                                    
                                    <h4>Occupation</h4>
                                    <p>{match.occupation}</p>
                                    
                                    <h4>Education</h4>
                                    <p>{match.education}</p>
                                    
                                    <h4>Interests</h4>
                                    <div className="interests-tags">
                                        {match.interests.map((interest, i) => (
                                            <span key={i} className="interest-tag">{interest}</span>
                                        ))}
                                    </div>

                                    <h4>Relationship Goals</h4>
                                    <p>{match.relationshipGoals}</p>
                                </div>
                                {match.matchScore && (
                                    <div className="match-score">
                                        <h4>Match Score</h4>
                                        <div className="score">{Math.round(match.matchScore * 100)}% Match</div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Matches;
