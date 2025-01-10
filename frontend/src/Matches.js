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
        try {
            // Get matches from location state or localStorage
            const matchData = location.state?.matches || JSON.parse(localStorage.getItem('matches') || '[]');
            console.log('Match data received:', matchData); // Debug log
            if (!Array.isArray(matchData)) {
                console.error('Invalid match data format:', matchData);
                setError('Invalid match data format');
                setMatches([]);
            } else {
                setMatches(matchData);
            }
        } catch (err) {
            console.error('Error processing match data:', err);
            setError('Error loading matches');
            setMatches([]);
        } finally {
            setLoading(false);
        }
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
                <div className="header-title">
                    <h2>AI Found These Profiles For You</h2>
                    <p className="subtitle">Based on your preferences</p>
                </div>
                <div className="header-buttons">
                    <button onClick={handleBackToPreferences} className="back-btn">
                        Back to Preferences
                    </button>
                    <button onClick={handleLogout} className="logout-btn">
                        Logout
                    </button>
                </div>
            </div>

            {!matches || matches.length === 0 ? (
                <div className="no-matches">
                    <p>No matches found. Try adjusting your preferences.</p>
                    <button onClick={handleBackToPreferences}>Update Preferences</button>
                </div>
            ) : (
                <div className="matches-grid">
                    {matches.map((match, index) => (
                        <div key={match.profile._id || index} className="match-card">
                            <div className="match-photos">
                                {match.profile.photos && match.profile.photos.length > 0 ? (
                                    <div className="photo-carousel">
                                        {match.profile.photos.map((photo, photoIndex) => (
                                            <img
                                                key={photoIndex}
                                                src={`http://localhost:5000${photo}`}
                                                alt={`${match.profile.name}'s photo ${photoIndex + 1}`}
                                                className="profile-photo"
                                                loading="lazy"
                                            />
                                        ))}
                                    </div>
                                ) : (
                                    <div className="no-photo">No photos available</div>
                                )}
                            </div>
                            <div className="match-info">
                                <h3>{match.profile.name}</h3>
                                <div className="match-score">
                                    {Math.round(match.matchScore * 100)}% Match
                                </div>
                                <div className="basic-info">
                                    <p>{match.profile.occupation}</p>
                                </div>
                                <div className="interests">
                                    {match.profile.interests.split(',').map((interest, i) => {
                                        // Take only first 3 words for display
                                        const words = interest.trim().split(' ');
                                        const displayText = words.slice(0, 3).join(' ');
                                        return (
                                            <span key={i} className="interest-tag">
                                                {displayText}
                                            </span>
                                        );
                                    })}
                                </div>
                                <p className="bio">{match.profile.aboutMe}</p>
                                <button 
                                    onClick={() => navigate(`/profile/${match.profile._id}`, { 
                                        state: { 
                                            profile: {
                                                ...match.profile,
                                                matchPercentage: Math.round(match.matchScore * 100)
                                            }
                                        } 
                                    })} 
                                    className="view-profile-btn"
                                >
                                    View Full Profile
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Matches;
