import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './App.css';

const FullProfile = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const profile = location.state?.profile;

    if (!profile) {
        return <div>Profile not found</div>;
    }

    const handleBack = () => {
        navigate(-1);
    };

    return (
        <div className="full-profile-container">
            <div className="full-profile-header">
                <button onClick={handleBack} className="back-btn">
                    Back to Matches
                </button>
                <button onClick={() => navigate('/preferences')} className="back-to-preferences">
                    Back to Preferences
                </button>
                <button onClick={() => {
                    localStorage.removeItem('token');
                    localStorage.removeItem('isAdmin');
                    navigate('/');
                }} className="logout-btn">
                    Logout
                </button>
            </div>

            <div className="full-profile-content">
                <h1>{profile.name}</h1>
                <div className="match-percentage">
                    <span>{profile.matchPercentage}% Match</span>
                </div>

                <div className="profile-occupation">{profile.occupation}</div>

                <div className="demographic-info">
                    <div className="demo-item">
                        <span className="demo-label">Age</span>
                        <span className="demo-value">{profile.age}</span>
                    </div>
                    <div className="demo-item">
                        <span className="demo-label">Location</span>
                        <span className="demo-value">{profile.location}</span>
                    </div>
                    <div className="demo-item">
                        <span className="demo-label">Height</span>
                        <span className="demo-value">{profile.height}</span>
                    </div>
                    <div className="demo-item">
                        <span className="demo-label">Ethnicity</span>
                        <span className="demo-value">{profile.ethnicity}</span>
                    </div>
                    <div className="demo-item">
                        <span className="demo-label">Education</span>
                        <span className="demo-value">{profile.education}</span>
                    </div>
                </div>

                {profile.photos && profile.photos.length > 0 && (
                    <div className="profile-photos">
                        {profile.photos.map((photo, index) => (
                            <img 
                                key={index}
                                src={`http://localhost:5000${photo}`}
                                alt={`${profile.name} photo ${index + 1}`}
                                className="profile-photo"
                            />
                        ))}
                    </div>
                )}

                <div className="profile-details">
                    <div className="bio-section">
                        <h2>About Me</h2>
                        <p>{profile.aboutMe}</p>
                    </div>

                    <div className="interests-section">
                        <h2>Interests</h2>
                        <p className="interests-text">{profile.interests}</p>
                    </div>

                    {profile.dealBreakers && (
                        <div className="dealbreakers-section">
                            <h2>Deal Breakers</h2>
                            <p>{profile.dealBreakers}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FullProfile;
