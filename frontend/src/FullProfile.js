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

                <div className="profile-photos">
                    {profile.photos.map((photo, index) => (
                        <img 
                            key={index}
                            src={photo}
                            alt={`${profile.name} photo ${index + 1}`}
                            className="profile-photo"
                        />
                    ))}
                </div>

                <div className="profile-details">
                    <div className="interests-section">
                        <h2>Interests</h2>
                        <div className="interests-tags">
                            {profile.interests.map((interest, index) => (
                                <span key={index} className="interest-tag">{interest}</span>
                            ))}
                        </div>
                    </div>

                    <div className="bio-section">
                        <h2>About Me</h2>
                        <p>{profile.bio}</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FullProfile;
