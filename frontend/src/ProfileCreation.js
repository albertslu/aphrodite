import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';
import config from './config';

const ProfileCreation = () => {
    const navigate = useNavigate();
    const [formFields, setFormFields] = useState({
        name: '',
        gender: '',
        sexualOrientation: '',
        age: '',
        height: '',
        ethnicity: '',
        occupation: '',
        location: '',
        education: '',
        photos: [],
        aboutMe: '',
        interests: '',
        relationshipGoals: '',
        idealDate: '',
        partnerPreferences: ''
    });

    const [photos, setPhotos] = useState([]);
    const [photoErrors, setPhotoErrors] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');

    const handleInputChange = (e) => {
        setFormFields({
            ...formFields,
            [e.target.name]: e.target.value
        });
    };

    const handlePhotoChange = (e) => {
        const files = Array.from(e.target.files);
        
        // Check if adding new files would exceed the 3 photo limit
        if (photos.length + files.length > 3) {
            setPhotoErrors('You can only upload up to 3 photos');
            return;
        }

        // Check file sizes
        if (files.some(file => file.size > 5 * 1024 * 1024)) {
            setPhotoErrors('Each photo must be less than 5MB');
            return;
        }

        setPhotoErrors('');
        // Append new photos to existing ones
        setPhotos(prevPhotos => [...prevPhotos, ...files].slice(0, 3));
    };

    const removePhoto = (index) => {
        setPhotos(prevPhotos => prevPhotos.filter((_, i) => i !== index));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSubmitting(true);
        setError('');

        try {
            const token = localStorage.getItem('token');
            const isAdmin = localStorage.getItem('isAdmin') === 'true';

            // First, upload all photos
            const uploadedPhotos = [];
            for (let i = 0; i < photos.length; i++) {
                const formData = new FormData();
                formData.append('photo', photos[i]);

                const uploadResponse = await fetch(`${config.apiUrl}/api/upload`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    },
                    body: formData
                });

                if (!uploadResponse.ok) {
                    throw new Error(`Failed to upload photo ${i + 1}`);
                }

                const uploadData = await uploadResponse.json();
                uploadedPhotos.push({
                    url: uploadData.url,
                    order: i + 1
                });
            }

            // Convert age to number and prepare profile data
            const profileData = {
                ...formFields,
                age: Number(formFields.age),
                photos: uploadedPhotos  // Add the uploaded photos to profile data
            };

            console.log('Sending profile data:', profileData);

            const response = await fetch(`${config.apiUrl}/api/profile`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(profileData)
            });

            const data = await response.json();
            console.log('Profile creation response:', data);
            
            if (!response.ok) {
                if (data.errors) {
                    // Format validation errors nicely
                    const errorMessages = Object.entries(data.errors)
                        .map(([field, error]) => `${field}: ${error.message}`)
                        .join('\n');
                    throw new Error(errorMessages);
                }
                throw new Error(data.message || data.error || 'Failed to create profile');
            }

            // For admin users, redirect to preferences with a success message
            // For regular users, just redirect to preferences
            navigate('/preferences', { 
                state: { 
                    message: isAdmin ? 'Profile created successfully. You can create another profile or set preferences.' : undefined 
                } 
            });
        } catch (error) {
            console.error('Profile creation error:', error);
            setError(error.message || 'Failed to create profile');
        } finally {
            setSubmitting(false);
        }
    };

    const renderMatchedProfiles = (matches) => {
        return matches.map((match, index) => (
            <div key={index} className="profile-card">
                <h2>{match.profile.name}</h2>
                <div className={`match-score ${match.matchScore >= 70 ? 'high-match' : match.matchScore >= 50 ? 'medium-match' : 'low-match'}`}>
                    {Math.round(match.matchScore)}% Match
                </div>
                
                <h3>{match.profile.occupation}</h3>
                <p>{match.profile.bio}</p>
                
                <div className="interests">
                    {match.profile.interests.split(',').map((interest, index) => (
                        <span key={index} className="interest-tag">
                            {interest.trim()}
                        </span>
                    ))}
                </div>
                
                {match.profile.photos && match.profile.photos.length > 0 && (
                    <div className="photos">
                        {match.profile.photos.map((photo, index) => (
                            <img 
                                key={index} 
                                src={`/uploads/${photo.split('/').pop()}`}
                                alt={`${match.profile.name}'s photo ${index + 1}`}
                            />
                        ))}
                    </div>
                )}
            </div>
        ));
    };

    function MatchScore({ score }) {
        const displayScore = typeof score === 'number' ? `${Math.round(score)}% Match` : 'No Match';
        return (
            <div className={`match-score ${score >= 70 ? 'high-match' : score >= 50 ? 'medium-match' : 'low-match'}`}>
                {displayScore}
            </div>
        );
    }

    function ProfileCard({ profile, score }) {
        return (
            <div className="profile-card">
                <h2>{profile.name}</h2>
                <MatchScore score={score} />
                
                <h3>{profile.occupation}</h3>
                
                <p>{profile.aboutMe}</p>
                
                <div className="interests">
                    {profile.interests.split(',').map((interest, index) => (
                        <span key={index} className="interest-tag">
                            {interest.trim()}
                        </span>
                    ))}
                </div>
                
                {profile.photos && profile.photos.length > 0 && (
                    <div className="photos">
                        {profile.photos.map((photo, index) => (
                            <img key={index} src={photo.url} alt={`${profile.name}'s photo ${index + 1}`} />
                        ))}
                    </div>
                )}
            </div>
        );
    }

    return (
        <div className="profile-creation-container">
            <h2>Create Your Profile</h2>
            <form onSubmit={handleSubmit}>
                <div className="section">
                    <h3>Basic Information</h3>
                    <input
                        type="text"
                        name="name"
                        placeholder="Name"
                        value={formFields.name}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <select
                        name="gender"
                        value={formFields.gender}
                        onChange={handleInputChange}
                        required
                    >
                        <option value="">Select Gender</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="non-binary">Non-binary</option>
                        <option value="other">Other</option>
                    </select>

                    <select
                        name="sexualOrientation"
                        value={formFields.sexualOrientation}
                        onChange={handleInputChange}
                        required
                    >
                        <option value="">Select Sexual Orientation</option>
                        <option value="straight">Straight</option>
                        <option value="gay">Gay</option>
                        <option value="lesbian">Lesbian</option>
                        <option value="bisexual">Bisexual</option>
                        <option value="pansexual">Pansexual</option>
                        <option value="other">Other</option>
                    </select>

                    <input
                        type="number"
                        name="age"
                        placeholder="Age"
                        value={formFields.age}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="height"
                        placeholder="Height"
                        value={formFields.height}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="ethnicity"
                        placeholder="Ethnicity"
                        value={formFields.ethnicity}
                        onChange={handleInputChange}
                    />
                    
                    <input
                        type="text"
                        name="occupation"
                        placeholder="Occupation"
                        value={formFields.occupation}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="location"
                        placeholder="Location"
                        value={formFields.location}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="education"
                        placeholder="Education"
                        value={formFields.education}
                        onChange={handleInputChange}
                    />
                </div>

                <div className="section">
                    <h3>Photos</h3>
                    <input
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handlePhotoChange}
                        required={photos.length === 0}
                    />
                    <small>Upload up to 3 photos (max 5MB each)</small>
                    {photoErrors && <p className="error">{photoErrors}</p>}
                    <div className="photo-preview">
                        {photos.map((photo, index) => (
                            <div key={index} className="photo-item">
                                <img
                                    src={URL.createObjectURL(photo)}
                                    alt={`Preview ${index + 1}`}
                                />
                                <button 
                                    type="button" 
                                    className="remove-photo"
                                    onClick={() => removePhoto(index)}
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="section">
                    <h3>About You</h3>
                    <textarea
                        name="aboutMe"
                        placeholder="Tell us about yourself..."
                        value={formFields.aboutMe}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <textarea
                        name="interests"
                        placeholder="What are your interests and hobbies?"
                        value={formFields.interests}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <textarea
                        name="relationshipGoals"
                        placeholder="What are you looking for in a relationship? (Long-term, casual, friendship, etc.)"
                        value={formFields.relationshipGoals}
                        onChange={handleInputChange}
                        required
                    />
                </div>

                <button type="submit" className="create-profile-btn" disabled={submitting}>
                    {submitting ? 'Creating Profile...' : 'Create Profile'}
                </button>
                {error && <p className="error">{error}</p>}
            </form>
        </div>
    );
};

export default ProfileCreation;
