import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './App.css';
import config from './config';

function ProfileCreation() {
    const navigate = useNavigate();
    const location = useLocation();
    const [isEditing, setIsEditing] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [photos, setPhotos] = useState([]);
    const [photoErrors, setPhotoErrors] = useState('');
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
        aboutMe: '',
        interests: '',
        relationshipGoals: ''
    });

    useEffect(() => {
        // Check if we're in edit mode
        const fetchProfile = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token) {
                    navigate('/login');
                    return;
                }

                const response = await fetch(`${config.apiUrl}/api/profile`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (response.ok) {
                    const profile = await response.json();
                    setIsEditing(true);
                    setFormFields({
                        name: profile.name || '',
                        gender: profile.gender || '',
                        sexualOrientation: profile.sexualOrientation || '',
                        age: profile.age || '',
                        height: profile.height || '',
                        ethnicity: profile.ethnicity || '',
                        occupation: profile.occupation || '',
                        location: profile.location || '',
                        education: profile.education || '',
                        aboutMe: profile.aboutMe || '',
                        interests: profile.interests || '',
                        relationshipGoals: profile.relationshipGoals || ''
                    });
                    
                    // Convert existing photos to the format expected by the component
                    if (profile.photos && profile.photos.length > 0) {
                        const existingPhotos = profile.photos.map(photo => ({
                            preview: photo.url,
                            existingUrl: photo.url // Flag to identify existing photos
                        }));
                        setPhotos(existingPhotos);
                    }
                }
            } catch (error) {
                console.error('Error fetching profile:', error);
            }
        };

        // Only fetch if we're in edit mode (accessed from preferences)
        if (location.pathname === '/edit-profile') {
            fetchProfile();
        }
    }, [navigate, location.pathname]);

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

        // Check each file's size and collect valid ones
        const validFiles = [];
        const largeFiles = [];
        
        files.forEach(file => {
            if (file.size > 10 * 1024 * 1024) {
                largeFiles.push(file.name);
            } else {
                validFiles.push(file);
            }
        });

        // If any files were too large, show specific error message
        if (largeFiles.length > 0) {
            const fileNames = largeFiles.join(', ');
            setPhotoErrors(`Please choose different photos. The following ${largeFiles.length === 1 ? 'file is' : 'files are'} too large (max 10MB): ${fileNames}`);
            e.target.value = ''; // Clear the file input
            return;
        }

        // Process valid files
        Promise.all(
            validFiles.map(file => {
                return new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        resolve({
                            file,
                            preview: reader.result
                        });
                    };
                    reader.readAsDataURL(file);
                });
            })
        ).then(newPhotos => {
            setPhotos(prevPhotos => [...prevPhotos, ...newPhotos]);
            setPhotoErrors('');
            e.target.value = ''; // Clear the file input
        });
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
            if (!token) {
                navigate('/login');
                return;
            }

            // Upload any new photos first
            const uploadedPhotos = [];
            for (let i = 0; i < photos.length; i++) {
                if (!photos[i].existingUrl) { // Only upload new photos
                    const formData = new FormData();
                    formData.append('photo', photos[i].file);

                    const uploadResponse = await fetch(`${config.apiUrl}/api/upload`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        body: formData
                    });

                    if (!uploadResponse.ok) {
                        throw new Error('Failed to upload photo');
                    }

                    const uploadResult = await uploadResponse.json();
                    uploadedPhotos.push({
                        url: uploadResult.url,
                        caption: '',
                        order: i
                    });
                } else {
                    // Keep existing photos
                    uploadedPhotos.push({
                        url: photos[i].existingUrl,
                        caption: '',
                        order: i
                    });
                }
            }

            // Create or update profile
            const profileData = {
                ...formFields,
                photos: uploadedPhotos
            };

            const url = isEditing ? 
                `${config.apiUrl}/api/profile` : 
                `${config.apiUrl}/api/profile/create`;

            const method = isEditing ? 'PUT' : 'POST';

            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(profileData)
            });

            if (!response.ok) {
                throw new Error('Failed to save profile');
            }

            // Navigate to preferences page after successful save
            navigate('/preferences');
        } catch (error) {
            console.error('Error saving profile:', error);
            setError('Failed to save profile. Please try again.');
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
            <h2>{isEditing ? 'Edit Your Profile' : 'Create Your Profile'}</h2>
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
                        placeholder="Occupation (optional)"
                        value={formFields.occupation}
                        onChange={handleInputChange}
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
                    <small>Upload up to 3 photos (max 10MB each)</small>
                    {photoErrors && <p className="error">{photoErrors}</p>}
                    <div className="photo-preview">
                        {photos.map((photo, index) => (
                            <div key={index} className="photo-item">
                                <img
                                    src={photo.preview}
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
                        placeholder="Tell us about yourself... (optional)"
                        value={formFields.aboutMe}
                        onChange={handleInputChange}
                    />
                    
                    <textarea
                        name="interests"
                        placeholder="What are your interests and hobbies? (optional)"
                        value={formFields.interests}
                        onChange={handleInputChange}
                    />
                    
                    <textarea
                        name="relationshipGoals"
                        placeholder="What are you looking for in a relationship? (optional)"
                        value={formFields.relationshipGoals}
                        onChange={handleInputChange}
                    />
                </div>

                <button type="submit" className="create-profile-btn" disabled={submitting}>
                    {submitting ? (isEditing ? 'Saving...' : 'Creating Profile...') : (isEditing ? 'Save Profile' : 'Create Profile')}
                </button>
                {error && <p className="error">{error}</p>}
            </form>
        </div>
    );
};

export default ProfileCreation;
