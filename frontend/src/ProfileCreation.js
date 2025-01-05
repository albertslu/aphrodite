import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './App.css';

const ProfileCreation = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
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
        idealDate: '',
        partnerPreferences: ''
    });

    const [photos, setPhotos] = useState([]);
    const [photoErrors, setPhotoErrors] = useState('');

    const handleInputChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handlePhotoChange = (e) => {
        const files = Array.from(e.target.files);
        
        if (files.length > 3) {
            setPhotoErrors('You can only upload up to 3 photos');
            return;
        }

        if (files.some(file => file.size > 5 * 1024 * 1024)) {
            setPhotoErrors('Each photo must be less than 5MB');
            return;
        }

        setPhotoErrors('');
        setPhotos(files);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                navigate('/login');
                return;
            }

            // Upload photos first
            const photoUrls = [];
            for (let i = 0; i < photos.length; i++) {
                const formData = new FormData();
                formData.append('photo', photos[i]);
                
                const photoRes = await axios.post('http://localhost:5000/api/upload', formData, {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'multipart/form-data'
                    }
                });
                
                photoUrls.push({
                    url: photoRes.data.url,
                    order: i + 1
                });
            }

            // Create profile with photo URLs
            const profileData = {
                ...formData,
                photos: photoUrls
            };

            const response = await axios.post('http://localhost:5000/api/profile', profileData, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            console.log('Profile created:', response.data);
            navigate('/preferences'); // Navigate to preferences page after profile creation
        } catch (error) {
            console.error('Error creating profile:', error.response?.data || error.message);
            alert(error.response?.data?.message || 'Error creating profile');
        }
    };

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
                        value={formData.name}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <select
                        name="gender"
                        value={formData.gender}
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
                        value={formData.sexualOrientation}
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
                        value={formData.age}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="height"
                        placeholder="Height"
                        value={formData.height}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="ethnicity"
                        placeholder="Ethnicity"
                        value={formData.ethnicity}
                        onChange={handleInputChange}
                    />
                    
                    <input
                        type="text"
                        name="occupation"
                        placeholder="Occupation"
                        value={formData.occupation}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="location"
                        placeholder="Location"
                        value={formData.location}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <input
                        type="text"
                        name="education"
                        placeholder="Education"
                        value={formData.education}
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
                        required
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
                            </div>
                        ))}
                    </div>
                </div>

                <div className="section">
                    <h3>About You</h3>
                    <textarea
                        name="aboutMe"
                        placeholder="Tell us about yourself..."
                        value={formData.aboutMe}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <textarea
                        name="interests"
                        placeholder="What are your interests and hobbies?"
                        value={formData.interests}
                        onChange={handleInputChange}
                        required
                    />
                    
                    <textarea
                        name="idealDate"
                        placeholder="Describe your ideal date..."
                        value={formData.idealDate}
                        onChange={handleInputChange}
                        required
                    />
                </div>

                <button type="submit" className="create-profile-btn">
                    Create Profile
                </button>
            </form>
        </div>
    );
};

export default ProfileCreation;
