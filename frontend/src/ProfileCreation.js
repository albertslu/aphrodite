// src/ProfileCreation.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const ProfileCreation = () => {
    const [formData, setFormData] = useState({
        name: '',
        age: '',
        height: '',
        ethnicity: '',
        occupation: '',
        location: '',
        education: '',
        photos: [],
        aboutMe: '',
        interests: '',
        idealDate: ''
    });
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handlePhotoChange = (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 3) {
            setError('Maximum 3 photos allowed');
            return;
        }
        setFormData(prev => ({
            ...prev,
            photos: files
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const formDataToSend = new FormData();
            Object.keys(formData).forEach(key => {
                if (key === 'photos') {
                    formData[key].forEach(photo => {
                        formDataToSend.append('photos', photo);
                    });
                } else {
                    formDataToSend.append(key, formData[key]);
                }
            });

            // TODO: Add API endpoint
            const response = await fetch('http://localhost:5000/api/profiles', {
                method: 'POST',
                body: formDataToSend
            });

            if (!response.ok) {
                throw new Error('Failed to create profile');
            }

            const data = await response.json();
            const profile = data;
            await profile.save();
            
            // Navigate to preferences page after profile creation
            navigate('/preferences');
        } catch (error) {
            setError(error.message);
            console.error('Error creating profile:', error);
        }
    };

    return (
        <div className="profile-creation">
            <h2>Create Your Profile</h2>
            {error && <div className="error">{error}</div>}
            <form onSubmit={handleSubmit}>
                <div className="form-section">
                    <h3>Basic Information</h3>
                    <div className="form-group">
                        <input
                            type="text"
                            name="name"
                            value={formData.name}
                            onChange={handleInputChange}
                            placeholder="Full Name"
                            className="form-control"
                            required
                        />
                    </div>
                    <div className="form-row">
                        <div className="form-group half">
                            <input
                                type="number"
                                name="age"
                                value={formData.age}
                                onChange={handleInputChange}
                                placeholder="Age"
                                className="form-control"
                                required
                            />
                        </div>
                        <div className="form-group half">
                            <input
                                type="text"
                                name="height"
                                value={formData.height}
                                onChange={handleInputChange}
                                placeholder="Height"
                                className="form-control"
                                required
                            />
                        </div>
                    </div>
                    <div className="form-group">
                        <input
                            type="text"
                            name="ethnicity"
                            value={formData.ethnicity}
                            onChange={handleInputChange}
                            placeholder="Ethnicity"
                            className="form-control"
                        />
                    </div>
                    <div className="form-group">
                        <input
                            type="text"
                            name="occupation"
                            value={formData.occupation}
                            onChange={handleInputChange}
                            placeholder="Occupation"
                            className="form-control"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <input
                            type="text"
                            name="location"
                            value={formData.location}
                            onChange={handleInputChange}
                            placeholder="Location"
                            className="form-control"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <input
                            type="text"
                            name="education"
                            value={formData.education}
                            onChange={handleInputChange}
                            placeholder="Education"
                            className="form-control"
                        />
                    </div>
                </div>

                <div className="form-section">
                    <h3>Photos</h3>
                    <div className="form-group">
                        <input
                            type="file"
                            onChange={handlePhotoChange}
                            accept="image/*"
                            multiple
                            max="3"
                            className="form-control"
                            required
                        />
                        <small>Upload up to 3 photos</small>
                    </div>
                </div>

                <div className="form-section">
                    <h3>About You</h3>
                    <div className="form-group">
                        <textarea
                            name="aboutMe"
                            value={formData.aboutMe}
                            onChange={handleInputChange}
                            placeholder="Tell us about yourself..."
                            className="form-control"
                            rows="4"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <textarea
                            name="interests"
                            value={formData.interests}
                            onChange={handleInputChange}
                            placeholder="What are your interests and hobbies?"
                            className="form-control"
                            rows="4"
                            required
                        />
                    </div>
                    <div className="form-group">
                        <textarea
                            name="idealDate"
                            value={formData.idealDate}
                            onChange={handleInputChange}
                            placeholder="Describe your ideal date..."
                            className="form-control"
                            rows="4"
                            required
                        />
                    </div>
                </div>

                <button type="submit" className="btn btn-primary">Create Profile</button>
            </form>
        </div>
    );
};

export default ProfileCreation;
