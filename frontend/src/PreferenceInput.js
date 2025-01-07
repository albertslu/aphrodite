import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './App.css';

const PreferenceInput = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        partnerPreferences: '',
        ageRange: {
            min: '',
            max: ''
        },
        genderPreferences: [],
        ethnicityPreferences: [],
        locationPreference: ''
    });

    const [error, setError] = useState('');

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        if (name.includes('.')) {
            const [parent, child] = name.split('.');
            setFormData(prev => ({
                ...prev,
                [parent]: {
                    ...prev[parent],
                    [child]: value
                }
            }));
        } else {
            setFormData(prev => ({
                ...prev,
                [name]: value
            }));
        }
    };

    const handleCheckboxChange = (e, category) => {
        const { value, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [category]: checked 
                ? [...prev[category], value]
                : prev[category].filter(item => item !== value)
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const token = localStorage.getItem('token');
            if (!token) {
                navigate('/login');
                return;
            }

            // Update profile with partner preferences
            const response = await axios.put('http://localhost:5000/api/profile', 
                { partnerPreferences: JSON.stringify(formData) },
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            console.log('Preferences updated:', response.data);
            navigate('/browse'); // Navigate to profile browsing
        } catch (error) {
            console.error('Error updating preferences:', error.response?.data || error.message);
            setError(error.response?.data?.message || 'Error updating preferences');
        }
    };

    return (
        <div className="preference-input-container">
            <h2>Describe Your Ideal Match</h2>
            {error && <div className="error-message">{error}</div>}
            
            <form onSubmit={handleSubmit}>
                <div className="section">
                    <h3>Partner Description</h3>
                    <textarea
                        name="partnerPreferences"
                        placeholder="Describe your ideal partner (e.g., 'Looking for someone 27-30, male, with dark hair. Preferably Hispanic/Latin or White...')"
                        value={formData.partnerPreferences}
                        onChange={handleInputChange}
                        required
                        className="full-width-input"
                    />
                </div>

                <div className="section">
                    <h3>Age Preference</h3>
                    <div className="age-range">
                        <input
                            type="number"
                            name="ageRange.min"
                            placeholder="Min Age"
                            value={formData.ageRange.min}
                            onChange={handleInputChange}
                            min="18"
                            max="100"
                        />
                        <span>to</span>
                        <input
                            type="number"
                            name="ageRange.max"
                            placeholder="Max Age"
                            value={formData.ageRange.max}
                            onChange={handleInputChange}
                            min="18"
                            max="100"
                        />
                    </div>
                </div>

                <div className="section">
                    <h3>Gender Preference</h3>
                    <div className="checkbox-group">
                        {['male', 'female', 'non-binary', 'other'].map(gender => (
                            <label key={gender}>
                                <input
                                    type="checkbox"
                                    value={gender}
                                    checked={formData.genderPreferences.includes(gender)}
                                    onChange={(e) => handleCheckboxChange(e, 'genderPreferences')}
                                />
                                {gender.charAt(0).toUpperCase() + gender.slice(1)}
                            </label>
                        ))}
                    </div>
                </div>

                <div className="section">
                    <h3>Ethnicity Preference</h3>
                    <div className="checkbox-group">
                        {[
                            'Asian', 'Black', 'Hispanic/Latin', 'White', 
                            'Middle Eastern', 'Native American', 'Pacific Islander', 'Other'
                        ].map(ethnicity => (
                            <label key={ethnicity}>
                                <input
                                    type="checkbox"
                                    value={ethnicity}
                                    checked={formData.ethnicityPreferences.includes(ethnicity)}
                                    onChange={(e) => handleCheckboxChange(e, 'ethnicityPreferences')}
                                />
                                {ethnicity}
                            </label>
                        ))}
                    </div>
                </div>

                <div className="section">
                    <h3>Location Preference</h3>
                    <input
                        type="text"
                        name="locationPreference"
                        placeholder="Preferred location"
                        value={formData.locationPreference}
                        onChange={handleInputChange}
                        className="full-width-input"
                    />
                </div>

                <button type="submit" className="submit-btn">
                    Save Preferences
                </button>
            </form>
        </div>
    );
};

export default PreferenceInput;
