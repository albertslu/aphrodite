// src/PreferenceInput.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

const PreferenceInput = () => {
    const [description, setDescription] = useState('');
    const [tags, setTags] = useState([]);
    const [currentTag, setCurrentTag] = useState('');
    const navigate = useNavigate();

    const handleDescriptionChange = (e) => setDescription(e.target.value);
    const handleCurrentTagChange = (e) => setCurrentTag(e.target.value);
    const handleTagsAdd = () => {
        if (currentTag && !tags.includes(currentTag)) {
            setTags([...tags, currentTag]);
            setCurrentTag('');
        }
    };
    const handleTagsKeyPress = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleTagsAdd();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        // Send data to backend
        const response = await fetch('http://127.0.0.1:5000/api/profiles', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ description, tags })
        });
        const profiles = await response.json();
        // Navigate to profile browsing screen with the profiles data
        navigate('/browse', { state: { profiles } });
    };

    return (
        <div>
            <h1>Describe Your Ideal Match</h1>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Describe your ideal match:</label>
                    <textarea className="full-width-input" value={description} onChange={handleDescriptionChange} required />
                </div>
                <div className="form-group">
                    <label>Preferences:</label>
                    <div className="input-group">
                        <input type="text" value={currentTag} onChange={handleCurrentTagChange} onKeyPress={handleTagsKeyPress} placeholder="Add a preference tag" />
                        <button type="button" onClick={handleTagsAdd}>Add</button>
                    </div>
                    <div>
                        {tags.map((tag, index) => (
                            <span key={index}>
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>
                <button type="submit">Submit</button>
            </form>
        </div>
    );
};

export default PreferenceInput;
