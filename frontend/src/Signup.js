import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './App.css';

const Signup = () => {
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        phoneNumber: ''
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

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const url = 'https://api.aphroditeai.org/api/auth/signup';
            console.log('Making request to:', url);
            console.log('With data:', formData);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            console.log('Full response:', {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers.entries())
            });

            const responseText = await response.text();
            console.log('Raw response text:', responseText);

            let data;
            try {
                data = JSON.parse(responseText);
                console.log('Parsed response data:', data);
            } catch (parseError) {
                console.error('JSON parse error:', parseError);
                throw new Error('Server response was not valid JSON');
            }
            
            if (!response.ok) {
                throw new Error(data.message || `Server error: ${response.status}`);
            }

            localStorage.setItem('token', data.token);
            navigate('/create-profile');
        } catch (error) {
            console.error('Signup error:', error);
            setError(error.message || 'Error creating account. Please try again.');
        }
    };

    return (
        <div className="login-container">
            <h2>Create Account</h2>
            <p className="subtitle">Join Aphrodite today</p>
            {error && <div className="error">{error}</div>}
            <form onSubmit={handleSubmit} className="login-form">
                <div className="form-group">
                    <input
                        type="text"
                        name="username"
                        value={formData.username}
                        onChange={handleInputChange}
                        placeholder="Username"
                        className="form-control"
                        required
                    />
                </div>
                <div className="form-group">
                    <input
                        type="password"
                        name="password"
                        value={formData.password}
                        onChange={handleInputChange}
                        placeholder="Password"
                        className="form-control"
                        required
                    />
                </div>
                <div className="form-group">
                    <input
                        type="tel"
                        name="phoneNumber"
                        value={formData.phoneNumber}
                        onChange={handleInputChange}
                        placeholder="Phone Number (Optional)"
                        className="form-control"
                    />
                </div>
                <button type="submit" className="btn btn-primary">Sign Up</button>
                <p className="signup-link">
                    Already have an account? <Link to="/">Log in</Link>
                </p>
            </form>
        </div>
    );
};

export default Signup;
