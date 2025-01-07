import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import './App.css';

const Login = () => {
    const [formData, setFormData] = useState({
        username: '',
        password: ''
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
        setError(''); // Clear previous errors
        
        // Validate input
        if (!formData.username || !formData.password) {
            setError('Please enter both username and password');
            return;
        }

        try {
            console.log('Attempting login with:', formData.username);
            const response = await fetch('http://localhost:5000/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            console.log('Login response:', data);

            if (!response.ok) {
                throw new Error(data.message || 'Invalid credentials');
            }

            localStorage.setItem('token', data.token);
            localStorage.setItem('isAdmin', data.isAdmin);

            // If admin user, always go to profile creation
            if (data.isAdmin) {
                navigate('/create-profile');
                return;
            }

            // For regular users, check if they have a profile
            const profileResponse = await fetch('http://localhost:5000/api/profile', {
                headers: {
                    'Authorization': `Bearer ${data.token}`
                }
            });

            if (profileResponse.ok) {
                // User has a profile, redirect to preferences
                navigate('/preferences');
            } else if (profileResponse.status === 404) {
                // User doesn't have a profile, redirect to profile creation
                navigate('/create-profile');
            } else {
                throw new Error('Error checking profile status');
            }
        } catch (error) {
            console.error('Login error:', error);
            setError(error.message || 'Failed to login. Please try again.');
        }
    };

    return (
        <div className="login-container">
            <h2>Welcome to</h2>
            <h1 className="app-title-login">Aphrodite</h1>
            <p className="subtitle">Find your perfect match</p>
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
                <button type="submit" className="btn btn-primary">Login</button>
                <p className="signup-link">
                    Don't have an account? <Link to="/signup">Sign up</Link>
                </p>
            </form>
        </div>
    );
};

export default Login;
