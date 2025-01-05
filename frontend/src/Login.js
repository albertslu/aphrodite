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
        try {
            const response = await fetch('http://localhost:5000/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.message || 'Invalid credentials');
            }

            const data = await response.json();
            localStorage.setItem('token', data.token);
            navigate('/create-profile');
        } catch (error) {
            setError('Please enter both username and password');
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
