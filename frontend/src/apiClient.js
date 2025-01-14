// Updated to use Cloudflare SSL

const fetchWithSSLBypass = async (endpoint, options = {}) => {
    try {
        console.log('Making request to:', endpoint, 'with options:', options);
        const response = await fetch(endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            mode: 'cors'
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: 'An error occurred' }));
            throw new Error(error.message || `HTTP error! status: ${response.status}`);
        }

        return response.json();
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
};

export const auth = {
    signup: (formData) => fetchWithSSLBypass('/api/auth/signup', {
        method: 'POST',
        body: JSON.stringify(formData)
    }),
    login: (formData) => fetchWithSSLBypass('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(formData)
    })
};

export const profile = {
    create: (formData, token) => fetchWithSSLBypass('/api/profile', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify(formData)
    }),
    update: (formData, token) => fetchWithSSLBypass('/api/profile', {
        method: 'PUT',
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify(formData)
    }),
    get: (token) => fetchWithSSLBypass('/api/profile', {
        headers: { Authorization: `Bearer ${token}` }
    })
};
