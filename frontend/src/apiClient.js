// Updated to use Cloudflare SSL

const fetchWithSSLBypass = async (endpoint, options = {}) => {
    const config = {
        apiUrl: ''  // Use relative path for proxy
    };
    const url = `${config.apiUrl}${endpoint}`;
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            mode: 'cors'
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.message || 'API request failed');
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
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData)
    }),
    update: (formData, token) => fetchWithSSLBypass('/api/profile', {
        method: 'PUT',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData)
    }),
    get: (token) => fetchWithSSLBypass('/api/profile', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
};
