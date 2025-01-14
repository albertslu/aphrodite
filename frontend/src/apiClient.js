// Updated to use Cloudflare SSL

const fetchWithSSLBypass = async (endpoint, options = {}) => {
    const config = {
        apiUrl: ''  // Use relative path for proxy
    };
    const url = `${config.apiUrl}${endpoint}`;
    try {
        console.log('Making request to:', url, 'with options:', options);
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            mode: 'cors'
        });

        console.log('Response status:', response.status);
        const text = await response.text(); // Get response as text first
        console.log('Response text:', text);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
        }

        try {
            const data = JSON.parse(text); // Try to parse as JSON
            return data;
        } catch (e) {
            console.error('Failed to parse JSON:', e);
            throw new Error('Invalid JSON response from server');
        }
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
