const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');

// POST endpoint for profile matching
router.post('/match-profiles', async (req, res) => {
    const { prompt } = req.body;
    
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        const pythonProcess = spawn('python', [pythonScript, '--prompt', prompt]);

        let matchData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            matchData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            console.error(`Python Error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Python process exited with code:', code);
                console.error('Error output:', errorData);
                return res.status(500).json({ 
                    error: 'Profile matching failed',
                    details: errorData
                });
            }

            try {
                const matches = JSON.parse(matchData);
                res.json({ 
                    matches,
                    metadata: {
                        totalMatches: matches.length,
                        timestamp: new Date().toISOString()
                    }
                });
            } catch (parseError) {
                console.error('Error parsing Python output:', parseError);
                res.status(500).json({ 
                    error: 'Error parsing matching results',
                    details: parseError.message
                });
            }
        });

    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            details: error.message
        });
    }
});

// GET endpoint to check if matching service is available
router.get('/status', (req, res) => {
    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        const pythonProcess = spawn('python', [pythonScript, '--check']);

        pythonProcess.on('close', (code) => {
            res.json({
                status: code === 0 ? 'available' : 'unavailable',
                timestamp: new Date().toISOString()
            });
        });
    } catch (error) {
        res.status(500).json({ 
            status: 'unavailable',
            error: error.message
        });
    }
});

module.exports = router;
