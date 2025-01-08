const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

// Use full Python path
const PYTHON_PATH = 'C:\\Users\\bert\\AppData\\Local\\Programs\\Python\\Python312\\python.exe';

// POST endpoint for profile matching
router.post('/match-profiles', async (req, res) => {
    const { prompt } = req.body;
    
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        console.log('Running matcher with script:', pythonScript);
        
        const matcherProcess = spawn(PYTHON_PATH, [
            pythonScript,
            '--prompt', prompt,
            '--debug'
        ]);

        let matcherData = '';
        let matcherError = '';

        matcherProcess.stdout.on('data', (data) => {
            console.log('Matcher output:', data.toString());
            matcherData += data.toString();
        });

        matcherProcess.stderr.on('data', (data) => {
            console.error('Matcher error:', data.toString());
            matcherError += data.toString();
        });

        matcherProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            return res.status(500).json({ 
                error: 'Failed to start matcher process',
                details: error.message
            });
        });

        matcherProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Matcher process failed with code:', code);
                console.error('Error output:', matcherError);
                return res.status(500).json({ 
                    error: 'Profile matching failed',
                    details: matcherError
                });
            }

            try {
                const matches = JSON.parse(matcherData);
                if (!matches || matches.length === 0) {
                    return res.status(404).json({
                        error: 'No matches found',
                        message: 'No profiles matched your preferences. Try broadening your search criteria.'
                    });
                }

                res.json({ 
                    matches,
                    metadata: {
                        totalMatches: matches.length,
                        timestamp: new Date().toISOString()
                    }
                });
            } catch (parseError) {
                console.error('Error parsing matcher output:', parseError);
                console.error('Raw output:', matcherData);
                res.status(500).json({ 
                    error: 'Invalid matcher output',
                    details: parseError.message,
                    rawOutput: matcherData
                });
            }
        });
    } catch (error) {
        console.error('Route error:', error);
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
        const pythonExecutable = os.platform() === 'win32' ? 'python' : 'python3';
        const pythonProcess = spawn(pythonExecutable, [pythonScript, '--check']);

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
