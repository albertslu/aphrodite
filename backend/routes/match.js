const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

// POST endpoint for profile matching
router.post('/match-profiles', async (req, res) => {
    const { prompt } = req.body;
    
    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        // Use python on Windows, and python3 on other platforms, and add debug flag
        const pythonExecutable = os.platform() === 'win32' ? 'python' : 'python3';
        const pythonProcess = spawn(pythonExecutable, [
            pythonScript,
            '--prompt', prompt,
            '--debug'  // Add debug flag
        ]);

        let matchData = '';
        let errorData = '';

        pythonProcess.stdout.on('data', (data) => {
            console.log('Python output:', data.toString());
            matchData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error('Python error:', data.toString());
            errorData += data.toString();
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            return res.status(500).json({ 
                error: 'Failed to start matcher process',
                details: error.message
            });
        });

        pythonProcess.on('close', (code) => {
            console.log('Python process exited with code:', code);
            if (code !== 0) {
                console.error('Error output:', errorData);
                return res.status(500).json({ 
                    error: 'Profile matching failed',
                    details: errorData
                });
            }

            try {
                // Try to parse the output as JSON
                const matches = JSON.parse(matchData);
                
                // If we got an empty array, return a more specific message
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
                console.error('Error parsing Python output:', parseError);
                console.error('Raw output:', matchData);
                res.status(500).json({ 
                    error: 'Invalid matcher output',
                    details: parseError.message,
                    rawOutput: matchData
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
