const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

// Determine Python path based on environment
const PYTHON_PATH = os.platform() === 'win32' 
    ? 'C:\\Users\\bert\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'
    : 'python3';

// Track if response has been sent
let hasResponded = false;

// Helper function to send response only once
const sendResponseOnce = (res, statusCode, data) => {
    if (!hasResponded) {
        hasResponded = true;
        res.status(statusCode).json(data);
    }
};

// POST endpoint for profile matching
router.post('/match-profiles', async (req, res) => {
    const { prompt } = req.body;
    hasResponded = false; // Reset for new request
    
    if (!prompt) {
        return sendResponseOnce(res, 400, { error: 'Prompt is required' });
    }

    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        console.log('Running matcher with script:', pythonScript);
        console.log('Using Python path:', PYTHON_PATH);
        
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
            sendResponseOnce(res, 500, { 
                error: 'Failed to start matcher process',
                details: error.message
            });
        });

        matcherProcess.on('close', (code) => {
            if (code !== 0) {
                console.error('Matcher process failed with code:', code);
                console.error('Error output:', matcherError);
                return sendResponseOnce(res, 500, { 
                    error: 'Profile matching failed',
                    details: matcherError || 'Unknown error occurred'
                });
            }

            try {
                const matches = JSON.parse(matcherData);
                if (!matches || matches.length === 0) {
                    return sendResponseOnce(res, 404, {
                        error: 'No matches found',
                        message: 'No profiles matched your preferences. Try broadening your search criteria.'
                    });
                }

                sendResponseOnce(res, 200, { 
                    matches,
                    metadata: {
                        totalMatches: matches.length,
                        timestamp: new Date().toISOString()
                    }
                });
            } catch (parseError) {
                console.error('Error parsing matcher output:', parseError);
                console.error('Raw output:', matcherData);
                sendResponseOnce(res, 500, { 
                    error: 'Invalid matcher output',
                    details: parseError.message
                });
            }
        });
    } catch (error) {
        console.error('Route error:', error);
        sendResponseOnce(res, 500, { 
            error: 'Internal server error',
            details: error.message
        });
    }
});

// GET endpoint to check if matching service is available
router.get('/status', (req, res) => {
    try {
        const pythonScript = path.join(__dirname, '..', 'utils', 'hybrid_profile_matcher.py');
        console.log('Starting Python process with:', PYTHON_PATH, pythonScript);
        
        const pythonProcess = spawn(PYTHON_PATH, [
            pythonScript,
            '--prompt', 'test status',
            '--debug'
        ]);

        let processOutput = '';
        let processError = '';

        pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log('Python stdout:', output);
            processOutput += output;
        });

        pythonProcess.stderr.on('data', (data) => {
            const error = data.toString();
            console.error('Python stderr:', error);
            processError += error;
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            res.status(500).json({
                status: 'unavailable',
                error: `Failed to start process: ${error.message}`,
                pythonPath: PYTHON_PATH
            });
        });

        pythonProcess.on('close', (code) => {
            console.log('Python process exited with code:', code);
            console.log('Final output:', processOutput);
            console.log('Final error:', processError);
            
            res.json({
                status: code === 0 ? 'available' : 'unavailable',
                timestamp: new Date().toISOString(),
                pythonPath: PYTHON_PATH,
                exitCode: code,
                error: processError || undefined,
                output: processOutput || undefined
            });
        });
    } catch (error) {
        console.error('Status check error:', error);
        res.status(500).json({ 
            status: 'unavailable',
            error: error.message,
            pythonPath: PYTHON_PATH
        });
    }
});

module.exports = router;
