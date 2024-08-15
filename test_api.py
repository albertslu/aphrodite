import requests

# URL of the API endpoint
url = 'http://127.0.0.1:5000/api/profiles'

# Sample data to send in the POST request
data = {
    'description': 'blonde hair blue eyes',
    'tags': ['blonde', 'blue eyes']
}

# Sending a POST request to the API
response = requests.post(url, json=data)

# Print the status code and response content
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
