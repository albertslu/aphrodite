import requests

url = 'http://127.0.0.1:5000/api/profiles'
data = {
    'description': 'blonde hair blue eyes',
    'tags': ['blonde', 'blue eyes']
}

response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")  # Print raw response text
