# app.py
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)

# Ensure punkt tokenizer is downloaded
nltk.download('punkt', quiet=True)

with open('data/profiles.json') as f:
    profiles = json.load(f)

def parse_description(description):
    tokens = word_tokenize(description.lower())
    return ' '.join(tokens)

def match_profiles(user_features, profiles):
    profile_features = [parse_description(profile['description']) for profile in profiles]
    vectorizer = CountVectorizer().fit_transform([user_features] + profile_features)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarities = cosine_matrix[0][1:]
    matched_profiles = sorted(zip(profiles, similarities), key=lambda x: x[1], reverse=True)
    return [profile for profile, _ in matched_profiles]

@app.route('/api/profiles', methods=['POST'])
def get_profiles():
    data = request.json
    description = data.get('description', '')
    tags = data.get('tags', [])

    user_features = parse_description(description)
    matched_profiles = match_profiles(user_features, profiles)
    
    return jsonify(matched_profiles)

if __name__ == '__main__':
    app.run(debug=True)
