#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from generate_dataset import generate_dataset
from generate_dataset import get_audio_features
import ast
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'songs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}

DATASET_PATH = 'data/dataset.csv'

def load_dataset():
    # Generate dataset if the dataset file is empty
    if not os.path.isfile(DATASET_PATH) or os.stat(DATASET_PATH).st_size == 0:
        generate_dataset()

    # Load the dataset from the CSV file
    dataset = pd.read_csv(DATASET_PATH)

    file_names = dataset['file_name'].values

    features = dataset.drop(columns=['file_name'])

    # Convert string representations of lists to actual lists
    for column in features.columns:
        features[column] = features[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Flatten the lists into individual rows
    max_length = max(features.applymap(lambda x: len(x) if isinstance(x, list) else 1).max())
    features = features.applymap(
        lambda x: np.pad(x, (0, max_length - len(x))) if isinstance(x, list) else np.array([x])
    )

    # Ensure columns are uniform
    feature_list = []
    for col in features.columns:
        feature_list.append(features[col].apply(lambda x: np.array(x) if isinstance(x, list) else x).values)

    # Stack features into a 2D numpy array
    dataset_features = np.vstack(feature_list).T

    return file_names, dataset_features

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to process features: Convert string to list if needed
def process_features(features):
    if isinstance(features, str):
        features = ast.literal_eval(features)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'song_file' not in request.files:
        return redirect(request.url)
    song_file = request.files['song_file']
    if song_file.filename == '':
        return redirect(request.url)
    if song_file and allowed_file(song_file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], song_file.filename)
        song_file.save(filename)

        # Load the dataset
        file_names, dataset_features = load_dataset()

        # Process the uploaded song and generate the dataset
        try:
            generate_dataset(filename)
        except Exception as e:
            return f"Error generating dataset: {e}"

        # Extract features from the uploaded song
        song_features = get_audio_features(filename)
        if song_features is None:
            return f"Error: Could not extract features from {filename}. Please try again."

        song_features = process_features(song_features)
        dataset_features = [process_features(feature) for feature in dataset_features]
        dataset_features = np.array(dataset_features)

        # Check if features are valid
        if song_features.size == 0 or len(dataset_features) == 0 or any(len(f) == 0 for f in dataset_features):
            return "Error: Feature extraction failed or empty feature array."

        if song_features.shape[0] != dataset_features.shape[1]:
            return f"Feature mismatch: song_features shape {song_features.shape} and dataset_features shape {dataset_features.shape}"

        # Reshape song_features for cosine similarity
        song_features = song_features.reshape(1, -1)

        # Standardize features
        scaler = StandardScaler()

        # Combine dataset_features and song_features to scale together
        features_combined = np.vstack([dataset_features, song_features])

        # Apply scaling to the combined features
        features_scaled = scaler.fit_transform(features_combined)

        # Separate back the dataset features and song features
        dataset_features_scaled = features_scaled[:-1]
        song_features_scaled = features_scaled[-1:]

        # Reshape for cosine similarity calculation
        song_features_scaled = song_features_scaled.reshape(1, -1)

        try:
            similarities = cosine_similarity(song_features_scaled, dataset_features_scaled)
        except ValueError as e:
            return f"Error calculating cosine similarity: {e}"

        tolerance = 1e-5
        top_indices = similarities[0].argsort()[-5:][::-1]

        # Exclude the song with similarity close to 1 (uploaded song itself) from the recommendations
        top_indices = [i for i in top_indices if abs(similarities[0][i] - 1) > tolerance]
        top_indices = top_indices[:5]

        recommendations = [file_names[i] for i in top_indices]
        
        return render_template('results.html', recommendations=recommendations)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
