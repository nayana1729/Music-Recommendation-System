from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from generate_dataset import generate_dataset  # Import the generate_dataset function from generate_dataset.py
from generate_dataset import get_audio_features
import ast
import numpy as np

app = Flask(__name__)

# Set up a folder to store uploaded files
UPLOAD_FOLDER = 'songs/'  # This is where MP3 files will be saved
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3'}

# Load pre-extracted dataset features
DATASET_PATH = 'data/dataset.csv'

def load_dataset():
    # Load the dataset from the CSV file
    dataset = pd.read_csv(DATASET_PATH)
    print(f"Original dataset shape: {dataset.shape}")

    # Store filenames separately for later use
    file_names = dataset['file_name'].values

    # Drop the 'file_name' column
    features = dataset.drop(columns=['file_name'])

    # Convert string representations of lists to actual lists
    for column in features.columns:
        features[column] = features[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Flatten the lists into individual rows, ensuring uniform length for all list-like entries
    max_length = max(features.applymap(lambda x: len(x) if isinstance(x, list) else 1).max())
    features = features.applymap(
        lambda x: np.pad(x, (0, max_length - len(x))) if isinstance(x, list) else np.array([x])
    )

    # Ensure all columns are now of uniform shape (list-like or scalar)
    feature_list = []
    for col in features.columns:
        feature_list.append(features[col].apply(lambda x: np.array(x) if isinstance(x, list) else x).values)

    # Stack all features into a 2D numpy array
    dataset_features = np.vstack(feature_list).T  # Transpose to match the original shape (47, 39)
    
    print(f"Processed features shape: {dataset_features.shape}")
    
    return file_names, dataset_features

# Load dataset when the app starts
file_names, dataset_features = load_dataset()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to process features: Convert string to list if needed
def process_features(features):
    # Convert the string to an actual list using ast.literal_eval if it's a string
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
        # Save the uploaded file to the 'songs' directory
        filename = os.path.join(app.config['UPLOAD_FOLDER'], song_file.filename)
        song_file.save(filename)

        # Call the generate_dataset function to process the uploaded song and generate the dataset
        try:
            generate_dataset(filename)  # Pass the uploaded file to generate the dataset
        except Exception as e:
            return f"Error generating dataset: {e}"

        # Load the dataset after generation
        file_names, dataset_features = load_dataset()

        # Extract features from the uploaded song (as before)
        song_features = get_audio_features(filename)  # Extract features (MFCCs) from the uploaded song
        
        if song_features is None:
            return f"Error: Could not extract features from {filename}. Please try again."

        # Process features (ensure numeric and handle empty arrays)
        song_features = process_features(song_features)
        dataset_features = [process_features(feature) for feature in dataset_features]
        dataset_features = np.array(dataset_features)
        
        # Check if features are valid and reshape for cosine similarity
        if song_features.size == 0 or len(dataset_features) == 0 or any(len(f) == 0 for f in dataset_features):
            return "Error: Feature extraction failed or empty feature array."

        # Ensure features are numeric and have the same number of columns
        # Convert all features in the dataset to numeric values
        dataset_features = np.array([pd.to_numeric(f, errors='coerce') for f in dataset_features])
        
        # Remove non-numeric rows (if any)
        dataset_features = dataset_features[~np.isnan(dataset_features).any(axis=1)]

        if song_features.shape[0] != dataset_features.shape[1]:
            return f"Feature mismatch: song_features shape {song_features.shape} and dataset_features shape {dataset_features.shape}"

        # Reshape song_features for cosine similarity (ensure 2D)
        song_features = song_features.reshape(1, -1)  # Reshape to 2D (1, N)

        # Calculate cosine similarity between uploaded song features and dataset features
        try:
            similarities = cosine_similarity(song_features, dataset_features)
        except ValueError as e:
            return f"Error calculating cosine similarity: {e}"

        # Get indices of the top 3 most similar songs
        top_indices = similarities[0].argsort()[-3:][::-1]
        recommendations = [file_names[i] for i in top_indices]

        # Render the results page with the recommendations
        return render_template('results.html', recommendations=recommendations)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
