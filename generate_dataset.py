import os
import pandas as pd
import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import logging

# Path where your audio files are located
AUDIO_FILES_PATH = r"C:\Users\nayan\Downloads\music_recommendation_app\songs"
# Path to save the generated dataset CSV
DATA_FOLDER = r"C:\Users\nayan\Downloads\music_recommendation_app\data"
DATASET_PATH = os.path.join(DATA_FOLDER, "dataset.csv")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_audio_features(file_path):
    """Extracts advanced audio features from an audio file."""
    try:
        # Debugging: Log the start of the process
        logger.debug(f"Loading audio file: {file_path}")
        
        # Load the audio file using librosa
        audio, sr = librosa.load(file_path, sr=None)  # sr=None keeps the original sampling rate
        logger.debug(f"Audio loaded successfully with sample rate: {sr}")  # You can see this in terminal
        
        # Extract duration
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"Duration: {duration}")  # Debugging output for duration
        
        # Extract sample width using pydub
        sample_width = AudioSegment.from_file(file_path).sample_width  # Sample width in bytes
        print(f"Sample width: {sample_width}")
        
        # Extract advanced features using librosa
        
        # MFCC (Mel-frequency cepstral coefficients) features - 20 coefficients
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)  # Mean MFCC coefficients for each band
        print(f"MFCC mean extracted: {mfcc_mean.shape}")  # Check MFCC shape
        
        # Spectral Centroid - Describes the "center of mass" of the spectrum
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        print(f"Spectral centroid mean: {spectral_centroid_mean}")
        
        # Spectral Bandwidth - Width of the spectral band
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        print(f"Spectral bandwidth mean: {spectral_bandwidth_mean}")
        
        # Zero Crossing Rate - Rate at which the signal changes sign
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        print(f"Zero crossing rate mean: {zero_crossing_rate_mean}")
        
        # Chroma - Related to the harmonic and chordal properties of the signal
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # Average chroma values across frames
        print(f"Chroma mean: {chroma_mean}")
        
        # Return all extracted features as a flat list
        return np.array([
            duration, 
            sample_width, 
            *mfcc_mean,  # Add MFCC means to the feature list
            spectral_centroid_mean, 
            spectral_bandwidth_mean, 
            zero_crossing_rate_mean, 
            *chroma_mean  # Add chroma means to the feature list
        ])

    except Exception as e:
        # Catch any error during the process and log it
        logger.error(f"Error loading or processing audio {file_path}: {e}")
        return None  # Return None to indicate an error

def generate_dataset(uploaded_song_path=None):
    """Generates a dataset from the audio files by extracting relevant features."""
    data = []
    file_names = []
    
    # If there's an uploaded song, process only it, otherwise process all audio files in the folder
    if uploaded_song_path:
        # Process single uploaded song
        features = get_audio_features(uploaded_song_path)
        if features is not None and np.any(features):  # Or `any(features)` for lists
            data.append(features.tolist())  # Convert features array to a list
            file_names.append(os.path.basename(uploaded_song_path))
    else:
        # Process all files in the folder using concurrent threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            # Iterate over all files in the directory
            for file_name in os.listdir(AUDIO_FILES_PATH):
                file_path = os.path.join(AUDIO_FILES_PATH, file_name)

                # Process only audio files (add more extensions if needed)
                if file_name.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    futures.append(executor.submit(process_file, file_path, file_name, data, file_names))
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)

    # Define column names for all features, creating one column for features
    column_names = ['file_name', 'features']
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=column_names)

    # Check and clean column names if necessary
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    print("Columns in dataset:", df.columns)  # Print columns to check

    # Save the dataset to a CSV file
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    if os.path.exists(DATASET_PATH):
        existing_dataset = pd.read_csv(DATASET_PATH)
        df = pd.concat([existing_dataset, df], ignore_index=True)
        
    df.to_csv(DATASET_PATH, index=False, header=True)
    logger.info(f"Dataset saved to {DATASET_PATH}")

def process_file(file_path, file_name, data, file_names):
    """Processes a single file and extracts its features."""
    features = get_audio_features(file_path)
    if features:
        # Add the features and file name to the data list
        data.append([file_name, features.tolist()])  # Store features as a list
        file_names.append(file_name)

def load_dataset():
    """Loads the dataset and returns file names and features."""
    try:
        dataset = pd.read_csv(DATASET_PATH)
        dataset.columns = dataset.columns.str.strip()
        file_names = dataset['file_name']
        # Convert string representation of lists back to actual lists
        dataset['features'] = dataset['features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        dataset_features = np.vstack(dataset['features'].values)
        return file_names, dataset_features
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None, None

    return file_names, dataset_features

def recommend_similar_songs(uploaded_song_features, dataset_features, file_names, top_n=5):
    """Recommends similar songs based on cosine similarity of the features."""
    # Reshape dataset_features to 2D (each song's features as a row)
    features_array = dataset_features

    # Reshape uploaded_song_features to be compatible for cosine similarity calculation
    uploaded_song_features = np.array(uploaded_song_features).reshape(1, -1)

    # Calculate cosine similarity between the uploaded song and all songs in the dataset
    similarities = cosine_similarity(uploaded_song_features, features_array)

    # Get the indices of the top_n most similar songs
    similar_indices = similarities[0].argsort()[-top_n:][::-1]

    # Get the file names of the most similar songs
    similar_songs = file_names.iloc[similar_indices]

    return similar_songs

def get_uploaded_song_features(file_path):
    """Extracts features from the uploaded song for recommendation."""
    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        logger.error(f"Error: The file '{file_path}' does not exist.")
        return None
    
    # Get features (duration and sample_width) for the uploaded song
    features = get_audio_features(file_path)
    if features:
        return features[:2]  # Extract only the duration and sample_width
    else:
        return None

# Main execution
if __name__ == "__main__":
    # Step 1: Generate the dataset
    generate_dataset()  # To process all songs

    # Step 2: Load the dataset
    file_names, dataset_features = load_dataset()

    if file_names is not None and dataset_features is not None:
        print("Dataset loaded successfully.")

        # Step 3: Simulate uploading a song for recommendation
        uploaded_song_path = r"C:\Users\nayan\Downloads\music_recommendation_app\songs\your_uploaded_song.mp3"  # Replace with actual uploaded song path
        uploaded_song_features = get_uploaded_song_features(uploaded_song_path)

        if uploaded_song_features:
            similar_songs = recommend_similar_songs(uploaded_song_features, dataset_features, file_names)
            print("Recommended songs:", similar_songs)
        else:
            logger.error("Failed to extract features from the uploaded song.")
    else:
        logger.error("Failed to load dataset.")
