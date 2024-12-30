from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import logging

AUDIO_FILES_PATH = "songs"
DATASET_PATH = os.path.join("data", "dataset.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLUMN_NAMES = (
    ['file_name', 'sample_width'] +
    [f'mfcc_mean_{i+1}' for i in range(20)] +
    ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'zero_crossing_rate_mean'] +
    [f'chroma_mean_{i+1}' for i in range(12)]
)

def get_audio_features(file_path):
    """Extracts audio features from audio file."""

    try:
        logger.debug(f"Loading audio file: {file_path}")

        audio, sr = librosa.load(file_path, sr=None)
        sample_width = AudioSegment.from_file(file_path).sample_width

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Combine features into an array
        features = np.array([
            sample_width, *mfcc_mean,
            spectral_centroid_mean, spectral_bandwidth_mean, zero_crossing_rate_mean, *chroma_mean
        ])
        return features

    except Exception as e:
        logger.error(f"Error processing audio {file_path}: {e}")
        return None

def generate_dataset(uploaded_song_path=None):
    """Generates dataset from audio files by extracting features."""

    data = []
    if uploaded_song_path:
        features = get_audio_features(uploaded_song_path)
        if features is not None:
            data.append([os.path.basename(uploaded_song_path)] + features.tolist())
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for file_name in os.listdir(AUDIO_FILES_PATH):
                file_path = os.path.join(AUDIO_FILES_PATH, file_name)
                futures.append(executor.submit(process_file, file_path, data))
            concurrent.futures.wait(futures)

    df = pd.DataFrame(data, columns=COLUMN_NAMES)

    df = df.drop_duplicates(subset='file_name', keep='first')

    # Standardize the features (ignore file_name column)
    features_df = df.drop(columns=['file_name'], errors='ignore')
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_df)

    standardized_df = pd.DataFrame(standardized_features, columns=features_df.columns)
    standardized_df['file_name'] = df['file_name']

    folder = os.path.dirname(DATASET_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if os.path.exists(DATASET_PATH):
        try:
            existing_dataset = pd.read_csv(DATASET_PATH)
            df = pd.concat([existing_dataset, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    df = df.drop_duplicates(subset='file_name', keep='first')
    df.to_csv(DATASET_PATH, index=False, header=True)
    logger.info(f"Dataset saved to {DATASET_PATH}")

def process_file(file_path, data):
    features = get_audio_features(file_path)
    if features is not None:
        data.append([os.path.basename(file_path)] + features.tolist())

def load_dataset():
    """Loads the dataset and returns file names and features."""

    try:
        dataset = pd.read_csv(DATASET_PATH)
        file_names = dataset['file_name']
        features = dataset.iloc[:, 1:].to_numpy()
        return file_names, features
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None, None