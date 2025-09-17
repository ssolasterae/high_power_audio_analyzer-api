
from flask import Flask, request, jsonify
import librosa
import numpy as np
import soundfile as sf

app = Flask(__name__)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    audio_file = request.files['audio']

    # Load audio file using librosa
    y, sr = librosa.load(audio_file, sr=None)

    # Start building the feature dictionary
    features = {}

    # Duration
    features['duration'] = float(librosa.get_duration(y=y, sr=sr))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo)

    # Energy and Zero-Crossing
    features['rmse_mean'] = float(np.mean(librosa.feature.rms(y=y)))
    features['zero_crossing_rate_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spectral features
    features['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['spectral_bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features['spectral_flatness_mean'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # Harmonic & Percussive
    harmonic, percussive = librosa.effects.hpss(y)
    features['harmonic_mean'] = float(np.mean(harmonic))
    features['percussive_mean'] = float(np.mean(percussive))

    # Chroma features
    features['chroma_stft_mean'] = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    features['chroma_cqt_mean'] = float(np.mean(librosa.feature.chroma_cqt(y=y, sr=sr)))
    features['chroma_cens_mean'] = float(np.mean(librosa.feature.chroma_cens(y=y, sr=sr)))

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features['tonnetz_mean'] = float(np.mean(tonnetz))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i-1]))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_spectrogram_mean'] = float(np.mean(mel))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = float(np.mean(contrast))

    # Polynomial Features
    poly = librosa.feature.poly_features(y=y, sr=sr)
    features['poly_features_mean'] = float(np.mean(poly))

    return jsonify(features)

if __name__ == '__main__':
    app.run(debug=True)
