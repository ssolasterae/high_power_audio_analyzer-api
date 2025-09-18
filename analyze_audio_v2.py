from flask import Flask, request, jsonify
import librosa
import numpy as np
import tempfile
import os

app = Flask(__name__)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        filepath = tmp.name
        file.save(filepath)

    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=None, mono=True)

        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = mfccs.mean(axis=1).tolist()

        return jsonify({
            "tempo": float(tempo),
            "chroma_mean": float(chroma),
            "spectral_centroid_mean": float(spectral_centroid),
            "rms_mean": float(rms),
            "zcr_mean": float(zcr),
            "spectral_bandwidth_mean": float(spectral_bandwidth),
            "spectral_rolloff_mean": float(spectral_rolloff),
            **{f"mfcc_{i+1}_mean": float(val) for i, val in enumerate(mfcc_means)}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temp file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
