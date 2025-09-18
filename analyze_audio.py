from flask import Flask, request, jsonify
import librosa
import numpy as np
import tempfile

app = Flask(__name__)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    # --- Make sure a file was uploaded ---
    if not request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # --- Grab the first uploaded file (robust to any field name) ---
    audio_file = next(iter(request.files.values()))

    # --- Save to temporary file so librosa can load it ---
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        audio_file.save(tmp.name)
        try:
            y, sr = librosa.load(tmp.name, sr=None)
        except Exception as e:
            return jsonify({"error": f"Could not process file: {str(e)}"}), 500

    # --- Extract audio features ---
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rmse = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = {f"mfcc_{i+1}_mean": float(np.mean(mfccs[i])) for i in range(20)}

        features = {
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "tempo": float(tempo),
            "rmse_mean": float(rmse),
            "zero_crossing_rate_mean": float(zcr),
            "spectral_centroid_mean": float(spectral_centroid),
            "spectral_bandwidth_mean": float(spectral_bandwidth),
            "spectral_rolloff_mean": float(spectral_rolloff),
            "spectral_flatness_mean": float(spectral_flatness),
        }
        features.update(mfcc_means)

        return jsonify(features)

    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
