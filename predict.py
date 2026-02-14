# =====================================================
# SPEECH EMOTION RECOGNITION - PREDICT (FIXED)
# =====================================================

import sys
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

import joblib
from tensorflow.keras.models import load_model

# =============================
# PARAMETERS (PHẢI GIỐNG TRAIN)
# =============================
SAMPLE_RATE = 22050
DURATION = 3
OFFSET = 0.5
MAX_LEN = 130

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(path):
    y, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        offset=OFFSET
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)

    features = np.vstack([mfcc, chroma, mel]).T

    if features.shape[0] < MAX_LEN:
        features = np.pad(
            features,
            ((0, MAX_LEN - features.shape[0]), (0, 0))
        )
    else:
        features = features[:MAX_LEN, :]

    return features


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py path_to_audio.wav")
        sys.exit()

    audio_path = sys.argv[1]

    print("🔄 Loading model, labels & scaler...")
    model = load_model("ser_bilstm_improved.h5")
    labels = np.load("labels.npy", allow_pickle=True)
    scaler = joblib.load("scaler.save")   # ✅ LOAD SCALER ĐÃ TRAIN

    # Extract features
    features = extract_features(audio_path)
    X = np.expand_dims(features, axis=0)

    # ✅ CHỈ TRANSFORM – KHÔNG FIT
    X = scaler.transform(
        X.reshape(1, -1)
    ).reshape(X.shape)

    # Predict
    preds = model.predict(X, verbose=0)
    pred_index = np.argmax(preds)
    pred_label = labels[pred_index]
    confidence = preds[0][pred_index] * 100

    print("\n🎤 Audio:", audio_path)
    print("🎯 Emotion:", pred_label)
    print(f"📊 Confidence: {confidence:.2f}%")

    print("\n🔢 Full probabilities:")
    for i, label in enumerate(labels):
        print(f"{label:10s}: {preds[0][i]*100:.2f}%")
