# =====================================================
# SPEECH EMOTION RECOGNITION - PREDICT FILE
# =====================================================

import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# =====================
# PATHS
# =====================
MODEL_PATH = "ser_final.keras"
LABEL_PATH = "labels.npy"
SCALER_PATH = "scaler.save"

TEST_FILE = "test.wav"   # 🔴 đổi thành file m muốn test

# =====================
# AUDIO PARAMS (PHẢI GIỐNG TRAIN)
# =====================
SR = 22050
DURATION = 3
OFFSET = 0.5
MAX_LEN = 130

# =====================
# LOAD MODEL & TOOLS
# =====================
model = load_model(MODEL_PATH)
labels = np.load(LABEL_PATH, allow_pickle=True)
scaler = joblib.load(SCALER_PATH)

# =====================
# FEATURE EXTRACTION
# =====================
def extract_features(path):
    y, sr = librosa.load(
        path,
        sr=SR,
        duration=DURATION,
        offset=OFFSET
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)

    feat = np.vstack([mfcc, chroma, mel]).T  # (time, features)

    if feat.shape[0] < MAX_LEN:
        feat = np.pad(
            feat,
            ((0, MAX_LEN - feat.shape[0]), (0, 0))
        )
    else:
        feat = feat[:MAX_LEN]

    return feat

# =====================
# PREDICT
# =====================
def predict_emotion(path):
    x = extract_features(path)

    # scale giống lúc train
    x = scaler.transform(x.reshape(1, -1)).reshape(1, MAX_LEN, -1)

    pred = model.predict(x, verbose=0)[0]

    idx = np.argmax(pred)
    emotion = labels[idx]

    return emotion, pred

# =====================
# RUN
# =====================
if __name__ == "__main__":
    emotion, probs = predict_emotion(TEST_FILE)

    print("🎧 File:", TEST_FILE)
    print("🎯 Emotion:", emotion)
    print("\n📊 Probabilities:")
    for lbl, p in zip(labels, probs):
        print(f"{lbl:10s}: {p:.3f}")
