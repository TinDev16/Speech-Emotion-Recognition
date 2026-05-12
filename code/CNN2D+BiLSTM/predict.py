import sys
import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model


# =============================
# CONFIG (PHẢI GIỐNG TRAIN)
# =============================
SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130


# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(path):
    try:
        y, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            offset=OFFSET
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel = librosa.power_to_db(mel)

        # pad / cut
        if mel.shape[1] < MAX_LEN:
            mel = np.pad(
                mel,
                ((0, 0), (0, MAX_LEN - mel.shape[1]))
            )
        else:
            mel = mel[:, :MAX_LEN]

        return mel.T

    except Exception as e:
        print(" Error loading audio:", e)
        return None


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    # check argument
    if len(sys.argv) < 2:
        print("Usage:")
        print("python predict.py your_audio.wav")
        sys.exit()

    audio_path = sys.argv[1]

    # check file tồn tại
    if not os.path.isfile(audio_path):
        print(" Audio file not found")
        sys.exit()

    print(" Loading model...")

    model = load_model("ser_best.keras")
    labels = np.load("labels.npy", allow_pickle=True)

    feat = extract_features(audio_path)

    if feat is None:
        sys.exit()

    # shape:
    # (1, time, freq, 1)
    X = np.expand_dims(feat, axis=0)
    X = np.expand_dims(X, axis=-1)

    preds = model.predict(X, verbose=0)[0]

    idx = np.argmax(preds)

    print("\n====================")
    print(" Emotion :", labels[idx])
    print(f" Confidence : {np.max(preds)*100:.2f}%")
    print("====================")

    # show all probabilities
    print("\n All emotions:")
    for label, prob in zip(labels, preds):
        print(f"{label:<10}: {prob*100:.2f}%")
