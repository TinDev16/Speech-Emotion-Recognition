import sys
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model


SAMPLE_RATE = 22050
DURATION = 4
OFFSET = 0.5
MAX_LEN = 130


def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE,
                         duration=DURATION, offset=OFFSET)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0,0),(0, MAX_LEN-mel.shape[1])))
    else:
        mel = mel[:, :MAX_LEN]

    return mel.T  # (time, 128)


if __name__ == "__main__":

    audio_path = sys.argv[1]

    model = load_model("ser_best.keras")
    labels = np.load("labels.npy", allow_pickle=True)

    feat = extract_features(audio_path)

    X = np.expand_dims(feat, axis=0)
    X = np.expand_dims(X, axis=-1)   # (1, time, freq, 1)

    preds = model.predict(X, verbose=0)

    idx = np.argmax(preds)
    print("Emotion:", labels[idx])
    print("Confidence:", float(np.max(preds)) * 100)